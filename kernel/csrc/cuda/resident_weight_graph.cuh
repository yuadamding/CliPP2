// Experimental fixed-chunk resident optimizer. Ordinary CUDA graphs execute
// 32 ordered request waves; the host reads only terminal flags between chunks.
// No objective, gradient, weight, or SLSQP state is delivered before completion.
// Include globally after scoring.cuh. Production dispatch remains opt-in.
#pragma once
#include "resident_weight_kernel.cuh"

namespace {
__device__ bool resident_terminal(const ResidentWeightState& state) {
    return state.operation==ResidentWeightOperation::Complete||state.operation==ResidentWeightOperation::Error;
}

__global__ void resident_graph_initialize(ResidentWeightItem* items) {
    if(threadIdx.x)return;
    auto& item=items[blockIdx.x];auto& state=*item.state;const auto& p=item.problem;
    state=ResidentWeightState{};
    if(!ResidentWeightState::valid_dimensions(p.n,p.q)||!item.arena||!item.indices||!item.initial)
        state.fail(ResidentWeightError::Contract);
    else {
        state.bind(static_cast<int>(p.n),static_cast<int>(p.q),item.arena,item.indices);
        ResidentWeightState::copy(state.weights,item.initial,static_cast<int>(p.q));
        resident_advance(state);
    }
}
__global__ void resident_graph_logs(ResidentWeightItem* items) {
    auto* item=items+blockIdx.y;
    const auto& state=*item->state;auto& p=item->problem;
    if(state.operation!=ResidentWeightOperation::Prepare)return;
    for(i64 k=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;k<p.q;k+=static_cast<i64>(gridDim.x)*blockDim.x)
        p.logs[k]=log(state.request_weights[k]);
    if(!blockIdx.x&&!threadIdx.x) {
        item->grow_weight=state.request_weights[state.grow];item->amount=state.request_weights[state.shrink];
    }
}
__global__ void resident_graph_pair(ResidentWeightItem* items) {
    auto* item=items+blockIdx.y;
    const auto& state=*item->state;auto& p=item->problem;
    if(state.operation!=ResidentWeightOperation::Prepare)return;
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<p.n;row+=static_cast<i64>(gridDim.x)*blockDim.x)
        mixture_pair_row(p.columns,p.offsets,p.logs,p.q,row,state.grow,state.shrink,
                         p.other,p.grow_log,p.shrink_log,p.numerator,p.sign);
}
__global__ void resident_graph_leaves(ResidentWeightItem* items) {
    auto* item=items+blockIdx.y;
    const auto& state=*item->state;const auto& p=item->problem;
    __shared__ double values[scoring_plane_tile*128],chains[128],masses[128];
    __shared__ bool active,evaluation;
    __shared__ double fraction;
    if(!threadIdx.x) {
        const auto operation=state.operation;
        active=!resident_terminal(state)&&!(operation==ResidentWeightOperation::Derivative&&state.fraction==1.);
        evaluation=operation==ResidentWeightOperation::Evaluate;
        fraction=operation==ResidentWeightOperation::Prepare?1.:state.fraction;
    }
    __syncthreads();
    const i64 row=threadIdx.x;
    if(active) {
    const double a=evaluation?0.:log(item->grow_weight+fraction*item->amount);
    const double b=evaluation?0.:log(item->amount*(1.-fraction));
    for(i64 leaf=blockIdx.x;leaf<item->leaf_count;leaf+=gridDim.x) {
        const auto range=scoring_leaf(leaf,item->full_chunks,item->tail);
        if(evaluation) {
            if(row<range.count)masses[row]=mixture_row_log_mass(p.columns,p.offsets,p.kernel,
                state.request_weights,p.n,p.q,range.first+row);
            __syncthreads();
            for(i64 first=0;first<p.q+2;first+=scoring_plane_tile) {
                const i64 count=p.q+2-first<scoring_plane_tile?p.q+2-first:scoring_plane_tile;
                if(row<range.count) {
                    const i64 source=range.first+row;const double offset=p.offsets[source],mass=masses[row];
                    for(i64 plane=0;plane<count;++plane) {
                        const i64 k=first+plane;
                        values[plane*128+row]=k==0?mass:(k==1?offset+mass:
                            scoring_capped_exp(p.columns[k-2][source]-offset-mass));
                    }
                }
                __syncthreads();
                scoring_leaf_reduce(values,range.count,first,count,leaf,item->leaf_count,chains,item->leaf_partials);
            }
        } else {
            if(row<range.count)values[row]=mixture_trial_row(range.first+row,p.other,p.grow_log,
                p.shrink_log,p.numerator,p.sign,a,b);
            __syncthreads();
            scoring_leaf_reduce(values,range.count,0,1,leaf,item->leaf_count,chains,item->leaf_partials);
        }
    }
    }
    // Replayed graph nodes reuse shared scratch. Keep explicit rendezvous
    // outside the tile loops, including terminal no-op requests.
    __syncthreads();
}
__global__ void resident_graph_combine(ResidentWeightItem* items) {
    auto* item=items+blockIdx.y;
    const auto& state=*item->state;const auto& p=item->problem;
    __shared__ double values[128];
    __shared__ bool active;
    __shared__ i64 planes;
    if(!threadIdx.x) {
        active=!resident_terminal(state)&&!(state.operation==ResidentWeightOperation::Derivative&&state.fraction==1.);
        planes=state.operation==ResidentWeightOperation::Evaluate?p.q+2:1;
    }
    __syncthreads();
    if(active) {
    for(i64 job=blockIdx.x;job<planes*item->chunks;job+=gridDim.x) {
        const i64 plane=job/item->chunks,chunk=job%item->chunks;
        const bool full=chunk<item->full_chunks;
        const i64 count=full?8192:p.n%8192,first=chunk*64;
        if(threadIdx.x<(full?64:item->tail_count)) {
            const i64 slot=full?2*threadIdx.x:item->tail[threadIdx.x].slot;
            values[slot]=item->leaf_partials[plane*item->leaf_count+first+threadIdx.x];
        }
        __syncthreads();
        int deepest=0;
        for(i64 maximum=count;maximum>128;++deepest) { i64 split=maximum/2;split-=split%8;maximum-=split; }
        for(int depth=deepest-1;depth>=0;--depth) {
            const int nodes=1<<depth;
            for(int node=threadIdx.x;node<nodes;node+=blockDim.x) {
                i64 offset,size;
                if(ordered_tree_node(count,depth,node,offset,size)&&size>128) {
                    const int left=node<<(7-depth),right=left+(1<<(6-depth));
                    values[left]=values[left]+values[right];
                }
            }
            __syncthreads();
        }
        if(!threadIdx.x) { double sum=-0.;sum+=values[0];item->chunk_partials[job]=sum; }
        __syncthreads();
    }
    }
    __syncthreads();
}
__global__ void resident_graph_reply(ResidentWeightItem* items) {
    auto* item=items+blockIdx.y;
    auto& state=*item->state;const auto& p=item->problem;
    if(resident_terminal(state))return;
    if(state.operation==ResidentWeightOperation::Derivative&&state.fraction==1.) {
        if(!blockIdx.x&&!threadIdx.x)state.derivative=item->endpoint;
        return;
    }
    const i64 planes=state.operation==ResidentWeightOperation::Evaluate?p.q+2:1;
    for(i64 plane=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;plane<planes;
        plane+=static_cast<i64>(gridDim.x)*blockDim.x) {
        double result=-0.;for(i64 chunk=0;chunk<item->chunks;++chunk)result+=item->chunk_partials[plane*item->chunks+chunk];
        if(state.operation==ResidentWeightOperation::Evaluate) {
            if(plane==0)state.cache_objective=-result/p.n;
            else if(plane==1)state.cache_likelihood=result;
            else state.cache_gradient[plane-2]=-result/p.n;
        } else if(state.operation==ResidentWeightOperation::Prepare)item->endpoint=result/p.n;
        else state.derivative=result/p.n;
    }
}

__global__ void resident_graph_advance(ResidentWeightItem* items) {
    if(threadIdx.x)return;
    auto& state=*items[blockIdx.x].state;
    if(!resident_terminal(state))resident_advance(state);
}
__global__ void resident_graph_terminal_flags(const ResidentWeightItem* items,i64 count,int* flags) {
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
        index+=static_cast<i64>(gridDim.x)*blockDim.x)
        flags[index]=resident_terminal(*items[index].state)?1:0;
}

constexpr int resident_chunk_steps=32;
constexpr int resident_chunk_nodes=6*resident_chunk_steps;
// One fixed ordinary graph serves the complete batch. Every numerical state
// persists across graph launches; terminal states perform no numerical work.
// No device-side graph control, global stopping cap, or CPU fallback is used.
void add_resident_weight_graph(cudaGraph_t graph,ResidentWeightItem* device_items,
                               i64 count,i64 n,i64 q,i64 leaves,i64 chunks) {
    if(count<1||count>65535)throw std::domain_error("Resident batch exceeds CUDA grid.y capacity");
    void* arguments[]={&device_items};
    cudaGraphNode_t prior=nullptr;
    auto kernel=[&](void* function,unsigned grid_x,unsigned grid_y) {
        cudaKernelNodeParams parameters{};
        parameters.func=function;parameters.gridDim=dim3(grid_x,grid_y);
        parameters.blockDim=dim3(threads);parameters.kernelParams=arguments;
        cudaGraphNode_t node{};
        cuda_check(cudaGraphAddKernelNode(&node,graph,prior?&prior:nullptr,prior?1:0,&parameters),
                   "resident chunk kernel");
        prior=node;
    };
    for(int step=0;step<resident_chunk_steps;++step) {
        kernel(reinterpret_cast<void*>(resident_graph_logs),blocks(q),static_cast<unsigned>(count));
        kernel(reinterpret_cast<void*>(resident_graph_pair),blocks(n),static_cast<unsigned>(count));
        kernel(reinterpret_cast<void*>(resident_graph_leaves),static_cast<unsigned>(std::min<i64>(65535,leaves)),
               static_cast<unsigned>(count));
        kernel(reinterpret_cast<void*>(resident_graph_combine),
               static_cast<unsigned>(std::min<i64>(65535,multiply(q+2,chunks))),static_cast<unsigned>(count));
        kernel(reinterpret_cast<void*>(resident_graph_reply),blocks(q+2),static_cast<unsigned>(count));
        kernel(reinterpret_cast<void*>(resident_graph_advance),static_cast<unsigned>(count),1);
    }
}
} // namespace
