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
        state.advance();
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
    if(active) {
    const double a=evaluation?0.:log(item->grow_weight+fraction*item->amount);
    const double b=evaluation?0.:log(item->amount*(1.-fraction));
    for(i64 leaf=blockIdx.x;leaf<item->leaf_count;leaf+=gridDim.x) {
        const auto range=scoring_leaf(leaf,item->full_chunks,item->tail);
        if(evaluation) {
            scoring_evaluate_leaf(p.columns,p.offsets,p.kernel,state.request_weights,p.n,p.q,range,
                                  leaf,item->leaf_count,item->leaf_partials,values,chains,masses);
        } else {
            scoring_trial_leaf(range,leaf,item->leaf_count,p.other,p.grow_log,p.shrink_log,
                p.numerator,p.sign,a,b,item->leaf_partials,values,chains);
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
        scoring_combine_chunk(item->leaf_partials,p.n,item->chunks,item->full_chunks,item->tail,
                              item->tail_count,item->leaf_count,job,item->chunk_partials,values);
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
        const double result=scoring_finish_plane(item->chunk_partials,item->chunks,plane);
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
    if(!resident_terminal(state))state.advance();
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
