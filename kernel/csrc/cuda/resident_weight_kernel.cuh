// Experimental one-CTA-per-small-problem resident optimizer.
// Include at global scope after scoring.cuh. The optional native entry point
// keeps default dispatch disabled pending whole-fit qualification.
#pragma once
#include "resident_weights.cuh"

namespace {
using ResidentWeightState=clipp2::resident_slsqp::WeightState<>;
using ResidentWeightOperation=clipp2::resident_slsqp::WeightOperation;
using ResidentWeightError=clipp2::resident_slsqp::WeightError;

struct ResidentWeightItem {
    WeightBatchItem problem;
    ResidentWeightState* state;
    double* arena;
    int* indices;
    const double* initial;
    i64 full_chunks=0,tail_count=0,leaf_count=0,chunks=0;
    const ScoringLeaf* tail=nullptr;
    double *leaf_partials=nullptr,*chunk_partials=nullptr;
    double endpoint=0.,grow_weight=0.,amount=0.;
};

__device__ void resident_advance(ResidentWeightState& state) {
    const auto start=clock64();
    state.advance();
    state.controller_cycles+=clock64()-start;
    ++state.controller_calls;
}

__global__ void resident_weight_small_kernel(const ResidentWeightItem* items,i64 count) {
    __shared__ double terms[small_mixture_rows*(small_mixture_columns+2)];
    __shared__ double endpoint,grow_weight,amount;
    const i64 row=threadIdx.x;
    for(i64 index=blockIdx.x;index<count;index+=gridDim.x) {
        const auto& item=items[index];const auto& p=item.problem;auto& state=*item.state;
        if(!row) {
            state=ResidentWeightState{};
            if(p.n<1||p.n>small_mixture_rows||p.q<1||p.q>small_mixture_columns||
                    !item.arena||!item.indices||!item.initial||blockDim.x!=small_mixture_rows) {
                state.fail(ResidentWeightError::Contract);
            } else {
                state.bind(static_cast<int>(p.n),static_cast<int>(p.q),item.arena,item.indices);
                ResidentWeightState::copy(state.weights,item.initial,static_cast<int>(p.q));
                resident_advance(state);
            }
        }
        __syncthreads();
        while(state.operation!=ResidentWeightOperation::Complete&&state.operation!=ResidentWeightOperation::Error) {
            if(state.operation==ResidentWeightOperation::Evaluate) {
                if(row<p.n)mixture_evaluate_row(p.columns,p.offsets,p.kernel,state.request_weights,p.n,p.q,row,terms);
                __syncthreads();
                if(!row) {
                    state.cache_objective=-small_mixture_sum(terms,p.n)/p.n;
                    state.cache_likelihood=small_mixture_sum(terms+p.n,p.n);
                }
                if(row<p.q)state.cache_gradient[row]=-small_mixture_sum(terms+(row+2)*p.n,p.n)/p.n;
            } else if(state.operation==ResidentWeightOperation::Prepare) {
                if(!row) { grow_weight=state.request_weights[state.grow];amount=state.request_weights[state.shrink]; }
                if(row<p.q)p.logs[row]=log(state.request_weights[row]);
                __syncthreads();
                if(row<p.n) {
                    mixture_pair_row(p.columns,p.offsets,p.logs,p.q,row,state.grow,state.shrink,
                                     p.other,p.grow_log,p.shrink_log,p.numerator,p.sign);
                    terms[row]=mixture_trial_row(row,p.other,p.grow_log,p.shrink_log,p.numerator,p.sign,
                                                log(grow_weight+1.*amount),log(amount*(1.-1.)));
                }
                __syncthreads();
                if(!row)endpoint=small_mixture_sum(terms,p.n)/p.n;
            } else if(state.operation==ResidentWeightOperation::Derivative) {
                if(state.fraction==1.) { if(!row)state.derivative=endpoint; }
                else {
                    const double a=log(grow_weight+state.fraction*amount),b=log(amount*(1.-state.fraction));
                    if(row<p.n)terms[row]=mixture_trial_row(row,p.other,p.grow_log,p.shrink_log,p.numerator,p.sign,a,b);
                    __syncthreads();
                    if(!row)state.derivative=small_mixture_sum(terms,p.n)/p.n;
                }
            }
            __syncthreads();
            if(!row)resident_advance(state);
            __syncthreads();
        }
        __syncthreads();
    }
}
} // namespace
