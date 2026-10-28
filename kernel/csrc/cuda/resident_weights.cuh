// Experimental device-resident weight-controller state machine; opt-in only.
// SLSQP/Brent orchestration follows kernel/weight_controller.py and its pinned
// SciPy 1.18.0 sources. See resident_slsqp.cuh for the retained BSD-3-Clause
// copyright, conditions and disclaimer governing this adaptation.
// One owner advances control; an evaluator services requests without requiring
// host access to intermediate objective, gradient, weights, or optimizer state.
#pragma once
#include "resident_slsqp.cuh"
#include "resident_blas.cuh"
#include <limits.h>

#ifdef __CUDACC__
#define CLIPP2_WEIGHT_FN __device__
#define CLIPP2_WEIGHT_SIZE_FN __host__ __device__
#else
#define CLIPP2_WEIGHT_FN
#define CLIPP2_WEIGHT_SIZE_FN
#endif

namespace clipp2::resident_slsqp {
enum class WeightOperation : int { Evaluate=0, Prepare=1, Derivative=2, Complete=3, Error=4 };
enum class WeightError : int { None=0, InvalidStart=1, Arithmetic=2, Convergence=3,
                               BrentSign=4, BrentNaN=5, BrentIterations=6, Contract=7 };
enum class WeightPhase : int {
    Start, FirstValue, CheckGradient, SlsqpInitial, SlsqpStep, SlsqpFunction,
    SlsqpGradient, CandidateValue, CandidateCompare, BoundaryValue,
    BoundaryCompare, Repair, RepairGradient, Prepared, Endpoint, BrentStart,
    BrentFirst, BrentSecond, BrentLoop, BrentValue, Transfer, Final, FinalValue, Done
};

template<class Blas=DeviceBlas> struct WeightState {
    // All pointed storage belongs to one caller-owned, budgeted problem arena.
    // doubles(q) returns the exact count, excluding this struct and evaluator.
    int n=0,q=0,repair_steps=0,grow=0,shrink=0,brent_iteration=0;
    WeightPhase phase=WeightPhase::Start;
    WeightOperation operation=WeightOperation::Evaluate;
    WeightError error=WeightError::None;
    SLSQP_vars slsqp{};
    double *weights=nullptr,*boundary=nullptr,*candidate=nullptr,*request_weights=nullptr;
    double *cache_weights=nullptr,*cache_gradient=nullptr,*scalar_x=nullptr,*scalar_gradient=nullptr;
    double *gradient=nullptr,*normals=nullptr,*multipliers=nullptr,*lower=nullptr,*upper=nullptr,*buffer=nullptr;
    int* indices=nullptr;
    double cache_objective=0.,cache_likelihood=0.,scalar_objective=0.,fx=0.,constraint=0.;
    double candidate_objective=0.,boundary_objective=0.,optimality_gap=0.,active_gap=0.;
    double fraction=0.,derivative=0.,xpre=0.,xcur=0.,xblk=0.,fpre=0.,fcur=0.,fblk=0.,spre=0.,scur=0.;
    bool cache_valid=false,waiting=false,scalar_f_valid=false,scalar_g_valid=false;
    int64_t evaluations=0,cache_hits=0,preparations=0,trials=0;
    // Experimental diagnostics only: no stopping or numerical decision uses
    // these device-clock counts. They are elapsed owner-thread cycles, not
    // FLOP counts or an assumed fixed-frequency wall clock.
    uint64_t controller_cycles=0,slsqp_cycles=0,controller_calls=0;

    CLIPP2_WEIGHT_SIZE_FN static int64_t buffer_count(int count) {
        return int64_t(count)*(count+1)/2+10*int64_t(count)*count+35*int64_t(count)+30;
    }
    CLIPP2_WEIGHT_SIZE_FN static int64_t doubles(int count) { return buffer_count(count)+14*int64_t(count)+3; }
    CLIPP2_WEIGHT_SIZE_FN static bool valid_dimensions(int64_t rows,int64_t count) {
        // Pinned LP64 core uses int workspace offsets, not just int BLAS
        // dimensions. Bound the complete arena before any int arithmetic.
        return rows>0&&rows<=INT_MAX&&count>0&&count<=46340&&doubles(static_cast<int>(count))<=INT_MAX;
    }
    CLIPP2_WEIGHT_FN void bind(int rows,int count,double* storage,int* index_storage) {
        n=rows;q=count;
        weights=storage; boundary=weights+q; candidate=boundary+q; request_weights=candidate+q;
        cache_weights=request_weights+q; cache_gradient=cache_weights+q;
        scalar_x=cache_gradient+q; scalar_gradient=scalar_x+q;
        gradient=scalar_gradient+q; normals=gradient+q; multipliers=normals+q;
        lower=multipliers+2*q+3;upper=lower+q;buffer=upper+q;indices=index_storage;
    }
    CLIPP2_WEIGHT_FN static void copy(double* target,const double* source,int count) {
        for(int i=0;i<count;++i)target[i]=source[i];
    }
    CLIPP2_WEIGHT_FN static double pairwise_piece(const double* values,int count) {
        if(count<8) { double result=-0.;for(int i=0;i<count;++i)result+=values[i];return result; }
        if(count<=128) {
            double r[8];for(int j=0;j<8;++j)r[j]=values[j];
            int i=8;for(;i<count-(count%8);i+=8)for(int j=0;j<8;++j)r[j]+=values[i+j];
            double result=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
            for(;i<count;++i)result+=values[i];return result;
        }
        int split=count/2;split-=split%8;
        return pairwise_piece(values,split)+pairwise_piece(values+split,count-split);
    }
    CLIPP2_WEIGHT_FN static double pairwise_sum(const double* values,int count) {
        double result=-0.;
        for(int first=0;first<count;first+=8192) {
            const int remaining=count-first,chunk=remaining<8192?remaining:8192;
            result+=pairwise_piece(values+first,chunk);
        }
        return result;
    }
    CLIPP2_WEIGHT_FN static bool bytes_equal(const double* a,const double* b,int count) {
        const auto* aa=reinterpret_cast<const unsigned char*>(a);
        const auto* bb=reinterpret_cast<const unsigned char*>(b);
        for(int64_t i=0;i<8*int64_t(count);++i)if(aa[i]!=bb[i])return false;
        return true;
    }
    CLIPP2_WEIGHT_FN bool same_scalar_x() const {
        for(int i=0;i<q;++i)if(candidate[i]!=scalar_x[i])return false;
        return true;
    }
    CLIPP2_WEIGHT_FN void fail(WeightError code) {
        error=code;operation=WeightOperation::Error;phase=WeightPhase::Done;waiting=false;
    }
    CLIPP2_WEIGHT_FN void uniform() { for(int i=0;i<q;++i)weights[i]=1./q; }
    CLIPP2_WEIGHT_FN void normalize(double* values) {
        double total=pairwise_sum(values,q);for(int i=0;i<q;++i)values[i]/=total;
    }
    CLIPP2_WEIGHT_FN bool evaluate(const double* at,WeightPhase next) {
        phase=next;
        if(cache_valid&&bytes_equal(at,cache_weights,q)) { ++cache_hits;return false; }
        copy(request_weights,at,q);copy(cache_weights,at,q);cache_valid=true;
        operation=WeightOperation::Evaluate;waiting=true;++evaluations;return true;
    }
    // The evaluator writes cache_objective/cache_gradient/cache_likelihood for
    // Evaluate, and derivative for Derivative, before the next advance call.
    // Prepare's reply has no scalar value; its completed pair state is external.
    CLIPP2_WEIGHT_FN void request_derivative(double at,WeightPhase next) {
        fraction=at;phase=next;operation=WeightOperation::Derivative;waiting=true;++trials;
    }
    CLIPP2_WEIGHT_FN bool scalar_request(bool need_gradient) {
        if(!same_scalar_x()) {
            copy(scalar_x,candidate,q);scalar_f_valid=scalar_g_valid=false;
        }
        if(need_gradient) {
            if(!scalar_g_valid)return evaluate(scalar_x,WeightPhase::SlsqpGradient);
            copy(gradient,scalar_gradient,q);for(int i=0;i<q;++i)normals[i]=1.;
        } else {
            if(!scalar_f_valid)return evaluate(scalar_x,WeightPhase::SlsqpFunction);
            fx=scalar_objective;constraint=pairwise_sum(candidate,q)-1.;
        }
        phase=WeightPhase::SlsqpStep;return false;
    }
    CLIPP2_WEIGHT_FN void advance() {
        if(phase==WeightPhase::Done)return;
        waiting=false;
        for(;;)switch(phase) {
        case WeightPhase::Start: {
            if(!valid_dimensions(n,q)||!weights||!indices) { fail(WeightError::Contract);return; }
            double high=0.;
            for(int i=0;i<q;++i) {
                if(!isfinite(weights[i])||weights[i]<0.) { fail(WeightError::InvalidStart);return; }
                if(weights[i]>high)high=weights[i];
            }
            if(!(high>0.)) { fail(WeightError::InvalidStart);return; }
            for(int i=0;i<q;++i)weights[i]/=high;
            normalize(weights);
            if(evaluate(weights,WeightPhase::FirstValue))return;
            break;
        }
        case WeightPhase::FirstValue:
            if(q==1) { optimality_gap=active_gap=0.;operation=WeightOperation::Complete;phase=WeightPhase::Done;return; }
            if(!isfinite(cache_objective))uniform();
            copy(boundary,weights,q);
            if(evaluate(weights,WeightPhase::CheckGradient))return;
            break;
        case WeightPhase::CheckGradient: {
            double high=-INFINITY;for(int i=0;i<q;++i)if(-cache_gradient[i]>high)high=-cache_gradient[i];
            if(high>1e100)uniform();
            copy(candidate,weights,q);copy(scalar_x,candidate,q);
            if(evaluate(scalar_x,WeightPhase::SlsqpInitial))return;
            break;
        }
        case WeightPhase::SlsqpInitial:
            scalar_objective=fx=cache_objective;
            copy(scalar_gradient,cache_gradient,q);copy(gradient,cache_gradient,q);
            scalar_f_valid=scalar_g_valid=true;
            slsqp={};slsqp.acc=1e-13;slsqp.tol=1e-12;slsqp.itermax=2000;
            slsqp.m=slsqp.meq=1;slsqp.n=q;
            for(int64_t i=0;i<buffer_count(q);++i)buffer[i]=0.;
            for(int i=0;i<2*q+3;++i) { indices[i]=0;multipliers[i]=0.; }
            for(int i=0;i<q;++i) { lower[i]=0.;upper[i]=1.;normals[i]=1.; }
            constraint=pairwise_sum(candidate,q)-1.;phase=WeightPhase::SlsqpStep;break;
        case WeightPhase::SlsqpStep: {
            // Finite bounds and finite clamped iterates imply mg>=2q in LSQ,
            // including its finite-bounded inconsistent-slack augmentation.
            for(int i=0;i<q;++i)if(!isfinite(candidate[i])||lower[i]!=0.||upper[i]!=1.) {
                fail(WeightError::Contract);return;
            }
#ifdef __CUDA_ARCH__
            const auto slsqp_start=clock64();
#endif
            Core<Blas>::step(&slsqp,&fx,gradient,normals,&constraint,candidate,
                            multipliers,lower,upper,buffer,indices);
#ifdef __CUDA_ARCH__
            slsqp_cycles+=clock64()-slsqp_start;
#endif
            if(slsqp.mode==1||slsqp.mode==-1) {
                for(int i=0;i<q;++i) {
                    if(candidate[i]<0.)candidate[i]=0.;else if(candidate[i]>1.)candidate[i]=1.;
                }
                if(scalar_request(slsqp.mode==-1))return;
                break;
            }
            bool finite=true;for(int i=0;i<q;++i) {
                // np.maximum(fit.x, 0.) also canonicalizes negative zero.
                finite=finite&&isfinite(candidate[i]);if(candidate[i]<=0.)candidate[i]=0.;
            }
            if(finite&&pairwise_sum(candidate,q)>0.) {
                normalize(candidate);if(evaluate(candidate,WeightPhase::CandidateValue))return;
            } else if(evaluate(boundary,WeightPhase::BoundaryValue))return;
            break;
        }
        case WeightPhase::SlsqpFunction:
            scalar_objective=fx=cache_objective;scalar_f_valid=true;
            constraint=pairwise_sum(candidate,q)-1.;phase=WeightPhase::SlsqpStep;break;
        case WeightPhase::SlsqpGradient:
            copy(scalar_gradient,cache_gradient,q);copy(gradient,cache_gradient,q);scalar_g_valid=true;
            for(int i=0;i<q;++i)normals[i]=1.;phase=WeightPhase::SlsqpStep;break;
        case WeightPhase::CandidateValue:
            candidate_objective=cache_objective;
            if(evaluate(weights,WeightPhase::CandidateCompare))return;
            break;
        case WeightPhase::CandidateCompare:
            if(candidate_objective<=cache_objective)copy(weights,candidate,q);
            if(evaluate(boundary,WeightPhase::BoundaryValue))return;
            break;
        case WeightPhase::BoundaryValue:
            boundary_objective=cache_objective;
            if(evaluate(weights,WeightPhase::BoundaryCompare))return;
            break;
        case WeightPhase::BoundaryCompare:
            if(boundary_objective<cache_objective)copy(weights,boundary,q);
            phase=WeightPhase::Repair;break;
        case WeightPhase::Repair:
            if(repair_steps==10000) { phase=WeightPhase::Final;break; }
            if(evaluate(weights,WeightPhase::RepairGradient))return;
            break;
        case WeightPhase::RepairGradient: {
            double high=-INFINITY,low=INFINITY;
            grow=shrink=0;
            for(int i=0;i<q;++i) {
                double score=-cache_gradient[i];
                if(score>high) { high=score;grow=i; }
                if(weights[i]>0.&&score<low) { low=score;shrink=i; }
            }
            if((fmax(0.,high-1.)<=1e-8&&high-low<=1e-8)||grow==shrink) {
                phase=WeightPhase::Final;break;
            }
            copy(request_weights,weights,q);operation=WeightOperation::Prepare;
            phase=WeightPhase::Prepared;waiting=true;++preparations;return;
        }
        case WeightPhase::Prepared:request_derivative(1.,WeightPhase::Endpoint);return;
        case WeightPhase::Endpoint:
            if(derivative>=0.) { fraction=1.;phase=WeightPhase::Transfer;break; }
            phase=WeightPhase::BrentStart;break;
        case WeightPhase::BrentStart:
            xpre=0.;xcur=1.;xblk=fblk=spre=scur=0.;brent_iteration=0;
            request_derivative(0.,WeightPhase::BrentFirst);return;
        case WeightPhase::BrentFirst:
            if(isnan(derivative)) { fail(WeightError::BrentNaN);return; }
            fpre=derivative;request_derivative(1.,WeightPhase::BrentSecond);return;
        case WeightPhase::BrentSecond:
            if(isnan(derivative)) { fail(WeightError::BrentNaN);return; }
            fcur=derivative;
            if(fpre==0.) { fraction=xpre;phase=WeightPhase::Transfer;break; }
            if(fcur==0.) { fraction=xcur;phase=WeightPhase::Transfer;break; }
            if(signbit(fpre)==signbit(fcur)) { fail(WeightError::BrentSign);return; }
            phase=WeightPhase::BrentLoop;break;
        case WeightPhase::BrentLoop: {
            if(brent_iteration==100) { fail(WeightError::BrentIterations);return; }
            if(fpre!=0.&&fcur!=0.&&signbit(fpre)!=signbit(fcur)) {
                xblk=xpre;fblk=fpre;spre=scur=xcur-xpre;
            }
            if(fabs(fblk)<fabs(fcur)) { xpre=xcur;xcur=xblk;xblk=xpre;fpre=fcur;fcur=fblk;fblk=fpre; }
            double delta=(1e-14+(4.*DBL_EPSILON)*fabs(xcur))/2.,sbis=(xblk-xcur)/2.;
            if(fcur==0.||fabs(sbis)<delta) { fraction=xcur;phase=WeightPhase::Transfer;break; }
            if(fabs(spre)>delta&&fabs(fcur)<fabs(fpre)) {
                double trial;
                if(xpre==xblk)trial=-fcur*(xcur-xpre)/(fcur-fpre);
                else {
                    double dpre=(fpre-fcur)/(xpre-xcur),dblk=(fblk-fcur)/(xblk-xcur);
                    trial=-fcur*(fblk*dblk-fpre*dpre)/(dblk*dpre*(fblk-fpre));
                }
                if(2.*fabs(trial)<fmin(fabs(spre),3.*fabs(sbis)-delta)) { spre=scur;scur=trial; }
                else spre=scur=sbis;
            } else spre=scur=sbis;
            xpre=xcur;fpre=fcur;
            xcur+=fabs(scur)>delta?scur:(sbis>0.?delta:-delta);
            request_derivative(xcur,WeightPhase::BrentValue);return;
        }
        case WeightPhase::BrentValue:
            if(isnan(derivative)) { fail(WeightError::BrentNaN);return; }
            fcur=derivative;++brent_iteration;phase=WeightPhase::BrentLoop;break;
        case WeightPhase::Transfer: {
            double amount=fraction*weights[shrink];weights[grow]+=amount;
            weights[shrink]=fraction==1.?0.:weights[shrink]-amount;
            normalize(weights);++repair_steps;phase=WeightPhase::Repair;break;
        }
        case WeightPhase::Final:
            if(evaluate(weights,WeightPhase::FinalValue))return;
            break;
        case WeightPhase::FinalValue: {
            double high=-INFINITY,low=INFINITY;
            for(int i=0;i<q;++i) {
                double score=-cache_gradient[i];if(score>high)high=score;
                if(weights[i]>0.&&score<low)low=score;
            }
            optimality_gap=n*fmax(0.,high-1.);active_gap=high-low;
            if(!isfinite(cache_objective)||!isfinite(optimality_gap)||!isfinite(active_gap)) {
                fail(WeightError::Arithmetic);return;
            }
            if(optimality_gap>n*1e-8||active_gap>1e-8) { fail(WeightError::Convergence);return; }
            operation=WeightOperation::Complete;phase=WeightPhase::Done;return;
        }
        case WeightPhase::Done:return;
        }
    }
};
} // namespace clipp2::resident_slsqp
#undef CLIPP2_WEIGHT_FN
#undef CLIPP2_WEIGHT_SIZE_FN
