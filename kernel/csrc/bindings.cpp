// CliPP2-owned chain kernel, derived from CliPP1.5 9e51e30ac8b0c8c9ab1f46821b496909843bda0c.
// AGPL-3.0; original attribution and local changes are recorded in the project NOTICE.
#include "chain.h"
#include <cstdio>
#include <exception>
#include <stdexcept>

#ifndef CLIPP2_KERNEL_BUILD_ID
#error "CLIPP2_KERNEL_BUILD_ID must bind the compiled kernel source"
#endif

namespace {
thread_local char last_error[1024] = "";
}

extern "C" int CliPP2KernelABI() { return 1; }
extern "C" const char* CliPP2KernelBuildId() { return CLIPP2_KERNEL_BUILD_ID; }
extern "C" const char* CliPP2KernelLastError() { return last_error; }

extern "C" int CliPP2SolveChain(int count, const int* alt, const int* depth,
    const int* major, const int* total, double purity, const double* pilot_cp,
    const int* capacities, int k_count, int* labels_out, double* cp_out,
    double* diagnostics_out, int* statuses_out, CliPP2Callback callback) {
    last_error[0]='\0';
    try {
        if(!labels_out || !cp_out || !diagnostics_out || !statuses_out)
            throw std::invalid_argument("Non-null chain output buffers are required.");
        const auto data=clipp2kernel::prepare_chain_inputs(count,alt,depth,major,
            total,purity,pilot_cp,capacities,k_count);
        clipp2kernel::ChainEvaluator evaluate=[&](const std::vector<double>& x,
            std::vector<double>& values,std::vector<double>& gradient,
            std::vector<double>& curvature) {
            values.resize(count);gradient.resize(count);curvature.resize(count);
            if(callback) {
                if(callback(count,x.data(),values.data(),gradient.data(),curvature.data()))
                    throw std::runtime_error("Likelihood callback failed; no CPU fallback.");
            } else {
                for(int i=0;i<count;++i) {
                    const auto value=clipp2kernel::multiplicity_likelihood_x(x[i],
                        data.alt[i],data.depth[i],data.major[i],data.scale[i],data.log_choose[i]);
                    values[i]=value.nll;gradient[i]=value.gradient;curvature[i]=value.curvature;
                }
            }
        };
        clipp2kernel::solve_chain(data,capacities,k_count,evaluate,labels_out,cp_out,
                                  diagnostics_out,statuses_out);
        return 0;
    } catch(const std::exception& error) {
        std::snprintf(last_error,sizeof(last_error),"%s",error.what());
    } catch(...) {
        std::snprintf(last_error,sizeof(last_error),"%s","Unknown native chain failure.");
    }
    return 1;
}

extern "C" int CliPP2ForestQP(int count, int regions, int edge_count,
    const int* edges, const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics) {
    last_error[0]='\0';
    try {
        clipp2kernel::solve_forest(count,regions,edge_count,edges,curvature,rhs,
                                 penalty,lower,upper,output,diagnostics);
        return 0;
    } catch(const std::exception& error) {
        std::snprintf(last_error,sizeof(last_error),"%s",error.what());
    } catch(...) {
        std::snprintf(last_error,sizeof(last_error),"%s","Unknown native forest failure.");
    }
    return 1;
}
