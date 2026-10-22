// CliPP2-owned regional forest interface, derived from CliPP1.5 9e51e30ac8b0c8c9ab1f46821b496909843bda0c.
// AGPL-3.0; original attribution and local changes are recorded in the project NOTICE.
#include "forest.h"
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>

#ifndef CLIPP2_KERNEL_BUILD_ID
#error "CLIPP2_KERNEL_BUILD_ID must bind the compiled kernel source"
#endif

namespace {
thread_local char last_error[1024] = "";
}

extern "C" int CliPP2KernelABI() { return 4; }
extern "C" const char* CliPP2KernelBuildId() { return CLIPP2_KERNEL_BUILD_ID; }
extern "C" const char* CliPP2KernelLastError() { return last_error; }
extern "C" long long CliPP2ForestRequiredBytes(int n,int edges) {
    if(n<1||edges<0||edges>=n)return -1;
    const long double required=(5*sizeof(double)+4*sizeof(long double)+8*sizeof(int)+1)*static_cast<long double>(n)
        +6.0L*sizeof(int)*edges+8192;
    return required>std::numeric_limits<long long>::max()?-1:static_cast<long long>(required);
}

extern "C" int CliPP2ForestQP(int count, int regions, int edge_count,
    const int* edges, const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics,long long host_budget_bytes) {
    last_error[0]='\0';
    // Synchronous standalone calls retain no unaccounted native storage.
    // Prepared CPU continuation has its separately admitted arena lifetime.
    clipp2kernel::reset_forest_workspace();
    struct Release {~Release(){clipp2kernel::reset_forest_workspace();}} release;
    try {
        const auto required=CliPP2ForestRequiredBytes(count,edge_count);
        if(required<0)throw std::invalid_argument("Invalid forest allocation dimensions");
        if(required>host_budget_bytes) {
            char message[192];std::snprintf(message,sizeof(message),
                "Native forest requires %lld tracked host bytes; available %lld bytes",required,host_budget_bytes);
            throw std::runtime_error(message);
        }
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
