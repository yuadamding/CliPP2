// CliPP2-owned regional forest interface, derived from CliPP1.5 9e51e30ac8b0c8c9ab1f46821b496909843bda0c.
// AGPL-3.0; original attribution and local changes are recorded in the project NOTICE.
#include "forest.h"
#include <cstdio>
#include <exception>

#ifndef CLIPP2_KERNEL_BUILD_ID
#error "CLIPP2_KERNEL_BUILD_ID must bind the compiled kernel source"
#endif

namespace {
thread_local char last_error[1024] = "";
}

extern "C" int CliPP2KernelABI() { return 3; }
extern "C" const char* CliPP2KernelBuildId() { return CLIPP2_KERNEL_BUILD_ID; }
extern "C" const char* CliPP2KernelLastError() { return last_error; }

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
