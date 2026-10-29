// CliPP2 CPU continuation over one immutable mutation tree. AGPL-3.0.
#pragma once
#include "common/views.h"

struct CliPP2ContinuationPolicy {
    int levels, iterations;
    double stationarity, constraint, maximum_penalty;
};
struct CliPP2ContinuationReport {
    double rho, stationarity, violation, block_range, likelihood, objective;
    std::int64_t iterations, evaluations, qp_calls, qp_unqualified, warm_starts, accepted_steps;
    int levels, longest, status;
};
// Optional host reverse-communication owner. Numerical terms are supplied by
// its caller (CUDA stays CUDA); suspended native workers never execute emission
// fallback. All native controller work is serialized within one owner.
struct CliPP2ContinuationRequest {
    std::uint64_t sequence;
    int kind, boundary, level; // kind: 0 busy, 1 terms, 2 progress, 3 done, 4 error
    std::int64_t iterations;
    const double* coordinates;
};
// Compatibility boundary for NumPy's platform-specific BLAS norm reduction.
// Return NaN on failure. The controller, emissions and all inner iterations
// remain native; this reduction is called only once at the end of each level.
using CliPP2Norm = double (*)(const double*, long long);
// NumPy's long-double einsum has a distinct, version/platform-specific inner
// reduction. Preserve it exactly, including near ties and subnormal jumps.
using CliPP2CutScores = int (*)(const long double*, long long, int, long double*);
extern "C" {
const char* CliPP2ContinuationLastError();
int CliPP2ContinueCPU(const clipp2::ObservationSoA*, const int* edges,
    const int* canonical_order, const double* initial, const double* lower,
    const double* upper, int tasks, const int* capacities,
    const CliPP2ContinuationPolicy*, const double* gaps, CliPP2Norm norm, CliPP2CutScores scores,
    int workers,
    int output_flags, double* coordinates, int* labels,
    unsigned char* cuts, CliPP2ContinuationReport* reports, int* actual_workers);
int CliPP2ContinuationOwnerCreate(int n,int regions,const int* edges,const int* canonical,
    const double* initial,const double* values,const double* gradient,const double* curvature,
    const double* lower,const double* upper,const double* gaps,int tasks,const int* capacities,
    const CliPP2ContinuationPolicy*,CliPP2Norm,CliPP2CutScores,
    int output_flags,int boundary_requests,void** owner);
int CliPP2ContinuationOwnerPoll(void* owner,int task,int wait,CliPP2ContinuationRequest* request);
int CliPP2ContinuationOwnerReply(void* owner,int task,std::uint64_t sequence,
    const double* values,const double* gradient,const double* curvature);
int CliPP2ContinuationOwnerResult(void* owner,int task,double* coordinates,int* labels,
    unsigned char* cuts,CliPP2ContinuationReport* report);
const char* CliPP2ContinuationOwnerError(void* owner,int task);
void CliPP2ContinuationOwnerDestroy(void* owner);
}
