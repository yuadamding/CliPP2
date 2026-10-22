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
    int workers, long long workspace_bytes, double* coordinates, int* labels,
    unsigned char* cuts, CliPP2ContinuationReport* reports, int* actual_workers);
}
