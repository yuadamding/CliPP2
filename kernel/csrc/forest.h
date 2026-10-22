// CliPP2-owned shared-forest interface; AGPL-3.0. See the project NOTICE.
#pragma once

namespace clipp2kernel {
void reset_forest_workspace();
void solve_forest(int count, int regions, int edge_count, const int* edges,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics);
}

// Caller-owned row-major [count,regions] arrays and [edge_count,2] edges.
// Outputs are valid only on return zero; diagnostics are active-set sweeps
// and regional sweep-budget hits. Python independently audits KKT residuals.
extern "C" {
int CliPP2KernelABI();
const char* CliPP2KernelBuildId();
const char* CliPP2KernelLastError();
int CliPP2ForestQP(int count, int regions, int edge_count, const int* edges,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics);
}
