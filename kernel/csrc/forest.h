// CliPP2-owned shared-forest interface; AGPL-3.0. See LICENSE and bindings.cpp.
#pragma once

namespace clipp2kernel {
struct PreparedTree;
void reset_forest_workspace();
void prepare_forest_workspace(int nodes,int edges);
void solve_prepared_forest(const PreparedTree&, const unsigned char* cuts, int regions,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics,
    int region_begin=0, int region_end=-1);
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
long long CliPP2ForestRequiredBytes(int count,int edge_count);
int CliPP2ForestQP(int count, int regions, int edge_count, const int* edges,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics,
    long long host_budget_bytes);
}
