#pragma once
#include "views.h"

// Each backend library exports this same ABI. The context owns compact prepared
// fields; input/output arguments below are host pointers. A call completes its
// requested result transfer before returning. No implicit CPU CUDA fallback.
struct CliPP2BackendDiagnostics {
    std::int64_t prepared_bytes, workspace_bytes, calls, cell_evaluations;
    std::int64_t cache_bytes, staging_bytes, workspace_limit;
    std::int64_t host_to_device_bytes, device_to_host_bytes, kernel_launches, synchronizations;
};

extern "C" {
int CliPP2BackendABI();
const char* CliPP2BackendBuildId();
const char* CliPP2BackendNumericalPolicy();
const char* CliPP2BackendLastError();
int CliPP2BackendCreate(const clipp2::ObservationSoA*, std::int64_t workspace_bytes,
                       int device_index, void** context);
void CliPP2BackendDestroy(void* context);
int CliPP2BackendGetDiagnostics(void* context, CliPP2BackendDiagnostics* output);
int CliPP2BackendDimensions(void* context, std::int64_t* mutations, std::int64_t* regions);
// phi and node results: [batch, region, mutation]. Status is one byte per cell.
int CliPP2EvaluateNodes(void*, std::int64_t batches, const double* phi, int mode,
                       double* loss, double* gradient, double* curvature,
                       std::uint8_t* status);
// Arbitrary prepared-cell indices, useful for batched ragged objectives.
int CliPP2EvaluateCells(void*, std::int64_t count, const std::int64_t* cells,
                       const double* phi, int mode, double* loss, double* gradient,
                       double* curvature, std::uint8_t* status);
// One region-major state; exact posterior MAP with the first multiplicity tie.
int CliPP2MultiplicityMAP(void*, const double* phi, double* multiplicity, std::uint8_t* status);
// Results [grid, row], read emissions only (no assignment bounds).
int CliPP2EvaluateRegionalGrid(void*, std::int64_t region, std::int64_t row_count,
                              const std::int64_t* rows, std::int64_t grid_count,
                              const double* grid, double* log_likelihood,
                              std::uint8_t* status);
// Centers [cluster, region], output [cluster, mutation]; bounds always apply.
int CliPP2EvaluateJointColumns(void*, std::int64_t count, const double* centers,
                              double* log_likelihood, std::uint8_t* status);
int CliPP2RefitCentersBatch(void*, const clipp2::RefitBatch*, clipp2::RefitResult*);
int CliPP2EvaluateConditionalValues(void*, const clipp2::ConditionalBatch*,
                                    double* log_likelihood, std::uint8_t* status);
}

namespace clipp2 {
// CPU implementation supplied by cpu/refit.cpp. CUDA has its own batch driver.
struct ProbabilityWork;
class ProbabilityCache;
void refit_centers(const ObservationSoA&, const RefitBatch&, RefitResult*, std::int64_t, ProbabilityWork&, ProbabilityCache* = nullptr);
void conditional_values(const ObservationSoA&, const ConditionalBatch&, double*, std::uint8_t*,
                        std::int64_t, ProbabilityWork&, ProbabilityCache* = nullptr);
}
