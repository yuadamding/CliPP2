#pragma once
#include "views.h"

// Each backend library exports this same ABI. The context owns compact prepared
// fields; input/output arguments below are host pointers unless represented by
// an explicit CUDA handle. A call completes its requested result transfer
// before returning. No implicit CPU CUDA fallback.
struct CliPP2BackendDiagnostics {
    std::int64_t prepared_bytes, workspace_bytes, calls, cell_evaluations;
    std::int64_t cache_bytes, staging_bytes;
    std::int64_t host_to_device_bytes, device_to_host_bytes, kernel_launches, synchronizations;
};

extern "C" {
int CliPP2BackendABI();
const char* CliPP2BackendBuildId();
const char* CliPP2BackendNumericalPolicy();
const char* CliPP2BackendLastError();
int CliPP2BackendCreate(const clipp2::ObservationSoA*, int device_index, void** context);
void CliPP2BackendDestroy(void* context);
int CliPP2BackendGetDiagnostics(void* context, CliPP2BackendDiagnostics* output);
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
// CUDA-only resident execution. Context-local handles own immutable columns;
// clearing cached entries never invalidates a live handle.
int CliPP2CreateJointColumns(void*, std::int64_t count, const double* centers,
                            std::int64_t* handles);
// Explicit host import is also useful for independent numerical qualification.
// Layout is [cluster, mutation], retaining finite and negative-infinite logs.
int CliPP2ImportJointColumns(void*, std::int64_t count, const double* log_columns,
                            std::int64_t* handles);
int CliPP2ReleaseJointColumns(void*, std::int64_t count, const std::int64_t* handles);
int CliPP2ExportJointColumns(void*, std::int64_t count, const std::int64_t* handles,
                            double* log_columns);
int CliPP2GatherJointColumns(void*, std::int64_t count, const std::int64_t* handles,
                            const std::int64_t* labels, double* assigned);
int CliPP2CreateMixtureProblem(void*, std::int64_t count, const std::int64_t* handles,
                              std::int64_t* problem);
int CliPP2ReleaseMixtureProblem(void*, std::int64_t problem);
int CliPP2EvaluateMixture(void*, std::int64_t problem, const double* weights,
                         double* objective, double* gradient, double* log_likelihood);
int CliPP2PrepareMixtureTransferEndpoint(void*, std::int64_t problem, const double* weights,
                                       std::int64_t grow, std::int64_t shrink, double* derivative);
int CliPP2MixtureTransferDerivative(void*, std::int64_t problem, double fraction,
                                   double* derivative);
// Fixed-length counter vectors; do not change the shared diagnostic struct.
int CliPP2CUDAReuseDiagnostics(void*, std::int64_t* counters9);
int CliPP2CUDAGridDiagnostics(void*, std::int64_t* counters3);
int CliPP2ReleaseCUDAOptionalCache(void*);
int CliPP2CUDAScoringDiagnostics(void*, std::int64_t* counters9);
int CliPP2CUDAScoringArenaDiagnostics(void*, std::int64_t* counters3);
}

namespace clipp2 {
// CPU implementation supplied by cpu/refit.cpp. CUDA has its own batch driver.
struct ProbabilityWork;
class ProbabilityCache;
void refit_centers(const ObservationSoA&, const RefitBatch&, RefitResult*, ProbabilityWork&, ProbabilityCache&);
void conditional_values(const ObservationSoA&, const ConditionalBatch&, double*, std::uint8_t*,
                        ProbabilityWork&, ProbabilityCache&);
}
