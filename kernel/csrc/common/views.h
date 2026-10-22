#pragma once
#include "policies.h"

namespace clipp2 {
// Every observation field is [region, mutation], with no support padding.
struct ObservationSoA {
    std::int64_t mutations, regions;
    const double *alt, *ref, *scaling, *log_constant, *lower, *upper;
    const std::int64_t* support;
    const std::uint8_t* flags;
};

struct Observation {
    double alt, ref, scaling, log_constant;
    std::int64_t support;
    std::uint8_t flags;
};

CLIPP_HD inline Observation observation(const ObservationSoA& data, std::int64_t cell) {
    return {data.alt[cell], data.ref[cell], data.scaling[cell],
            data.log_constant[cell], data.support[cell], data.flags[cell]};
}

struct CellTerms {
    double loss, gradient, curvature;
    EmitStatus status;
};

// Task members are mutation indices in CSR order; grids are strictly increasing.
// Bounds include missing members. NaN incumbent means no supplied incumbent.
struct RefitBatch {
    std::int64_t tasks, members, grid_points;
    const std::int64_t *regions, *member_offsets, *member_rows, *grid_offsets;
    const double *grid, *lower, *upper, *xatol, *incumbent;
};
struct RefitResult {
    double base_center, base_log_likelihood, center, log_likelihood;
    std::int64_t evaluations;
    std::int32_t status;
};
struct ConditionalBatch {
    std::int64_t tasks, members;
    const std::int64_t *regions, *member_offsets, *member_rows;
    const double* centers;
};
}
