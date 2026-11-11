#pragma once
#include "policies.h"

namespace clipp2 {
// Every observation field is [region, mutation], with no support padding.
struct ObservationSoA {
    std::int64_t mutations, regions;
    const double *alt, *ref, *scaling, *log_constant, *lower, *upper;
    // Multiplicities are the inclusive interval [minimum, minimum + support - 1].
    // lower/upper above remain CCF-domain bounds.
    const std::int64_t *minimum, *support;
    const std::uint8_t* flags;
    LikelihoodAggregation aggregation = LikelihoodAggregation::UniformMarginal;
};

struct Observation {
    double alt, ref, scaling, log_constant;
    std::int64_t minimum, support;
    std::uint8_t flags;
    LikelihoodAggregation aggregation = LikelihoodAggregation::UniformMarginal;
};

CLIPP_HD inline Observation observation(const ObservationSoA& data, std::int64_t cell) {
    return {data.alt[cell], data.ref[cell], data.scaling[cell],
            data.log_constant[cell], data.minimum[cell], data.support[cell], data.flags[cell], data.aggregation};
}

struct CellTerms {
    double loss, gradient, curvature;
    EmitStatus status;
};

// One compact record per observation, never a padded component tensor.
// Status 4 denotes missing reads; all numerical fields then remain NaN.
struct ComponentDiagnostics {
    double log_emission, max_component_log_mass, map_multiplicity;
    double map_log_probability, top_two_gap, accepted_log_probability;
    std::uint8_t has_second, status;
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
