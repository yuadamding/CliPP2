#pragma once
#include "views.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clipp2 {
inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
inline std::int64_t checked_product(std::int64_t a, std::int64_t b,
                                    std::size_t element_bytes = sizeof(double)) {
    require(a >= 0 && b >= 0 && (!b || a <= std::numeric_limits<std::int64_t>::max() / b),
            "Backend dimension product overflow");
    const auto count = a * b;
    require(element_bytes && static_cast<std::uint64_t>(count) <=
            std::numeric_limits<std::size_t>::max() / element_bytes,
            "Backend array byte count overflow");
    return count;
}
inline std::int64_t validate_observations(const ObservationSoA& input) {
    require(input.aggregation == LikelihoodAggregation::UniformMarginal ||
            input.aggregation == LikelihoodAggregation::IntegerProfile, "Unknown likelihood aggregation");
    require(input.mutations > 0 && input.regions > 0, "Model dimensions must be positive");
    const auto cells = checked_product(input.mutations, input.regions);
    require(input.alt && input.ref && input.scaling && input.log_constant &&
            input.lower && input.upper && input.minimum && input.support && input.flags, "Null model field");
    checked_product(cells, 65, 1);
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        require((input.flags[cell] & ~observed_flag) == 0, "Only single-state observations are supported");
        require(input.support[cell] > 0, "Multiplicity support must be positive");
        require(input.minimum[cell] > 0 && input.minimum[cell] <=
                std::numeric_limits<std::int64_t>::max() - (input.support[cell] - 1),
                "Multiplicity interval must have positive endpoints without overflow");
        require(std::isfinite(input.scaling[cell]) && input.scaling[cell] > 0,
                "Scaling must be finite and positive");
        require(input.lower[cell] == 0 && input.upper[cell] == 1,
                "Single-state CCF bounds must be [0,1]");
        if (input.flags[cell] & observed_flag)
            require(std::isfinite(input.alt[cell]) && input.alt[cell] >= 0 &&
                    input.alt[cell] == std::floor(input.alt[cell]) &&
                    std::isfinite(input.ref[cell]) && input.ref[cell] >= 0 &&
                    input.ref[cell] == std::floor(input.ref[cell]) &&
                    std::isfinite(input.log_constant[cell]), "Invalid observed counts or constant");
    }
    return cells;
}
}
