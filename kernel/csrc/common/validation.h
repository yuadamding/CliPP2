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
inline std::int64_t validate_observations(const ObservationSoA& input, std::int64_t budget) {
    require(input.mutations > 0 && input.regions > 0, "Model dimensions must be positive");
    require(budget >= 0, "Workspace budget must be nonnegative");
    const auto cells = checked_product(input.mutations, input.regions);
    require(input.alt && input.ref && input.scaling && input.log_constant &&
            input.lower && input.upper && input.support && input.flags, "Null model field");
    checked_product(cells, 57, 1);
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        require((input.flags[cell] & ~3u) == 0, "Unknown observation flags");
        require(input.support[cell] > 0, "Multiplicity support must be positive");
        require(std::isfinite(input.scaling[cell]) && input.scaling[cell] > 0,
                "Scaling must be finite and positive");
        require(std::isfinite(input.lower[cell]) && std::isfinite(input.upper[cell]) &&
                input.lower[cell] >= 0 && input.upper[cell] <= 1 &&
                input.lower[cell] <= input.upper[cell], "Invalid assignment box");
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
