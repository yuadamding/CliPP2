#pragma once
#include "validation.h"
#include <cmath>
#include <stdexcept>

namespace clipp2 {
inline void validate_conditional_batch(const ObservationSoA& model, const ConditionalBatch& batch,
                                       const double* output, const std::uint8_t* status) {
    if (batch.tasks < 0 || batch.members < 0)
        throw std::invalid_argument("Negative conditional-value dimensions");
    checked_product(batch.tasks, 8, 1);
    checked_product(batch.members, 8, 1);
    checked_product(batch.tasks + 1, 8, 1);
    if (!batch.tasks) return;
    if (!output || !status || !batch.regions || !batch.member_offsets
        || !batch.member_rows || !batch.centers)
        throw std::invalid_argument("Missing conditional-value array");
    if (batch.member_offsets[0] || batch.member_offsets[batch.tasks] != batch.members)
        throw std::invalid_argument("Invalid conditional-value CSR endpoints");
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        if (batch.regions[task] < 0 || batch.regions[task] >= model.regions
            || batch.member_offsets[task] < 0
            || batch.member_offsets[task] > batch.member_offsets[task + 1]
            || batch.member_offsets[task + 1] > batch.members)
            throw std::invalid_argument("Invalid conditional-value task");
        for (auto index = batch.member_offsets[task]; index < batch.member_offsets[task + 1]; ++index)
            if (batch.member_rows[index] < 0 || batch.member_rows[index] >= model.mutations)
                throw std::invalid_argument("Conditional membership is outside the model");
    }
}

inline void validate_refit_batch(const ObservationSoA& model, const RefitBatch& batch,
                                 const RefitResult* output, bool check_model_bounds = true) {
    if (batch.tasks < 0 || batch.members < 0 || batch.grid_points < 0)
        throw std::invalid_argument("Negative conditional-refit dimensions");
    checked_product(batch.tasks, sizeof(RefitResult), 1);
    checked_product(batch.members, 8, 1);
    checked_product(batch.grid_points, 8, 1);
    checked_product(batch.tasks + 1, 8, 1);
    if (!batch.tasks) return;
    if (!output || !batch.regions || !batch.member_offsets || !batch.member_rows
        || !batch.grid_offsets || !batch.grid || !batch.lower || !batch.upper || !batch.xatol)
        throw std::invalid_argument("Missing conditional-refit array");
    if (batch.member_offsets[0] || batch.grid_offsets[0]
        || batch.member_offsets[batch.tasks] != batch.members
        || batch.grid_offsets[batch.tasks] != batch.grid_points)
        throw std::invalid_argument("Invalid conditional-refit CSR endpoints");
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        if (batch.regions[task] < 0 || batch.regions[task] >= model.regions
            || batch.member_offsets[task] < 0
            || batch.member_offsets[task] > batch.member_offsets[task + 1]
            || batch.member_offsets[task + 1] > batch.members
            || batch.grid_offsets[task] < 0
            || batch.grid_offsets[task] >= batch.grid_offsets[task + 1]
            || batch.grid_offsets[task + 1] > batch.grid_points
            || !std::isfinite(batch.xatol[task]) || batch.xatol[task] <= 0)
            throw std::invalid_argument("Invalid conditional-refit task");
        const auto begin = batch.grid_offsets[task], end = batch.grid_offsets[task + 1];
        for (auto index = begin; index < end; ++index)
            if (!std::isfinite(batch.grid[index]) || batch.grid[index] < batch.lower[task]
                || batch.grid[index] > batch.upper[task]
                || (index > begin && batch.grid[index] <= batch.grid[index - 1]))
                throw std::invalid_argument("Conditional grid must increase inside its box");
        if (batch.grid[begin] != batch.lower[task] || batch.grid[end - 1] != batch.upper[task])
            throw std::invalid_argument("Conditional grid must include both exact bounds");
        for (auto index = batch.member_offsets[task]; index < batch.member_offsets[task + 1]; ++index) {
            const auto row = batch.member_rows[index];
            if (row < 0 || row >= model.mutations)
                throw std::invalid_argument("Conditional membership is outside the model");
            const auto cell = batch.regions[task] * model.mutations + row;
            if (check_model_bounds && (batch.lower[task] < model.lower[cell]
                                       || batch.upper[task] > model.upper[cell]))
                throw std::invalid_argument("Conditional box violates an original membership bound");
        }
    }
}
}
