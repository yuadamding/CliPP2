#pragma once
#include "views.h"
#include <cmath>

/* The bounded-search transitions are adapted from SciPy's
 * scipy.optimize._optimize._minimize_scalar_bounded (SciPy 1.18.0).
 * Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
namespace clipp2 {
// No optimizer arithmetic depends on a physical worker count or cache budget.
struct BoundedSearchState {
    double lower, upper, x, w, v, fx, fw, fv, step, previous_step;
    double trial, trial_value, xatol, midpoint, tolerance;
    std::int32_t evaluations, max_evaluations, status;
};

CLIPP_HD inline double sign_or_one(double value) {
    return value < 0.0 ? -1.0 : 1.0;
}

CLIPP_HD inline void bounded_finish(BoundedSearchState& state, int status) {
    state.status = (std::isnan(state.x) || std::isnan(state.fx)
                    || std::isnan(state.trial_value)) ? 2 : status;
}

CLIPP_HD inline void bounded_propose(BoundedSearchState& state) {
    state.midpoint = 0.5 * (state.lower + state.upper);
    state.tolerance = std::sqrt(2.2e-16) * std::abs(state.x) + state.xatol / 3.0;
    const double twice_tolerance = 2.0 * state.tolerance;
    if (!(std::abs(state.x - state.midpoint)
          > twice_tolerance - 0.5 * (state.upper - state.lower))) {
        bounded_finish(state, 0);
        return;
    }
    bool golden = true;
    if (std::abs(state.previous_step) > state.tolerance) {
        golden = false;
        double r = (state.x - state.w) * (state.fx - state.fv);
        double q = (state.x - state.v) * (state.fx - state.fw);
        double p = (state.x - state.v) * q - (state.x - state.w) * r;
        q = 2.0 * (q - r);
        if (q > 0.0) p = -p;
        q = std::abs(q);
        r = state.previous_step;
        state.previous_step = state.step;
        if ((std::abs(p) < std::abs(0.5 * q * r))
            && (p > q * (state.lower - state.x))
            && (p < q * (state.upper - state.x))) {
            state.step = (p + 0.0) / q;
            const double candidate = state.x + state.step;
            if ((candidate - state.lower < twice_tolerance)
                || (state.upper - candidate < twice_tolerance))
                state.step = state.tolerance * sign_or_one(state.midpoint - state.x);
        } else golden = true;
    }
    if (golden) {
        state.previous_step = state.x >= state.midpoint
            ? state.lower - state.x : state.upper - state.x;
        state.step = (0.5 * (3.0 - std::sqrt(5.0))) * state.previous_step;
    }
    // fmax would silently discard NaN; NumPy maximum does not.
    const double magnitude = std::abs(state.step);
    const double distance = std::isnan(magnitude) ? magnitude
        : (magnitude > state.tolerance ? magnitude : state.tolerance);
    state.trial = state.x + sign_or_one(state.step) * distance;
}

CLIPP_HD inline BoundedSearchState bounded_initialize(
        double lower, double upper, double xatol, int max_evaluations = 500) {
    const double initial = lower + (0.5 * (3.0 - std::sqrt(5.0))) * (upper - lower);
    return {lower, upper, initial, initial, initial, 0.0, 0.0, 0.0,
            0.0, 0.0, initial, positive_infinity, xatol,
            0.0, 0.0, 0, max_evaluations, -1};
}

CLIPP_HD inline void bounded_consume(BoundedSearchState& state, double value) {
    if (state.status != -1) return;
    if (state.evaluations++ == 0) {
        state.fx = state.fw = state.fv = value;
        bounded_propose(state);
        return;
    }
    state.trial_value = value;
    if (value <= state.fx) {
        if (state.trial >= state.x) state.lower = state.x;
        else state.upper = state.x;
        state.v = state.w; state.fv = state.fw;
        state.w = state.x; state.fw = state.fx;
        state.x = state.trial; state.fx = value;
    } else {
        if (state.trial < state.x) state.lower = state.trial;
        else state.upper = state.trial;
        if (value <= state.fw || state.w == state.x) {
            state.v = state.w; state.fv = state.fw;
            state.w = state.trial; state.fw = value;
        } else if (value <= state.fv || state.v == state.x || state.v == state.w) {
            state.v = state.trial; state.fv = value;
        }
    }
    // SciPy checks this budget even when the just-updated bracket converged.
    if (state.evaluations >= state.max_evaluations) bounded_finish(state, 1);
    else bounded_propose(state);
}

// NumPy's float64 pairwise reduction topology (block size 128). The logical
// grouping is explicit and shared; scheduling never changes the arithmetic.
template<int Depth, class Values>
CLIPP_HD double pairwise_block_sum(const Values& values, std::int64_t offset,
                                   std::int64_t count) {
    if (count < 8) {
        double result = -0.0;
        for (std::int64_t i = 0; i < count; ++i) result += values(offset + i);
        return result;
    }
    if (count <= 128) {
        double accumulators[8];
        for (int j = 0; j < 8; ++j) accumulators[j] = values(offset + j);
        std::int64_t i = 8;
        for (; i < count - count % 8; i += 8)
            for (int j = 0; j < 8; ++j) accumulators[j] += values(offset + i + j);
        double result = ((accumulators[0] + accumulators[1])
                         + (accumulators[2] + accumulators[3]))
                      + ((accumulators[4] + accumulators[5])
                         + (accumulators[6] + accumulators[7]));
        for (; i < count; ++i) result += values(offset + i);
        return result;
    }
    if constexpr (Depth > 0) {
        std::int64_t split = count / 2;
        split -= split % 8;
        return pairwise_block_sum<Depth - 1>(values, offset, split)
             + pairwise_block_sum<Depth - 1>(values, offset + split, count - split);
    }
    // Only called with at most 8192 values and six static bisection levels.
    return quiet_nan;
}

template<class Values>
CLIPP_HD double pairwise_sum(const Values& values, std::int64_t offset,
                             std::int64_t count) {
    // NumPy's default external reduction loop consumes 8192-value buffers.
    // Fix that logical grouping here, independent of execution workspace and
    // NumPy's mutable process-global buffer-size setting.
    double result = -0.0;
    for (std::int64_t first = 0; first < count; first += 8192) {
        const auto remaining = count - first;
        result += pairwise_block_sum<6>(values, offset + first, remaining < 8192 ? remaining : 8192);
    }
    return result;
}

enum class RefitStatus : std::int32_t {
    Ok = 0, IntervalFailure = 1, NoFiniteLikelihood = 2,
    InvalidBox = 3, Unsupported = 4
};

CLIPP_HD inline bool refit_prefer(double value, double center,
                                  double incumbent_value, double incumbent_center) {
    return value > incumbent_value || (value == incumbent_value && center < incumbent_center);
}

template<class Objective>
CLIPP_HD bool refit_interval(double lower, double upper, double xatol,
                             const Objective& objective, RefitResult& result) {
    auto state = bounded_initialize(lower, upper, xatol);
    while (state.status == -1) bounded_consume(state, -objective(state.trial));
    result.evaluations += state.evaluations;
    if (state.status || !std::isfinite(state.fx)) {
        result.status = static_cast<std::int32_t>(RefitStatus::IntervalFailure);
        return false;
    }
    if (refit_prefer(-state.fx, state.x, result.base_log_likelihood, result.base_center)) {
        result.base_center = state.x;
        result.base_log_likelihood = -state.fx;
    }
    return true;
}

// Objective is a host callable or a collective CUDA-block callable: every
// participant receives the same total. store_grid writes once; finish_grid
// publishes those values before every participant follows the same decisions.
template<class Objective>
CLIPP_HD RefitResult refit_search(const RefitBatch& batch, std::int64_t task,
                                  double* grid_values, const Objective& objective) {
    constexpr double nan = quiet_nan;
    RefitResult result{nan, nan, nan, nan, 0, 0};
    const double lower = batch.lower[task], upper = batch.upper[task];
    if (!(std::isfinite(lower) && std::isfinite(upper) && 0 <= lower
          && lower <= upper && upper <= 1.0)) {
        result.status = static_cast<std::int32_t>(RefitStatus::InvalidBox);
        return result;
    }
    const std::int64_t first = batch.grid_offsets[task];
    const std::int64_t count = batch.grid_offsets[task + 1] - first;
    const double* grid = batch.grid + first;
    for (std::int64_t i = 0; i < count; ++i) {
        const double value = objective(grid[i]);
        objective.store_grid(grid_values + i, value);
        if (!i || value > result.base_log_likelihood) {
            result.base_center = grid[i]; result.base_log_likelihood = value;
        }
    }
    result.evaluations = count;
    objective.finish_grid();
    if (count > 1) {
        if (!refit_interval(grid[0], grid[1], batch.xatol[task], objective, result))
            return result;
        for (std::int64_t i = 1; i + 1 < count; ++i) {
            const double a = grid_values[i - 1], b = grid_values[i], c = grid_values[i + 1];
            if (b >= a && b >= c && (b > a || b > c))
                if (!refit_interval(grid[i - 1], grid[i + 1], batch.xatol[task],
                                    objective, result)) return result;
        }
        // For two grid points this duplicates the already searched left edge.
        if (count > 2 && !refit_interval(grid[count - 2], grid[count - 1],
                                       batch.xatol[task], objective, result)) return result;
    }
    if (!std::isfinite(result.base_log_likelihood)) {
        result.status = static_cast<std::int32_t>(RefitStatus::NoFiniteLikelihood);
        return result;
    }
    result.center = result.base_center;
    result.log_likelihood = result.base_log_likelihood;
    const double incumbent = batch.incumbent ? batch.incumbent[task] : nan;
    if (std::isfinite(incumbent) && lower <= incumbent && incumbent <= upper) {
        const double value = objective(incumbent);
        ++result.evaluations;
        if (value > result.log_likelihood) {
            result.center = incumbent; result.log_likelihood = value;
        }
    }
    return result;
}
}
