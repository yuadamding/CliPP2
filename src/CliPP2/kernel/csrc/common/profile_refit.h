#pragma once
#include "emission_math.h"
#include "scalar_search.h"

namespace clipp2 {
// The host prepares the exact union of adjacent-component switch brackets.
// Only membership/branch metadata cross devices; all objectives and roots on
// CUDA are evaluated in the allocated device, not by a host likelihood fallback.
struct ProfileBranchDerivative {
    ObservationSoA data;
    const std::int64_t* rows;
    std::int64_t count, region;
    double probe, center;
    CLIPP_HD double operator()(std::int64_t member) const {
        const auto o = observation(data, region * data.mutations + rows[member]);
        if (!(o.flags & observed_flag)) return 0;
        const auto m = profile_multiplicity(o, probe);
        const double slope = o.scaling * static_cast<double>(m), p = slope * center;
        if (p > 1) return o.ref == 0 ? 0 : -positive_infinity;
        return (o.alt == 0 ? 0 : o.alt / center)
             - (o.ref == 0 ? 0 : o.ref * slope / (1 - p));
    }
    CLIPP_HD double at(double value) const {
        auto copy = *this; copy.center = value;
        return pairwise_sum(copy, 0, count);
    }
};

template<class Objective, class Derivative>
CLIPP_HD RefitResult profile_refit_search(const RefitBatch& batch, std::int64_t task,
        const Objective& objective, Derivative derivative) {
    RefitResult out{quiet_nan, -positive_infinity, quiet_nan, -positive_infinity, 0, 0};
    const auto first = batch.grid_offsets[task], last = batch.grid_offsets[task + 1];
    for (auto index = first; index < last; ++index) {
        const double center = batch.grid[index], value = objective(center);
        ++out.evaluations;
        if (value != value || value == positive_infinity) { out.status = 1; return out; }
        if (out.base_center != out.base_center ||
                refit_prefer(value, center, out.base_log_likelihood, out.base_center)) {
            out.base_center = center; out.base_log_likelihood = value;
        }
        if (index + 1 == last) continue;
        double lo = center, hi = batch.grid[index + 1];
        derivative.probe = lo + (hi - lo) / 2;
        if (!(lo < derivative.probe && derivative.probe < hi)) continue;
        // Every open interval is a fixed-branch concave binomial problem.
        const double gl = derivative.at(lo), gh = derivative.at(hi);
        if (gl != gl || gh != gh) { out.status = 1; return out; }
        if (!(gl > 0 && gh < 0)) continue; // Its maximum is an evaluated endpoint.
        int iterations = 0;
        while (hi - lo > batch.xatol[task]) {
            const double middle = lo + (hi - lo) / 2;
            if (!(lo < middle && middle < hi)) break;
            const double g = derivative.at(middle);
            if (g != g || ++iterations > 128) { out.status = 1; return out; }
            if (g == 0) { lo = hi = middle; break; }
            if (g > 0) lo = middle; else hi = middle;
        }
        const double root = lo + (hi - lo) / 2, ll = objective(root);
        ++out.evaluations;
        if (!finite_number(ll)) { out.status = 1; return out; }
        if (refit_prefer(ll, root, out.base_log_likelihood, out.base_center)) {
            out.base_center = root; out.base_log_likelihood = ll;
        }
    }
    if (!finite_number(out.base_log_likelihood)) out.status = 2;
    out.center = out.base_center; out.log_likelihood = out.base_log_likelihood;
    return out; // Initial centers do not select a local profile branch.
}
}
