#pragma once
#include "views.h"
#include <cmath>

namespace clipp2 {
CLIPP_HD inline bool finite_number(double value) {
    return value <= maximum_finite && value >= -maximum_finite;
}

CLIPP_HD inline double component_probability(const Observation& o, double phi,
                                              std::int64_t multiplicity) {
    const double mass = (o.scaling * static_cast<double>(multiplicity)) * phi;
    if (o.flags & mixed_flag)
        return mass < mixed_epsilon ? mixed_epsilon :
               (mass > 1 - mixed_epsilon ? 1 - mixed_epsilon : mass);
    return mass > 1 ? 1 : mass;
}

CLIPP_HD inline double component_log_mass(const Observation& o, double p) {
    // The zero-count branches implement xlogy(0, 0) and xlog1py(0, -1).
    return (o.alt == 0 ? 0 : o.alt * ::log(p))
         + (o.ref == 0 ? 0 : o.ref * ::log1p(-p)) + o.log_constant;
}

template<EmitMode Mode, int FixedSupport = 0>
CLIPP_HD inline CellTerms evaluate_supported(const Observation& o, double phi) {
    constexpr double infinity = positive_infinity;
    const std::int64_t support = FixedSupport ? FixedSupport : o.support;
    double maximum = -infinity;
    for (std::int64_t index = 0; index < support; ++index) {
        const double p = component_probability(o, phi, index + 1);
        const double mass = component_log_mass(o, p);
        if (mass > maximum) maximum = mass;
    }
    if (maximum == -infinity)
        return {infinity, 0, 1, EmitStatus::ZeroLikelihood};
    if (!finite_number(maximum))
        return {infinity, 0, 1, EmitStatus::NumericalFailure};
    // Fixed increasing-support order; independent cells can be scheduled freely.
    double total = 0;
    for (std::int64_t index = 0; index < support; ++index) {
        const double k = component_log_mass(o, component_probability(o, phi, index + 1));
        total += ::exp(k - maximum);
    }
    const double ll = maximum + ::log(total);
    if constexpr (Mode == EmitMode::LossOnly)
        return {-ll, 0, 0, EmitStatus::Ok};

    double gradient = 0, curvature = 0, endpoint_gradient = 0, endpoint_curvature = 0;
    bool infinite_curvature = false;
    for (std::int64_t index = 0; index < support; ++index) {
        const double slope = o.scaling * static_cast<double>(index + 1);
        const double mass = slope * phi;
        const double p = component_probability(o, phi, index + 1);
        const double k = component_log_mass(o, p);
        const double posterior = finite_number(k) ? ::exp(k - ll) : 0;
        if (posterior > 0) {
            const double derivative = (o.flags & mixed_flag) &&
                (mass <= mixed_epsilon || mass >= 1 - mixed_epsilon) ? 0 : slope;
            const double g = derivative * ((o.alt == 0 ? 0 : -o.alt / p) +
                                            (o.ref == 0 ? 0 : o.ref / (1 - p)));
            const double h = derivative * derivative *
                ((o.alt == 0 ? 0 : o.alt / (p * p)) +
                 (o.ref == 0 ? 0 : o.ref / ((1 - p) * (1 - p))));
            gradient += posterior * g;
            curvature += posterior * h;
        }
        // A zero-probability endpoint component can have a finite limiting
        // derivative. Preserve these terms even when its posterior is zero.
        if (!(o.flags & mixed_flag) && p == 1) {
            if (o.ref == 1) {
                endpoint_gradient += slope * ::exp(o.log_constant - ll);
                infinite_curvature = true;
            } else if (o.ref == 2) {
                endpoint_curvature += 2 * slope * slope * ::exp(o.log_constant - ll);
            }
        }
    }
    gradient += endpoint_gradient;
    curvature += endpoint_curvature;
    curvature = infinite_curvature ? infinity :
        (curvature < curvature_floor ? curvature_floor : curvature);
    if (gradient != gradient || curvature != curvature)
        return {-ll, gradient, curvature, EmitStatus::NumericalFailure};
    return {-ll, gradient, curvature, EmitStatus::Ok};
}

template<EmitMode Mode>
CLIPP_HD inline CellTerms evaluate_observation(const Observation& o, double phi) {
    if (!(o.flags & observed_flag)) return {0, 0, 0, EmitStatus::Ok};
    if (!finite_number(phi) || phi < 0 || phi > 1)
        return {positive_infinity, 0, 1, EmitStatus::InvalidInput};
    switch (o.support) {
    case 1: return evaluate_supported<Mode, 1>(o, phi);
    case 2: return evaluate_supported<Mode, 2>(o, phi);
    case 3: return evaluate_supported<Mode, 3>(o, phi);
    case 4: return evaluate_supported<Mode, 4>(o, phi);
    default: return evaluate_supported<Mode>(o, phi);
    }
}

CLIPP_HD inline double multiplicity_map(const Observation& o, double phi, EmitStatus& status) {
    status = EmitStatus::Ok;
    constexpr double missing = quiet_nan;
    if (!(o.flags & observed_flag)) return missing;
    if (!finite_number(phi) || phi < 0 || phi > 1) {
        status = EmitStatus::InvalidInput; return missing;
    }
    const auto terms = evaluate_observation<EmitMode::LossOnly>(o, phi);
    status = terms.status;
    if (status != EmitStatus::Ok) return missing;
    double best = -1;
    double result = missing;
    for (std::int64_t index = 0; index < o.support; ++index) {
        const double k = component_log_mass(o, component_probability(o, phi, index + 1));
        const double posterior = ::exp(k + terms.loss);
        if (posterior > best) { best = posterior; result = static_cast<double>(index + 1); }
    }
    return result;
}
}
