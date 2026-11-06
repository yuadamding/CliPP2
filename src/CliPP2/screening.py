"""Mask-aware descriptive profile-deviance summaries, without calibration claims."""
import numpy as np

from .config import DescriptiveScreen


def _deviance_part(x, mean, difference):
    """Stable ``x*log(x/mean) + mean-x`` on positive, finite arguments."""
    v = difference / (x + mean)
    near = np.abs(v) < .1
    result = np.empty(x.shape)
    result[~near] = x[~near] * (np.log(x[~near]) - np.log(mean[~near])) - difference[~near]
    # This series avoids subtracting first-order terms at a saturated component.
    delta, ratio, values = difference[near], v[near], x[near]
    total = delta * ratio
    power = 2 * values * ratio
    for denominator in range(3, 101, 2):
        power *= ratio * ratio
        updated = total + power / denominator
        if np.array_equal(updated, total):
            break
        total = updated
    result[near] = total
    return result


def binomial_profile_deviance(alt, depth, probability, informative=None):
    """Direct binomial deviance, without subtracting binomial log masses.

    The narrow saturation convention is ``probability == alt/depth`` gives
    exactly zero. Adjacent representable probabilities retain positive loss.
    Missing/excluded pairs are NaN. Impossible informative components, including
    an undefined MAP probability, have infinite deviance.
    """
    alt, depth, probability = np.broadcast_arrays(np.asarray(alt, float),
        np.asarray(depth, float), np.asarray(probability, float))
    mask = depth > 0 if informative is None else np.broadcast_to(np.asarray(informative, bool), depth.shape)
    if (np.any(~np.isfinite(depth[mask])) or np.any(depth[mask] <= 0)
            or np.any(~np.isfinite(alt[mask])) or np.any(alt[mask] < 0) or np.any(alt[mask] > depth[mask])
            or np.any(np.isinf(probability[mask])) or np.any(probability[mask] < 0)
            or np.any(probability[mask] > 1)):
        raise ValueError("Profile deviance requires valid informative counts and probabilities")
    result = np.full(depth.shape, np.nan)
    result[mask] = np.inf
    valid = mask & np.isfinite(probability)
    q = np.divide(alt, depth, out=np.zeros(depth.shape), where=mask)
    equal = valid & (probability == q)
    result[equal] = 0.
    interior = valid & ~equal & (probability > 0) & (probability < 1)
    zero = interior & (q == 0)
    one = interior & (q == 1)
    result[zero] = -2 * depth[zero] * np.log1p(-probability[zero])
    result[one] = -2 * depth[one] * np.log(probability[one])
    middle = interior & (q > 0) & (q < 1)
    # Preserve the same difference for both Bernoulli outcomes: subtracting
    # rounded complements can otherwise erase an adjacent positive discrepancy.
    difference = q[middle] - probability[middle]
    result[middle] = 2 * depth[middle] * (
        _deviance_part(q[middle], probability[middle], difference)
        + _deviance_part(1-q[middle], 1-probability[middle], -difference))
    return result


def profile_deviance_from_diagnostics(observation, centers, diagnostic):
    """Reconstruct the conditional MAP probability in native multiplication order."""
    probability = np.minimum((observation.scaling * diagnostic["map_multiplicity"]) * centers, 1.)
    return binomial_profile_deviance(observation.alt, observation.depth, probability, observation.observed)


def summarize_profile_deviance(deviance, informative, screen, *, axis):
    """Summarize only informative pairs along the caller's aggregation axis.

    Infinite discrepancy is an outlier and fails the mean gate. Missing pairs
    contribute neither values nor denominator; an empty domain cannot pass.
    """
    values, mask = np.asarray(deviance, float), np.asarray(informative, bool)
    if type(screen) is not DescriptiveScreen or values.shape != mask.shape:
        raise ValueError("Profile screen requires aligned deviations/masks and validated thresholds")
    if np.isnan(values[mask]).any() or np.any(values[mask] < 0):
        raise ValueError("Informative profile deviances must be nonnegative and nonmissing")
    count = mask.sum(axis=axis)
    mean = np.divide(np.where(mask, values, 0.).sum(axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)
    outliers = (mask & (values > screen.pair_deviance_max)).sum(axis=axis)
    fraction = np.divide(outliers, count, out=np.full(np.shape(count), np.nan), where=count > 0)
    return dict(informative_count=count, mean_profile_deviance=mean,
                outlier_count=outliers, outlier_fraction=fraction,
                descriptive_pass=(count > 0) & np.isfinite(mean)
                    & (mean <= screen.mean_deviance_max) & (fraction <= screen.outlier_fraction_max))
