"""Mask-aware descriptive profile-deviance summaries, without calibration claims."""
import numpy as np

from .config import DescriptiveScreen


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
