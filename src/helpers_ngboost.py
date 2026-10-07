"""Helper functions adapting ordboost's evaluation pipeline to NGBoost's
parametric (scipy-based) predictive distributions, so plotting.py can be
reused completely unchanged.
"""

from __future__ import annotations

import numpy as np
import properscoring as ps
import xarray as xr
from ordboost.metrics import (
    baseline_distribution,
    crps_score,
    interval_coverage_rate,
    sharpness,
)
from scores.probability import PitFcstAtObs


class NGBoostDistAdapter:
    """Duck-typed wrapper exposing ordboost.PredictiveDistribution-style
    methods over an NGBoost predicted distribution's underlying scipy
    frozen distribution.

    Parameters
    ----------
    scipy_frozen_dist : scipy.stats rv_frozen
        The `.dist` attribute of an NGBoost `pred_dist(X)` result -- a
        scipy frozen distribution with array-valued parameters, one per
        sample.
    """

    def __init__(self, scipy_frozen_dist):
        self._dist = scipy_frozen_dist

    def mean(self) -> np.ndarray:
        """Per-sample mean."""
        return np.asarray(self._dist.mean())

    def median(self) -> np.ndarray:
        """Per-sample median."""
        return np.asarray(self._dist.median())

    def ppf(self, q) -> np.ndarray:
        """Per-sample quantile at level(s) q."""
        return np.asarray(self._dist.ppf(q))

    def interval(self, alpha: float = 0.10):
        """Central prediction interval at significance level alpha.

        Note: scipy's own `.interval()` takes a *confidence* level (the
        fraction covered), the inverse convention from ordboost's alpha
        (tail significance) -- converted here so callers use the same
        alpha= convention as ordboost.PredictiveDistribution throughout.
        """
        lower, upper = self._dist.interval(1.0 - alpha)
        return np.asarray(lower), np.asarray(upper)

    def cdf(self, y) -> np.ndarray:
        """CDF at y: scalar broadcasts across all samples; an array of
        length n_samples evaluates each sample at its own y."""
        return np.asarray(self._dist.cdf(y))

    def __getitem__(self, mask) -> "NGBoostDistAdapter":
        """Boolean-mask row-slicing, re-freezing the underlying scipy
        distribution on the masked parameter arrays."""
        mask = np.asarray(mask, dtype=bool)
        new_args = tuple(
            np.asarray(a)[mask] if np.ndim(a) > 0 else a for a in self._dist.args
        )
        new_kwds = {
            k: (np.asarray(v)[mask] if np.ndim(v) > 0 else v)
            for k, v in self._dist.kwds.items()
        }
        return NGBoostDistAdapter(self._dist.dist(*new_args, **new_kwds))


def crps_ngboost(y_true, dist: NGBoostDistAdapter) -> float:
    """Model CRPS for an NGBoostDistAdapter: closed-form for a Normal
    Dist (fast), else per-sample numerical quadrature via properscoring
    (slower -- consider subsampling large test sets)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    frozen = dist._dist
    if getattr(frozen.dist, "name", "") == "norm":
        return float(
            np.mean(ps.crps_gaussian(y_true_arr, mu=frozen.mean(), sig=frozen.std()))
        )

    scores = np.empty(len(y_true_arr))
    for i, y in enumerate(y_true_arr):
        row = dist[np.arange(len(y_true_arr)) == i]
        scores[i] = ps.crps_quadrature(
            y, row._dist.cdf, xmin=row.ppf(1e-6)[0], xmax=row.ppf(1 - 1e-6)[0]
        )
    return float(np.mean(scores))


def marginal_calibration_curve_ngboost(y_true, dist, grid_y):
    """Marginal calibration for an NGBoostDistAdapter, evaluated over a
    caller-supplied grid_y (no fitted grid available)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    mean_cdf = np.array([np.mean(dist.cdf(y)) for y in grid_y])
    empirical_cdf = np.array([np.mean(y_true_arr <= y) for y in grid_y])
    return grid_y, empirical_cdf - mean_cdf


def pit_diagnostics_ngboost(y_true, dist: NGBoostDistAdapter) -> PitFcstAtObs:
    """Exact PIT diagnostics for an NGBoostDistAdapter, using the same
    scores.probability.PitFcstAtObs machinery as ordboost.metrics.pit_diagnostics

    Values are rounded to 2 decimal places before construction, matching
    the workaround found for the OrdBoost side: PitFcstAtObs's memory use.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    fcst_at_obs = np.round(dist.cdf(y_true_arr), 2)

    fcst_da = xr.DataArray(fcst_at_obs, dims=["sample"])

    return PitFcstAtObs(fcst_da)


def cdf_grid_ngboost(dist: NGBoostDistAdapter, grid_y) -> np.ndarray:
    """Evaluate an NGBoostDistAdapter's CDF at a shared set of grid_y
    points for every sample, producing a (n_samples, n_grid_points) array
    directly analogous to ContinuousPredictiveDistribution.grid_cdf.

    Parameters
    ----------
    dist : NGBoostDistAdapter
    grid_y : array-like
        Same grid used for the ordboost distribution, e.g.
        ordboost_dist.grid_y, so both models' CDFs can be compared or
        plotted point-for-point on identical x-values.

    Returns
    -------
    np.ndarray of shape (n_samples, len(grid_y))
    """
    grid_y = np.asarray(grid_y, dtype=float)
    n_samples = len(dist.mean())
    grid_cdf = np.empty((n_samples, len(grid_y)), dtype=float)
    for j, y in enumerate(grid_y):
        grid_cdf[:, j] = dist.cdf(y)
    return grid_cdf
