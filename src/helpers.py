"""Helper functions for conference paper results: CRPS via the `scores`
package, region filtering, and LaTeX table generation.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper
from ordboost.metrics import (
    baseline_distribution,
    marginal_calibration_curve,
    pit_diagnostics,
    sharpness,
    winkler_score,
)
from scores.probability import crps_cdf

from src.bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOT,
    bootstrap_statistic,
    percentile_interval,
)
from src.helpers_ngboost import (
    NGBoostDistAdapter,
    cdf_grid_ngboost,
    crps_ngboost_samples,
    marginal_calibration_curve_ngboost,
    pit_diagnostics_ngboost,
)


def wrap_ngboost_pred_dist(ngb_pred_dist) -> NGBoostDistAdapter:
    """Wrap the output of `NGBRegressor.pred_dist(X)` in an adapter.

    Parameters
    ----------
    ngb_pred_dist
        NGBoost predicted distribution. Its `dist` attribute must be a scipy
        frozen distribution.

    Returns
    -------
    NGBoostDistAdapter
        Adapter exposing the ordboost distribution interface.

    """
    return NGBoostDistAdapter(ngb_pred_dist.dist)


def sample_crps(y_true, dist) -> np.ndarray:
    """Compute the per-sample CRPS of a predictive distribution.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        CRPS of each sample.

    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(dist, NGBoostDistAdapter):
        return crps_ngboost_samples(y_true_arr, dist)

    fcst = xr.DataArray(
        dist.grid_cdf,
        dims=["sample", "threshold"],
        coords={"threshold": dist.grid_y},
    )
    obs = xr.DataArray(y_true_arr, dims=["sample"])
    result = crps_cdf(fcst, obs, threshold_dim="threshold", preserve_dims=["sample"])
    return result.total.values


def compute_metrics(
    y_true,
    y_pred,
    dist,
    y_train,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
):
    """Compute the MAE, MAESS, CRPS and CRPSS for one model, with 95% CIs.

    The skill scores are computed relative to a baseline: the median of
    `y_true` for the MAE, and the baseline distribution built from `y_train`
    for the CRPS. If the model has no predictive distribution, the CRPS and
    CRPSS fall back to the MAE and MAESS. The per-sample errors are computed
    once, and the confidence intervals are percentile intervals over
    bootstrap resamples of the test samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Point predictions.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution or None
        Predictive distributions. If None, only point metrics are used.
    y_train : array-like of shape (n_train_samples,)
        Training targets used to build the baseline distribution.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Seed of the bootstrap random generator. The same seed gives the same
        resamples for every model.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Dictionary with keys "mae", "maess", "crps" and "crpss". Each value
        is a tuple (estimate, lower, upper) with the estimate computed on
        the full sample.

    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    abs_err = np.abs(y_true_arr - np.asarray(y_pred, dtype=float).ravel())

    if dist is not None:
        crps_i = sample_crps(y_true_arr, dist)
        baseline = baseline_distribution(y_train, len(y_true_arr))
        baseline_crps_i = sample_crps(y_true_arr, baseline)

    def statistic(idx):
        y = y_true_arr[idx]
        mae = abs_err[idx].mean()
        maess = 1 - mae / np.abs(y - np.quantile(y, 0.5)).mean()
        if dist is None:
            return np.array([mae, maess, mae, maess])
        crps = crps_i[idx].mean()
        crpss = 1 - crps / baseline_crps_i[idx].mean()
        return np.array([mae, maess, crps, crpss])

    estimates = statistic(np.arange(len(y_true_arr)))
    draws = bootstrap_statistic(statistic, len(y_true_arr), n_boot, seed)
    lower, upper = percentile_interval(draws)

    names = ["mae", "maess", "crps", "crpss"]
    return {
        name: (float(estimates[i]), float(lower[i]), float(upper[i]))
        for i, name in enumerate(names)
    }


def compute_sample_cdfs(dist, indices, grid_y):
    """Extract the CDF at selected indices.

    Parameters
    ----------
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Distribution object.
    indices : dict[int, int]
        Dictionary with {index: daoh}.
    grid_y : array-like
        Grid points to evaluate the CDF on for NGBoost. Ignored for ordboost
        distributions, which use their own grid.

    Returns
    -------
    cdfs : dict[str, Any]
        Dictionary with:
        "grid_y": grid_y
        "cdfs": {daoh: cdf_array}

    """
    cdfs = {"grid_y": grid_y, "cdfs": {}}

    # Extract proper CDF grids
    if isinstance(dist, NGBoostDistAdapter):
        cdf_grid = cdf_grid_ngboost(dist, grid_y)
    else:
        cdf_grid = dist.grid_cdf

    for index, daoh in indices.items():  # Loop over indices
        cdfs["cdfs"][daoh] = cdf_grid[index]
    return cdfs


def save_table(latex_str, path: str | Path = "table.txt"):
    """Write a LaTeX table string to a text file.

    Parameters
    ----------
    latex_str : str
        LaTeX source of the table.
    path : str or Path, default="table.txt"
        Destination file. Overwritten if it exists.

    """
    with open(path, "w") as f:
        f.write(latex_str + "\n")


def pit_histogram(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    mapper: BaseBinMapper | None = None,
    n_bins: int = 20,
) -> dict:
    """Compute the PIT histogram and alpha score for a model.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution or NGBoostDistAdapter
        Predicted distributions.
    mapper : BaseBinMapper, optional
        Bin mapper of the OrdBoost model. Required unless `dist` is an
        NGBoostDistAdapter.
    n_bins : int, default=20
        Number of histogram bins.

    Returns
    -------
    dict
        Dictionary with keys "alpha" (the PIT alpha score) and "hist_values"
        (the histogram values).

    Raises
    ------
    ValueError
        If `dist` is not an NGBoostDistAdapter and `mapper` is None.

    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(dist, NGBoostDistAdapter):
        pit = pit_diagnostics_ngboost(y_true_arr, dist)

    elif mapper is not None:
        pit = pit_diagnostics(y_true_arr, dist, mapper)

    else:
        raise ValueError("mapper needs to be provided with OrdBoost model.")

    alpha = pit.alpha_score()
    hist_values = pit.hist_values(n_bins)

    return {"alpha": alpha, "hist_values": hist_values}


def marginal_calibration(y_true, dist, grid_y, mapper):
    """Compute the marginal calibration curve for a model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.
    grid_y : array-like
        Grid to evaluate the curve on. Only used for NGBoost.
    mapper : BaseBinMapper
        Bin mapper of the OrdBoost model. Only used for ordboost
        distributions.

    Returns
    -------
    tuple
        The grid and the difference between the empirical CDF and the mean
        predicted CDF at each grid point.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    if isinstance(dist, NGBoostDistAdapter):
        results = marginal_calibration_curve_ngboost(y_true_arr, dist, grid_y)
    else:
        results = marginal_calibration_curve(y_true_arr, dist, mapper)
    return results


def coverage_sharpness_curve(y_true, dist, coverage_levels):
    """Compute coverage, sharpness and Winkler score across coverage levels.

    The alpha convention matches ordboost.metrics: alpha = 1 - coverage / 100,
    e.g. coverage=90 gives alpha=0.10.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.
    coverage_levels : array-like
        Nominal central-interval coverage levels in percent (e.g. 90 for a
        90% interval).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys "coverage_levels", "empirical_coverage",
        "winkler" and "sharpness" (mean interval width), each with one value
        per coverage level.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    coverage_levels = np.asarray(coverage_levels, dtype=float)
    alphas = 1.0 - coverage_levels / 100.0

    empirical_coverage = np.empty_like(alphas)
    sharp = np.empty_like(alphas)
    wink = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        empirical_coverage[i] = interval_coverage(
            y_true_arr,
            dist,
            alpha=alpha,
        )
        sharp[i] = sharpness(dist, alpha=alpha)
        wink[i] = winkler_score(y_true_arr, dist, alpha)

    return {
        "coverage_levels": coverage_levels,
        "empirical_coverage": empirical_coverage,
        "winkler": wink,
        "sharpness": sharp,
    }


def interval_coverage(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    alpha: float = 0.10,
) -> float:
    """Compute empirical coverage rate for a central prediction interval.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution
        Predicted continuous distributions.
    alpha : float, default=0.10
        Tail significance level (e.g., alpha=0.10 specifies a 90% interval).

    Returns
    -------
    float
        Proportion of true observations lying within predicted interval bounds.

    Raises
    ------
    ValueError
        If alpha not within (0.0, 1.0);

    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("Significance level 'alpha' must lie within (0.0, 1.0).")

    y_true_arr = np.asarray(y_true, dtype=float)
    lower, upper = dist.interval(alpha=alpha)

    covered = (
        (y_true_arr >= np.round(lower)) & (y_true_arr <= np.round(upper))
    ).astype(float)

    return float(np.mean(covered))


def estimate_patients(dist, ind: dict[int, int], alpha: float) -> dict[int, tuple]:
    """Estimate the DAOH and prediction interval of selected patients.

    The point prediction is the rounded median of the distribution.

    Parameters
    ----------
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predictive distribution.
    ind : dict[int, int]
        Dictionary of indices to check with associated true DAOH.
    alpha : float
        Tail significance level of the prediction interval (e.g. 0.05 for a
        95% interval).

    Returns
    -------
    dict[int, tuple]
        Dictionary with the index as key and a tuple (true DAOH, prediction,
        lower bound, upper bound) as value.

    """
    predictions = dist.median()
    results = {}
    l_bound, u_bound = dist.interval(alpha)

    for index, daoh in ind.items():
        results[index] = (
            daoh,
            np.round(predictions[index]),
            np.round(l_bound[index]),
            np.round(u_bound[index]),
        )

    return results


def _format_continuous(series):
    """Format a series as 'median (Q1--Q3)', ignoring NaNs.

    Parameters
    ----------
    series : pd.Series
        Continuous values to summarise.

    Returns
    -------
    str
        The formatted summary, or "--" if there are no valid values.

    """
    valid = series.dropna()
    if len(valid) == 0:
        return "--"
    median = np.median(valid)
    q1, q3 = np.percentile(valid, [25, 75])
    return f"{median:.1f} ({q1:.1f}--{q3:.1f})"


def _format_n_pct(n, total):
    """Format a count as 'n (pct%)' with a LaTeX-escaped percent sign.

    Parameters
    ----------
    n : int
        Count of interest.
    total : int
        Full size of the dataset (missing values included).

    Returns
    -------
    str
        The formatted count and percentage. The percentage is 0 if `total`
        is not positive.

    """
    pct = 100 * n / total if total > 0 else 0.0
    return f"{n:,} ({pct:.1f}\\%)"


def _categorical_rows(datasets, col, categories=None):
    """Build the table rows for a categorical column across datasets.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        Datasets to summarise.
    col : str
        Categorical column name.
    categories : list, optional
        Category order. Defaults to the sorted union of the categories found
        in the datasets.

    Returns
    -------
    list[tuple[str, list[str]]]
        One (label, cells) row per category, plus a trailing "Missing" row
        if any dataset has NaNs in this column.

    """
    if categories is None:
        categories = sorted(
            set().union(*[set(d[col].dropna().unique()) for d in datasets])
        )

    rows = []
    for cat in categories:
        counts = [int((d[col] == cat).sum()) for d in datasets]
        totals = [len(d) for d in datasets]
        rows.append((str(cat), [_format_n_pct(n, t) for n, t in zip(counts, totals)]))

    missing = [int(d[col].isna().sum()) for d in datasets]
    if any(m > 0 for m in missing):
        totals = [len(d) for d in datasets]
        rows.append(("Missing", [_format_n_pct(n, t) for n, t in zip(missing, totals)]))

    return rows


def generate_patient_characteristics_table(
    X_train,
    X_test,
    continuous_features,
    categorical_features,
    feature_labels=None,
    category_orders=None,
    caption="Patient characteristics by dataset.",
    label="tab:patient_chars",
):
    """Build a LaTeX longtable/booktabs patient characteristics table.

    The table compares the train and test splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with the original (pre-OrdinalEncoder) categorical
        values, so category labels render as text rather than integer codes.
    X_test : pd.DataFrame
        Test features, with the same format as `X_train`.
    continuous_features : list[str]
        Columns summarised as median (IQR). Only the first one is used, and
        it is reported as the age.
    categorical_features : list[str]
        Columns summarised as one row per category, N (%).
    feature_labels : dict[str, str], optional
        Column name -> display name. Defaults to the column name.
    category_orders : dict[str, list], optional
        Column name -> explicit category order, e.g. {"trauma": ["No","Yes"]},
        so binary features don't sort alphabetically as ["No","Yes"] vs
        ["Yes","No"] depending on the data.
    caption : str, default="Patient characteristics by dataset."
        Table caption.
    label : str, default="tab:patient_chars"
        LaTeX label of the table.

    Returns
    -------
    str
        Complete LaTeX longtable source.
    """
    feature_labels = feature_labels or {}
    category_orders = category_orders or {}
    datasets = [X_train, X_test]
    n_train, n_test = (len(d) for d in datasets)

    header = (
        r"\textbf{Features} & "
        rf"\shortstack[l]{{\textbf{{Train (2010--2022)}} \\ N = {n_train:,}}} & "
        rf"\shortstack[l]{{\textbf{{Test (2023--2024)}} \\ N = {n_test:,}}} \\"
    )
    lines = [
        r"\begin{longtable}{l l l}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule",
        header,
        r"\midrule\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{l}{\textit{(continued)}} \\",
        r"\toprule",
        header,
        r"\midrule\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    # Age first, as specified
    age_col = continuous_features[0]
    age_label = feature_labels.get(age_col, "Age")
    cells = [_format_continuous(d[age_col]) for d in datasets]
    lines.append(rf"{age_label}, median (IQR) & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")

    for col in categorical_features:
        label_txt = feature_labels.get(col, col)
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{label_txt}}}}} \\")
        for cat_label, cells in _categorical_rows(
            datasets, col, category_orders.get(col)
        ):
            lines.append(rf"\quad {cat_label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)
