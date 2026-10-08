"""Helper functions for conference paper results: CRPS via the `scores`
package, region filtering, and LaTeX table generation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scores.probability import Pit, PitFcstAtObs
from sklearn.metrics import mean_absolute_error
from src.helpers_ngboost import (
    NGBoostDistAdapter,
    cdf_grid_ngboost,
    crps_ngboost,
    marginal_calibration_curve_ngboost,
    pit_diagnostics_ngboost,
)

from medpipe import MedpipeRegressor
from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper
from ordboost.metrics import (
    baseline_distribution,
    crps_score,
    crps_skill_score,
    interval_coverage_rate,
    marginal_calibration_curve,
    pit_diagnostics,
    sharpness,
    winkler_score,
)


def wrap_ngboost_pred_dist(ngb_pred_dist) -> NGBoostDistAdapter:
    """Wrap NGBRegressor.pred_dist(X)'s output in an NGBoostDistAdapter."""
    return NGBoostDistAdapter(ngb_pred_dist.dist)


def compute_metrics(y_true, y_pred, dist, y_train):
    """MAE, CRPS, CRPSS for predictions from one model."""
    mae = mean_absolute_error(y_true, y_pred)
    medians = np.quantile(y_true, 0.5) * np.ones((len(y_true),))
    maess = 1 - (mae / mean_absolute_error(y_true, medians))

    if dist is not None:
        if isinstance(dist, NGBoostDistAdapter):
            crps_val = float(crps_ngboost(y_true, dist))
        else:
            crps_val = float(crps_score(y_true, dist))
        dist_baseline = baseline_distribution(y_train, len(y_true))
        baseline_cprs = float(crps_score(y_true, dist_baseline))
        crpss_val = 1 - (crps_val / baseline_cprs)
    else:
        crps_val = mae
        crpss_val = maess

    return {
        "mae": mae,
        "maess": maess,
        "crps": crps_val,
        "crpss": crpss_val,
    }


def compute_sample_cdfs(dist, indices, grid_y):
    """Extracts the CDF at selected indices.

    Parameters
    ----------
    dist
        Distribution object.
    indices
        Dictionary with {index: daoh}.
    grid_y
        Grid points to evaluate CDF for NGBoost.

    Returns
    -------
    cdfs : dict[str, Any]
        Dictionary with:
        "grid_y": grid_y
        "cdfs": {daoh, cdf_array}

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
    with open(path, "w") as f:
        f.write(latex_str + "\n")


def pit_histogram(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    mapper: BaseBinMapper | None = None,
    n_bins: int = 20,
) -> dict:
    """Compute PIT histogram for a model."""
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
    """Marginal calibration for a model.

    Returns dict[label -> (grid_y, diff)]
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    if isinstance(dist, NGBoostDistAdapter):
        results = marginal_calibration_curve_ngboost(y_true_arr, dist, grid_y)
    else:
        results = marginal_calibration_curve(y_true_arr, dist, mapper)
    return results


def coverage_sharpness_curve(y_true, dist, coverage_levels, tolerance=0.0):
    """For each nominal central-interval coverage level (in percent, e.g.
    90 for a 90% interval), compute empirical coverage, sharpness (mean
    interval width).

    alpha convention (matches ordboost.metrics): alpha = 1 - coverage/100,
    e.g. coverage=90 -> alpha=0.10.

    Returns dict with keys "coverage_levels", "empirical_coverage",
    "sharpness".
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    coverage_levels = np.asarray(coverage_levels, dtype=float)
    alphas = 1.0 - coverage_levels / 100.0

    empirical_coverage = np.empty_like(alphas)
    sharp = np.empty_like(alphas)
    wink = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        empirical_coverage[i] = interval_coverage_rate(
            y_true_arr,
            dist,
            alpha=alpha,
            tolerance=tolerance,
        )
        sharp[i] = sharpness(dist, alpha=alpha)
        wink[i] = winkler_score(y_true_arr, dist, alpha)

    return {
        "coverage_levels": coverage_levels,
        "empirical_coverage": empirical_coverage,
        "winkler": wink,
        "sharpness": sharp,
    }


def estimate_patients(dist, ind: dict[int, int], alpha: float) -> dict[int, tuple]:
    """Estimates the patient's DAOH and interval using the median() function.

    Parameters
    ----------
    dist
        Predictive distribution.
    ind : dict[int, int]
        Dictionary of indices to check with associated true DAOH.
    alpha : float
        Prediction interval.

    Returns
    -------
    dict[int, tuple]
        Dictionary containing the index as key and the prediction with
        intervals as a tuple.

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
    """Median (IQR) as 'median (Q1--Q3)', ignoring NaNs."""
    valid = series.dropna()
    if len(valid) == 0:
        return "--"
    median = np.median(valid)
    q1, q3 = np.percentile(valid, [25, 75])
    return f"{median:.1f} ({q1:.1f}--{q3:.1f})"


def _format_n_pct(n, total):
    """'n (pct%)', percentage of the dataset's full N (missing included)."""
    pct = 100 * n / total if total > 0 else 0.0
    return f"{n:,} ({pct:.1f}\\%)"


def _categorical_rows(datasets, col, categories=None):
    """One (label, [cells]) row per category across datasets, plus a
    trailing Missing row if any dataset has NaNs for this column.
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
    """Build a LaTeX longtable/booktabs patient characteristics table
    across the train/recalibration/test splits.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature dataframes with ORIGINAL (pre-OrdinalEncoder) categorical
        values, so category labels render as text rather than integer codes.
    continuous_features : list[str]
        Columns summarised as median (IQR).
    categorical_features : list[str]
        Columns summarised as one row per category, N (%).
    feature_labels : dict[str, str], optional
        Column name -> display name. Defaults to the column name.
    category_orders : dict[str, list], optional
        Column name -> explicit category order, e.g. {"trauma": ["No","Yes"]},
        so binary features don't sort alphabetically as ["No","Yes"] vs
        ["Yes","No"] depending on the data.
    caption, label : str

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
