"""Plot functions for conference paper figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from medpipe.visualisation.themes import MedpipeTheme

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    }
)
THEME = MedpipeTheme()
COLOURS = {
    "ngboost": THEME.palette[0],
    "3-bins": THEME.palette[0],
    "mean": THEME.palette[0],
    "ordboost": THEME.palette[1],
    "median": THEME.palette[2],
    "5-bins": THEME.palette[2],
    "uniform": THEME.palette[3],
    "quantile": THEME.palette[4],
    "continuous": THEME.palette[4],
}


def plot_binned_vs_continuous(
    y_cont,
    y_binned,
    bin_edges,
    colors,
    n_fine_bins: int = 90,
    density: bool = False,
    log_scale: bool = True,
    ax=None,
    save_path=None,
):
    """Plot the continuous target and its binned version on the same
    DAOH-value x-axis, so the two distributions are directly comparable.

    Parameters
    ----------
    y_cont : array-like
        Continuous target values (e.g. DAOH-90 in days).
    y_binned : array-like
        Integer bin labels for the same samples, 0..n_bins-1.
    bin_edges : array-like
        Edges used to create y_binned. Length must be n_bins + 1.
    colors : list[str]
        Colors to plot with.
    n_fine_bins : int
        Number of bins for the continuous histogram.
    density : bool
        If True, plot binned bars as counts / bin_width, so bar *area*
        (not just height) is comparable across your very uneven bin
        widths. If False, plot raw counts.
    log_scale : bool
        Use a log y-axis.
    ax : matplotlib.axes.Axes, optional
    """
    y_cont = np.asarray(y_cont, dtype=float)
    y_binned = np.asarray(y_binned, dtype=int)
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)

    # Continuous distribution: fine fixed-width histogram.
    axes[0].hist(
        y_cont,
        bins=90,
        color=colors[0],
        edgecolor="black",
        label=r"Continuous $DAOH_{90}$",
    )

    # Binned distribution: bars positioned at the real bin_edges values,
    # each spanning its true [edges[k], edges[k+1]) width.
    counts = np.bincount(y_binned, minlength=n_bins).astype(float)
    widths = np.diff(edges)
    heights = counts / widths if density else counts

    axes[1].bar(
        edges[:-1],
        heights[1:],
        width=widths,
        align="edge",
        color=colors[1],
        edgecolor="black",
        label=r"Binned $DAOH_{90}$",
    )

    if log_scale:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    axes[1].set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    axes[0].set_ylabel("Count / bin width" if density else "Count", fontweight="bold")
    axes[1].set_ylabel("Count / bin width" if density else "Count", fontweight="bold")
    axes[0].set_title(
        r"Continuous and binned $\mathbf{DAOH_{90}}$ distributions", fontweight="bold"
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    axes[0].set_ylim(bottom=10.5)
    axes[1].set_ylim(bottom=10.5)

    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    if save_path is not None:
        fig.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_sample_cdfs(model_results, colors, title, save_path=None):
    """Figure 2: example CDFs for different DAOH values."""

    fig, ax = plt.subplots(figsize=(7, 5), sharex=True)
    grid_y = model_results["grid_y"]

    for i, (daoh, cdf) in enumerate(model_results["cdfs"].items()):
        ax.plot(
            grid_y,
            cdf,
            label=r"$DAOH_{90}$ = " + f"{daoh}",
            color=colors[i],
            linewidth=2,
        )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Median",
    )

    ax.set_ylim(top=1.4)
    ax.set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.set_title(f"{title} cumulative distribution functions", fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_pit_histogram_grouped(
    model_hists,
    colors,
    display_labels=None,
    n_bins=20,
    save_path=None,
    ax=None,
):
    """Randomized PIT histogram, one grouped bar per bin per region, on a
    single shared axis. Sized for single-column placement.
    """
    display_labels = display_labels or {}
    models = list(model_hists.keys())
    n_groups = len(models)

    style = {
        "ngboost": COLOURS["ngboost"],
        "ordboost": COLOURS["ordboost"],
        "uniform": COLOURS["uniform"],
        "quantile": COLOURS["quantile"],
        "continuous": COLOURS["continuous"],
        "median": COLOURS["median"],
        "mean": COLOURS["mean"],
        "3-bins": COLOURS["3-bins"],
        "5-bins": COLOURS["5-bins"],
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    bin_width = 1.0 / n_bins
    group_width = bin_width * 0.9
    bar_width = group_width / n_groups

    for i, model in enumerate(models):
        hist = model_hists[model]["hist_values"]
        bin_centres = hist["bin_centre"].values
        bin_lefts = bin_centres - bin_width / 2.0
        bar_starts = bin_lefts + i * bar_width + (bin_width - group_width) / 2.0

        label = (
            f"{display_labels.get(model, model)}"
            + r" $\alpha$ score: "
            + f"{model_hists[model]["alpha"].values:.3f}"
        )
        ax.bar(
            bar_starts,
            hist.values,
            width=bar_width,
            align="edge",
            color=style.get(model, None),
            edgecolor="black",
            linewidth=1.5,
            label=label,
        )

    ax.axhline(
        1.0 / n_bins,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Uniform",
    )

    ax.set_xlabel("PIT value", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_title("PIT histograms", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
    )

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_marginal_calibration(
    model_results, colors, display_labels, save_path=None, ax=None
):
    """Figure 4: marginal calibration plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    style = {
        "ngboost": dict(color=COLOURS["ngboost"]),
        "ordboost": dict(color=COLOURS["ordboost"]),
        "uniform": dict(color=COLOURS["uniform"]),
        "quantile": dict(color=COLOURS["quantile"]),
        "continuous": dict(color=COLOURS["continuous"]),
        "median": dict(color=COLOURS["median"]),
        "mean": dict(color=COLOURS["mean"]),
        "3-bins": dict(color=COLOURS["3-bins"]),
        "5-bins": dict(color=COLOURS["5-bins"]),
    }

    for label, results in model_results.items():
        ax.plot(
            results[0],
            results[1],
            linewidth=2,
            label=display_labels.get(label, label),
            **style.get(label, {}),
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    ax.set_ylabel(r"eCDF - $\mathbf{\overline{CDF}}$", fontweight="bold")
    ax.set_title("Marginal calibration curves", fontweight="bold")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_coverage(model_results, colors, display_labels=None, save_path=None):
    """Figure 5: coverage reliability, sharpness, score across
    nominal coverage levels, one line per region on each subplot.
    """
    display_labels = display_labels or {}
    style = {
        "ngboost": dict(color=COLOURS["ngboost"], marker="o"),
        "ordboost": dict(color=COLOURS["ordboost"], marker="s"),
        "uniform": dict(color=COLOURS["uniform"], marker="s"),
        "quantile": dict(color=COLOURS["quantile"], marker="s"),
        "continuous": dict(color=COLOURS["continuous"], marker="s"),
        "mean": dict(color=COLOURS["mean"], marker="s"),
        "median": dict(color=COLOURS["median"], marker="s"),
        "3-bins": dict(color=COLOURS["3-bins"], marker="s"),
        "5-bins": dict(color=COLOURS["5-bins"], marker="s"),
    }

    fig, ax = plt.subplots(figsize=(7, 5), sharex=True)

    for model, results in model_results.items():
        ax.plot(
            results["coverage_levels"],
            results["empirical_coverage"] * 100,
            label=display_labels.get(model, model),
            **style.get(model, {}),
            linewidth=2,
        )
    ax.plot(
        [0, 100], [0, 100], linestyle="--", color="black", linewidth=1.5, label="Ideal"
    )
    ax.set_xlabel("Nominal coverage (%)", fontweight="bold")
    ax.set_ylabel("Empirical coverage (%)", fontweight="bold")
    ax.set_title("Coverage reliability", fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_sharpness(model_results, colors, display_labels=None, save_path=None):
    """Figure 4: sharpness across nominal coverage levels, one line per
    model.
    """
    display_labels = display_labels or {}
    style = {
        "ngboost": dict(color=COLOURS["ngboost"], marker="o"),
        "ordboost": dict(color=COLOURS["ordboost"], marker="s"),
        "uniform": dict(color=COLOURS["uniform"], marker="s"),
        "quantile": dict(color=COLOURS["quantile"], marker="s"),
        "continuous": dict(color=COLOURS["continuous"], marker="s"),
        "mean": dict(color=COLOURS["mean"], marker="s"),
        "median": dict(color=COLOURS["median"], marker="s"),
        "3-bins": dict(color=COLOURS["3-bins"], marker="s"),
        "5-bins": dict(color=COLOURS["5-bins"], marker="s"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for model, results in model_results.items():
        ax.plot(
            results["coverage_levels"],
            results["sharpness"],
            label=display_labels.get(model, model),
            **style.get(model, {}),
            linewidth=2,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Nominal coverage (%)", fontweight="bold")
    ax.set_ylabel("Mean interval width (days)", fontweight="bold")
    ax.set_title("Sharpness", fontweight="bold")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_winkler(model_results, colors, display_labels=None, save_path=None):
    """Figure 4: sharpness across nominal coverage levels, one line per
    model.
    """
    display_labels = display_labels or {}
    style = {
        "ngboost": dict(color=COLOURS["ngboost"], marker="o"),
        "ordboost": dict(color=COLOURS["ordboost"], marker="s"),
        "uniform": dict(color=COLOURS["uniform"], marker="s"),
        "quantile": dict(color=COLOURS["quantile"], marker="s"),
        "continuous": dict(color=COLOURS["continuous"], marker="s"),
        "mean": dict(color=COLOURS["mean"], marker="s"),
        "median": dict(color=COLOURS["median"], marker="s"),
        "3-bins": dict(color=COLOURS["3-bins"], marker="s"),
        "5-bins": dict(color=COLOURS["5-bins"], marker="s"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for model, results in model_results.items():
        ax.plot(
            results["coverage_levels"],
            results["winkler"],
            label=display_labels.get(model, model),
            **style.get(model, {}),
            linewidth=2,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Nominal coverage (%)", fontweight="bold")
    ax.set_ylabel("Winkler score", fontweight="bold")
    ax.set_title("Winkler score", fontweight="bold")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
