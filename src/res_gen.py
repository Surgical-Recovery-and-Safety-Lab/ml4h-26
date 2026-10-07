from pathlib import Path
from typing import Literal

import numpy as np
from medpipe import MedpipeRegressor
from medpipe.visualisation import themes
from ordboost.mappers import BaseBinMapper
from ordboost.metrics import pit_diagnostics

from src.helpers import (
    compute_metrics,
    compute_sample_cdfs,
    coverage_sharpness_curve,
    estimate_patients,
    generate_patient_characteristics_table,
    marginal_calibration,
    pit_histogram,
    save_table,
    wrap_ngboost_pred_dist,
)
from src.plotting import (
    plot_binned_vs_continuous,
    plot_coverage,
    plot_marginal_calibration,
    plot_pit_histogram_grouped,
    plot_sample_cdfs,
    plot_sharpness,
    plot_winkler,
)

THEME = themes.MedpipeTheme()
BINNING_EXP_ORDER = ["3-bins", "5-bins", "uniform", "quantile", "ordboost"]
MAPPERS_EXP_ORDER = ["mean", "median", "uniform", "continuous", "ordboost"]


def generate_results(
    ordboost_version: str,
    ngboost_version: str,
) -> None:
    """Main result generating function saved in the ordboost artifact directory.

    Parameters
    ----------
    ordboost_version : str
        Version of the ordboost model to load.
    ngboost_version : str
        Version of the ngboost model to load.

    """
    ordboost_pipe = MedpipeRegressor.load(f"artifacts/ordboost/{ordboost_version}")
    ngboost_pipe = MedpipeRegressor.load(f"artifacts/ngboost/{ngboost_version}")

    ordboost_pipe._orchestrator.prepare_data()

    (ordboost_pipe.run_dir / "results").mkdir(parents=True, exist_ok=True)
    (ngboost_pipe.run_dir / "results").mkdir(parents=True, exist_ok=True)
    (ordboost_pipe.run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (ngboost_pipe.run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Extract the data
    X_test = ordboost_pipe.data_split.X_test
    y_test = ordboost_pipe.data_split.y_test.to_numpy().squeeze()

    X_train = ordboost_pipe.data_split.X_train
    y_train = ordboost_pipe.data_split.y_train.to_numpy().squeeze()

    # Create the CDF distributions
    ngboost_dist = wrap_ngboost_pred_dist(
        ngboost_pipe.models["DAOH_90"].predict_dist(
            X_test,
        )
    )
    ordboost_dist = ordboost_pipe.models["DAOH_90"].predict_dist(
        X_test,
    )

    models = {
        "ngboost": ngboost_dist,
        "ordboost": ordboost_dist,
    }
    display_labels = {"ngboost": "NGBoost", "ordboost": "OrdBoost"}
    mapper = ordboost_pipe.models["DAOH_90"]["regressor"].mapper_

    # Generate data tables
    generate_data_table(ordboost_pipe)
    # Generate the data distribution figure (fig 1)
    generate_data_distribution(
        y_test,
        ordboost_pipe,
        mapper,
    )

    results = {}
    order = ["ngboost", "ordboost"]
    for model, dist in models.items():
        results[model] = compute_metrics(
            y_test,
            np.round(dist.median()),
            dist,
            y_train,
        )

    generate_result_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "results/table.txt",
    )

    grid_y = ordboost_dist.grid_y  # Extracted for NGBoost to match
    indices = {298960: 0, 340896: 20, 26586: 70, 101: 85, 20: 89}

    results = {}
    for model, dist in models.items():
        results[model] = compute_sample_cdfs(dist, indices, grid_y)

    plot_sample_cdfs(
        results["ordboost"],
        colors=THEME.palette,
        title=display_labels["ordboost"],
        save_path=ordboost_pipe.run_dir / "plots/example_cdfs.png",
    )
    plot_sample_cdfs(
        results["ngboost"],
        colors=THEME.palette,
        title=display_labels["ngboost"],
        save_path=ngboost_pipe.run_dir / "plots/example_cdfs.png",
    )

    results = {}
    alpha = 0.05
    for model, dist in models.items():
        results[model] = estimate_patients(dist, indices, alpha)

    generate_prediction_table(
        results,
        interval=(1 - alpha) * 100,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "results/prediction_table.txt",
    )

    results = {}
    for model, dist in models.items():
        results[model] = pit_histogram(y_test, dist, mapper)

    plot_pit_histogram_grouped(
        results,
        colors=THEME.palette,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/pit_histogram.png",
    )

    results = {}
    for model, dist in models.items():
        results[model] = marginal_calibration(y_test, dist, grid_y, mapper=mapper)

    plot_marginal_calibration(
        results,
        colors=THEME.palette,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/marginal_calibration.png",
    )

    coverage_levels = np.arange(10, 96, 5)  # 10, 15, ..., 90

    results = {}
    for model, dist in models.items():
        results[model] = coverage_sharpness_curve(y_test, dist, coverage_levels)

    plot_coverage(
        results,
        THEME.palette,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/coverage.png",
    )

    plot_sharpness(
        results,
        THEME.palette,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/sharpness.png",
    )


def generate_experiment_results(experiment: Literal["binning", "mappers"]) -> None:
    """Experiment result generating function.

    The results are saved in the experiments/{experiment} plots and
    results folders.

    Parameters
    ----------
    experiment : str, {"binning", "mappers"}
        Experiment to generate results for.

    Raises
    ------
    FileNotFoundError
        If the artifacts directory or the subfolders are not found.

    """
    # Check that the artifacts folder exists and has 5 subfolders
    src_dir = Path(f"experiments/{experiment}")
    artifacts_dir = src_dir / "artifacts"

    if not artifacts_dir.is_dir():
        msg = f"Missing directory: {artifacts_dir}"
        raise FileNotFoundError(msg)

    expected = {f"v{i}" for i in range(1, 6)}
    found = {p.name for p in artifacts_dir.iterdir() if p.is_dir()}
    if missing := expected - found:
        msg = f"{artifacts_dir} is missing subfolders: {sorted(missing)}. "
        "Run the experiment command with the --run flag first."
        raise FileNotFoundError(msg)

    pipes = [MedpipeRegressor.load(artifacts_dir / f"v{i}") for i in range(1, 6)]

    # Extract the data
    pipes[0]._orchestrator.prepare_data()
    X_test = pipes[0].data_split.X_test
    y_test = pipes[0].data_split.y_test.to_numpy().squeeze()
    y_train = pipes[0].data_split.y_train.to_numpy().squeeze()

    models = {}
    mappers = {}
    for i, pipe in enumerate(pipes):
        name = pipe.mp_config.meta.project_name
        dist = pipe.models["DAOH_90"].predict_dist(X_test)

        models[name] = dist
        mappers[name] = pipe.models["DAOH_90"]["regressor"].mapper_

    display_labels = {
        "3-bins": "3-bins",
        "5-bins": "5-bins",
        "quantile": "Quantile",
        "uniform": "Uniform",
        "ordboost": "OrdBoost",
        "mean": "Mean",
        "median": "Median",
        "continuous": "Continuous",
    }

    results = {}
    if experiment == "binning":
        order = BINNING_EXP_ORDER
    else:
        order = MAPPERS_EXP_ORDER

    for model, dist in models.items():
        results[model] = compute_metrics(
            y_test,
            np.round(dist.median()),
            dist,
            y_train,
        )

    generate_result_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "results/table.txt",
    )

    results = {}
    for model, dist in models.items():
        mapper = mappers[model]
        results[model] = pit_histogram(y_test, dist, mapper)

    plot_pit_histogram_grouped(
        results,
        colors=THEME.palette,
        display_labels=display_labels,
        save_path=src_dir / "plots/pit_histogram.png",
    )

    results = {}
    for model, dist in models.items():
        grid_y = dist.grid_y
        mapper = mappers[model]
        results[model] = marginal_calibration(y_test, dist, grid_y, mapper=mapper)

    plot_marginal_calibration(
        results,
        colors=THEME.palette,
        display_labels=display_labels,
        save_path=src_dir / "plots/marginal_calibration.png",
    )

    coverage_levels = np.arange(10, 96, 5)  # 10, 15, ..., 90

    results = {}
    for model, dist in models.items():
        results[model] = coverage_sharpness_curve(
            y_test, dist, coverage_levels, tolerance=1e-4
        )

    plot_winkler(
        results,
        THEME.palette,
        display_labels=display_labels,
        save_path=src_dir / "plots/winkler.png",
    )
    plot_coverage(
        results,
        THEME.palette,
        display_labels=display_labels,
        save_path=src_dir / "plots/coverage.png",
    )

    plot_sharpness(
        results,
        THEME.palette,
        display_labels=display_labels,
        save_path=src_dir / "plots/sharpness.png",
    )


def generate_result_table(results, order, display_labels, save_path):
    """results: dict[key -> {"mae","crps","crpss"}]
    order: list of keys controlling row order
    display_labels: dict[key -> LaTeX-escaped display string]
    save_path: str
    """
    lines = [
        r"\begin{tabular}{ccc}",
        r"\toprule",
        r"\textbf{Metrics} & "
        + " & ".join(rf"\textbf{{{display_labels[key]}}}" for key in order)
        + r" \\",
        r"\midrule",
    ]

    metric_rows = [
        ("MAE (days)", "mae", "{:.2f}"),
        ("MAESS", "maess", "{:.3f}"),
        ("CRPS (days)", "crps", "{:.2f}"),
        ("CRPSS", "crpss", "{:.3f}"),
    ]

    for row_label, metric_key, fmt in metric_rows:
        cells = [fmt.format(results[key][metric_key]) for key in order]
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)

    save_table(table, save_path)


def generate_prediction_table(results, interval, order, display_labels, save_path):
    """results: dict[key -> (DAOH, pred, l_bound, u_bound]
    order: list of keys controlling row order
    display_labels: dict[key -> LaTeX-escaped display string]
    save_path: str
    """
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
    ]
    indices = results[order[0]].keys()

    # Header row 1: DAOH label + model names
    header_row_1 = [r"$\mathbf{DAOH_{90}}$"]
    for key in order:
        label = display_labels.get(key, key)
        header_row_1.append(rf"\textbf{{{label}}}")
    lines.append(" & ".join(header_row_1) + r" \\")

    # Header row 2: units + median/PI subheader
    header_row_2 = [r"days"]
    for _ in order:
        header_row_2.append(rf"median [{interval:.0f}\% PI]")
    lines.append(" & ".join(header_row_2) + r" \\")

    lines.append(r"\midrule")

    for idx in indices:
        row_daoh = None
        row_cells = []
        for key in order:
            daoh, pred, lower, upper = results[key][idx]
            if row_daoh is None:
                row_daoh = daoh
            row_cells.append(f"{pred:.0f} [{lower:.0f}--{upper:.0f}]")

        lines.append(f"{row_daoh:.0f} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)
    save_table(table, save_path)


def generate_data_table(pipe: MedpipeRegressor) -> None:
    """Generate the patient characteristic table for data splits.

    Parameters
    ----------
    pipe : MedpipeRegressor
        MedpipeRegressor pipeline to use for loading splits and saving.

    """
    X_test = pipe.data_split.X_test
    X_train = pipe.data_split.X_train

    category_orders = {
        "TRAUMA": X_test["TRAUMA"].unique(),
        "PRIOR_CANCER": X_test["TRAUMA"].unique(),
        "SEX": X_test["SEX"].unique(),
        "ETHNICITY": X_test["ETHNICITY"].unique(),
    }

    latex_table = generate_patient_characteristics_table(
        X_train,
        X_test,
        continuous_features=["AGE"],
        categorical_features=[
            "SEX",
            "ETHNICITY",
            "ADMISSION_SOURCE",
            "ADMISSION_ACUITY",
            "CATEGORY_LEVEL_1",
            "CATEGORY_LEVEL_2",
            "TRAUMA",
            "PRIOR_CANCER",
            "ASA",
            "OP_SEVERITY",
        ],
        feature_labels={
            "ASA": "ASA score",
            "OP_SEVERITY": "Operation severity",
            "ETHNICITY": "Ethnicity",
            "ADMISSION_SOURCE": "Admission source",
            "ADMISSION_ACUITY": "Admission acuity",
            "TRAUMA": "Trauma",
            "SEX": "Sex at birth",
            "PRIOR_CANCER": "Prior cancer",
            "CATEGORY_LEVEL_1": "Specialty",
            "CATEGORY_LEVEL_2": "Sub-specialty",
        },
        category_orders=category_orders,
        caption="Patient characteristics by dataset split.",
        label="tab:patient_chars",
    )
    save_table(latex_table, pipe.run_dir / "results/table_patient_chars.txt")


def generate_data_distribution(
    y_test: np.ndarray, pipe: MedpipeRegressor, mapper: BaseBinMapper
) -> None:
    """Generate the data distribution before and after binning.

    y_test : np.ndarray of shape (n_samples,)
        Test data to plot.
    pipe : MedpipeRegressor
        MedpipeRegressor pipeline to save the data.
    mapper : BaseBinMapper
        Mapper used to extract the binned version of the data.

    """
    bin_edges = np.array(mapper.bin_edges)
    y_binned = mapper.digitize(y_test, bin_edges)
    primary_color = THEME.primary_color

    plot_bins = np.concatenate(([0, 1], bin_edges[2:] - 1, [89]))
    plot_binned_vs_continuous(
        y_test,
        y_binned,
        plot_bins,
        THEME.palette,
        save_path=pipe.run_dir / "plots/data_distributions.png",
    )
