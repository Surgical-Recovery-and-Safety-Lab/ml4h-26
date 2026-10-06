#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Literal

from res_gen import generate_results

from medpipe import MedpipeRegressor


def run_model(model: str) -> None:
    """Fit an NGBoost or OrdBoost model.

    Parameters
    ----------
    model : str, {"ordboost", "ngboost"}
        Name of the model to run.

    """
    config = f"DAOH_config_{model}.toml"
    dir = f"artifacts/{model}"

    pipe = MedpipeRegressor(config=config, base_artifact_dir=dir)
    pipe.run()
    # Make directories to save results in case they don't exist
    plot_path = pipe.run_dir / "plots"
    plot_path.mkdir(exist_ok=True)
    res_path = pipe.run_dir / "results"
    res_path.mkdir(exist_ok=True)


def run_experiment(
    experiment: Literal["binning"], run_flag: Literal["run", "results"]
) -> None:
    """Run the selected experiment.

    Parameters
    ----------
    experiment : str, {"binning"}
        Selected experiment to run.
    run_flag : str, {"run", "results"}
        Flag to select how to run the experiment. The `run` mode fits all the
        models in the experiment. The `results` mode generates the results, if
        the models have been fitted first.

    """
    src_dir = Path(f"experiments/{experiment}")
    pattern = "*.toml"  # Get only .toml files

    config_files = sorted(p for p in src_dir.glob(pattern) if p.is_file())

    for config_file in config_files:
        if run_flag == "run":
            pipe = MedpipeRegressor(
                config=config_file, base_artifact_dir=src_dir / "artifacts"
            )
            pipe.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main", description="Fit pipelines and generate figures for ml4h paper"
    )

    subparsers = parser.add_subparsers(prog="main")

    # Main model fitting parser
    run_parser = subparsers.add_parser(
        "run", help="fit a model", description="Run ordboost or ngboost model"
    )

    run_parser.add_argument(
        "model", help="model to select", choices=["ordboost", "ngboost"]
    )

    # Main results generation parser
    gen_parser = subparsers.add_parser(
        "results", help="generate results", description="Generate one or more figures"
    )
    gen_parser.add_argument(
        "ordboost_version",
        metavar="ordboost-version",
        help="version for the ordboost model (e.g. v1)",
        nargs="?",
        default="v1",
    )
    gen_parser.add_argument(
        "ngboost_version",
        metavar="ngboost-version",
        help="version for the ordboost model (e.g. v1)",
        nargs="?",
        default="v1",
    )

    # Experiment parser
    exp_parser = subparsers.add_parser(
        "experiment",
        help="run an experiment",
        description="Run one of the available experiments",
    )
    exp_parser.add_argument(
        "experiment", choices=["binning", "mappers"], help="experiment to run"
    )
    exp_group = exp_parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        "--run",
        action="store_true",
        help="fit all the models from the selected experiment",
    )
    exp_group.add_argument(
        "--results",
        action="store_true",
        help="generate all the results for the selected experiment",
    )

    args = parser.parse_args()

    if args.command == "run":
        # The run subcommand was called
        run_model(model=args.model)
    elif args.command == "results":
        # The results command was called
        generate_results(args.ordboost_version, args.ngboost_version)

    else:
        # The experiment command was called
        if args.run:
            run_flag = "run"
        else:
            run_flat = "results"
        run_experiment(experiment=args.experiment, run_flag=run_flag)
