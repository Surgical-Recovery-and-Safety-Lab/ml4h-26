#!/usr/bin/env python3

import argparse

from medpipe import MedpipeRegressor

from res_gen import generate_results


def run_model(
    model: str,
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main", description="Fit pipelines and generate figures for ml4h paper"
    )

    subparsers = parser.add_subparsers(prog="main")
    run_parser = subparsers.add_parser(
        "run",
        help="fit a model",
        description="Run ordboost or ngboost model",
    )

    run_parser.add_argument(
        "model",
        help="model to select",
        choices=["ordboost", "ngboost"],
    )

    gen_parser = subparsers.add_parser(
        "res-gen", help="generate results", description="Generate one or more figures"
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

    args = parser.parse_args()
    if hasattr(args, "model"):
        # The run subcommand was called
        run_model(
            model=args.model,
        )
    else:
        # The res-gen command was called
        generate_results(args.ordboost_version, args.ngboost_version)
