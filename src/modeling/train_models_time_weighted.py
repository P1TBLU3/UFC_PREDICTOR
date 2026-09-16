"""Command-line entry point for experiments with time-weighted training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.time_weighted_training import (
    TimeWeightConfig,
    train_time_weighted_experiment,
)
from src.modeling.ufc_modeling import (
    CatBoostConfig,
    RandomForestConfig,
    SplitConfig,
    find_project_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and CatBoost with date-based fight weighting."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--weight-mode",
        choices=["none", "exclude_before", "step", "exponential"],
        default="none",
    )
    parser.add_argument("--cutoff-date", default="2005-01-01")
    parser.add_argument("--pre-cutoff-weight", type=float, default=0.25)
    parser.add_argument("--half-life-years", type=float, default=8.0)
    parser.add_argument("--minimum-weight", type=float, default=0.10)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test set. Keep disabled while comparing weighting strategies.",
    )
    parser.add_argument("--rf-estimators", type=int, default=800)
    parser.add_argument("--rf-max-depth", type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=4)
    parser.add_argument("--cat-iterations", type=int, default=2000)
    parser.add_argument("--cat-learning-rate", type=float, default=0.03)
    parser.add_argument("--cat-depth", type=int, default=5)
    parser.add_argument("--cat-early-stopping", type=int, default=150)
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--validation-end", default="2024-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root()
    )

    train_time_weighted_experiment(
        project_root=project_root,
        time_weight_config=TimeWeightConfig(
            mode=args.weight_mode,
            cutoff_date=args.cutoff_date,
            pre_cutoff_weight=args.pre_cutoff_weight,
            half_life_years=args.half_life_years,
            minimum_weight=args.minimum_weight,
        ),
        rf_config=RandomForestConfig(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
        ),
        cat_config=CatBoostConfig(
            iterations=args.cat_iterations,
            learning_rate=args.cat_learning_rate,
            depth=args.cat_depth,
            early_stopping_rounds=args.cat_early_stopping,
        ),
        split_config=SplitConfig(
            train_end=args.train_end,
            validation_end=args.validation_end,
        ),
        evaluate_test=args.evaluate_test,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
