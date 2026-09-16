"""Command-line entry point for the standard Random Forest and CatBoost training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.ufc_modeling import (
    CatBoostConfig,
    RandomForestConfig,
    SplitConfig,
    find_project_root,
    train_and_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare Random Forest and CatBoost "
            "on fight_dataset.parquet."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="UFC_PREDICTOR project root. Detected automatically by default.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--cat-iterations",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--cat-learning-rate",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--cat-depth",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cat-early-stopping",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--train-end",
        default="2022-01-01",
    )
    parser.add_argument(
        "--validation-end",
        default="2024-01-01",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root()
    )

    rf_config = RandomForestConfig(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
    )

    cat_config = CatBoostConfig(
        iterations=args.cat_iterations,
        learning_rate=args.cat_learning_rate,
        depth=args.cat_depth,
        early_stopping_rounds=args.cat_early_stopping,
    )

    split_config = SplitConfig(
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    train_and_compare(
        project_root=project_root,
        rf_config=rf_config,
        cat_config=cat_config,
        split_config=split_config,
    )


if __name__ == "__main__":
    main()
