"""Score fight records with the trained Random Forest, CatBoost, or ensemble model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling.ufc_modeling import (
    DEFAULT_ARTIFACTS_RELATIVE_DIR,
    build_feature_frame,
    find_project_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score fights with the trained models."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Parquet file matching the fight_dataset.parquet schema; "
            "target is optional."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "catboost", "ensemble"],
        default="catboost",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/modeling/predictions/new_fight_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ensemble-catboost-weight",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.ensemble_catboost_weight <= 1.0:
        raise ValueError(
            "--ensemble-catboost-weight must be between 0 and 1."
        )

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root()
    )

    input_path = (
        args.input
        if args.input.is_absolute()
        else project_root / args.input
    )
    output_path = (
        args.output
        if args.output.is_absolute()
        else project_root / args.output
    )

    df = pd.read_parquet(input_path)
    X, _, categorical_features = build_feature_frame(df)

    artifacts = project_root / DEFAULT_ARTIFACTS_RELATIVE_DIR
    rf_path = artifacts / "models" / "random_forest_pipeline.joblib"
    cat_path = artifacts / "models" / "catboost_model.cbm"

    probabilities: np.ndarray

    if args.model in {"random_forest", "ensemble"}:
        random_forest = joblib.load(rf_path)
        rf_probabilities = random_forest.predict_proba(X)[:, 1]

    if args.model in {"catboost", "ensemble"}:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is not installed. Run: pip install catboost"
            ) from exc

        catboost = CatBoostClassifier()
        catboost.load_model(cat_path)
        cat_probabilities = catboost.predict_proba(X)[:, 1]

    if args.model == "random_forest":
        probabilities = rf_probabilities
    elif args.model == "catboost":
        probabilities = cat_probabilities
    else:
        weight = args.ensemble_catboost_weight
        probabilities = (
            weight * cat_probabilities
            + (1.0 - weight) * rf_probabilities
        )

    result_columns = [
        column
        for column in [
            "fight_id",
            "event_date",
            "event_name",
            "weight_class",
            "fighter_a_id",
            "fighter_a_name",
            "fighter_b_id",
            "fighter_b_name",
            "target",
        ]
        if column in df.columns
    ]

    result = df[result_columns].copy()
    result["model"] = args.model
    result["probability_fighter_a"] = probabilities
    result["probability_fighter_b"] = 1.0 - probabilities
    result["prediction"] = (
        result["probability_fighter_a"] >= 0.5
    ).astype("int64")

    if {
        "fighter_a_name",
        "fighter_b_name",
    }.issubset(result.columns):
        result["predicted_winner_name"] = np.where(
            result["prediction"].eq(1),
            result["fighter_a_name"],
            result["fighter_b_name"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    print(f"Predictions generated: {len(result):,}")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print()
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
