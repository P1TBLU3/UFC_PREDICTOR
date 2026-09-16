"""Predict upcoming UFC fights using the frozen final CatBoost model."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DEFAULT_UPCOMING = "data/processed/ufcstats/upcoming_fight_dataset.parquet"
DEFAULT_MODEL = "reports/final_test/final_catboost_model.cbm"
DEFAULT_CALIBRATOR = "reports/final_test/final_calibrator.pkl"
DEFAULT_CALIBRATION_DECISION = (
    "reports/final_model_optimization/frozen_calibration_decision.json"
)
DEFAULT_FEATURE_MANIFEST = (
    "reports/final_model_optimization/frozen_feature_manifest.json"
)
DEFAULT_OUTPUT_DIR = "reports/upcoming_predictions"


def find_project_root() -> Path:
    starts = [Path.cwd(), Path(__file__).resolve().parent]

    for start in starts:
        for path in [start, *start.parents]:
            if (path / "src").exists() and (path / "data").exists():
                return path

    raise FileNotFoundError(
        "Could not locate the UFC_PREDICTOR project root. "
        "Run this module from inside the project."
    )


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )
    return np.log(probabilities / (1.0 - probabilities))


def prepare_frame(
    dataframe: pd.DataFrame,
    model_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    missing = [column for column in model_features if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Upcoming dataset does not match the frozen feature manifest. "
            "Missing columns:\n  - "
            + "\n  - ".join(missing)
        )

    frame = dataframe[model_features].copy()

    for column in categorical_features:
        frame[column] = (
            frame[column]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    return frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the frozen final UFC model and classify the upcoming fights."
        )
    )
    parser.add_argument(
        "--upcoming",
        default=DEFAULT_UPCOMING,
        help=f"Upcoming-fight Parquet file. Default: {DEFAULT_UPCOMING}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Prediction output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    root = find_project_root()

    upcoming_path = root / args.upcoming
    model_path = root / DEFAULT_MODEL
    calibrator_path = root / DEFAULT_CALIBRATOR
    calibration_path = root / DEFAULT_CALIBRATION_DECISION
    manifest_path = root / DEFAULT_FEATURE_MANIFEST
    output_dir = root / args.output_dir

    required = [
        upcoming_path,
        model_path,
        calibration_path,
        manifest_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required prediction artifacts are missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    print(f"Reading upcoming fights: {upcoming_path}")
    upcoming = pd.read_parquet(upcoming_path)

    if upcoming.empty:
        raise ValueError("The upcoming-fight dataset is empty.")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    with calibration_path.open("r", encoding="utf-8") as file:
        calibration_decision = json.load(file)

    model_features = list(manifest["all_model_features"])
    categorical_features = list(manifest["categorical_model_features"])
    calibration_method = str(calibration_decision["method"]).strip().lower()

    forbidden = [
        column
        for column in model_features
        if "effective" in column or "elo_decay_fraction" in column
    ]
    if forbidden:
        raise ValueError(
            "Frozen feature manifest contains forbidden Effective/decay Elo features: "
            f"{forbidden}"
        )

    X = prepare_frame(
        dataframe=upcoming,
        model_features=model_features,
        categorical_features=categorical_features,
    )

    print(f"Loading final CatBoost model: {model_path}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    probability_a_raw = model.predict_proba(X)[:, 1]

    if calibration_method == "none":
        probability_a = probability_a_raw.copy()

    elif calibration_method == "platt":
        if not calibrator_path.exists():
            raise FileNotFoundError(
                f"Frozen calibration method is Platt but calibrator is missing: "
                f"{calibrator_path}"
            )

        with calibrator_path.open("rb") as file:
            calibrator = pickle.load(file)

        probability_a = calibrator.predict_proba(
            probability_to_logit(probability_a_raw).reshape(-1, 1)
        )[:, 1]

    elif calibration_method == "isotonic":
        if not calibrator_path.exists():
            raise FileNotFoundError(
                f"Frozen calibration method is isotonic but calibrator is missing: "
                f"{calibrator_path}"
            )

        with calibrator_path.open("rb") as file:
            calibrator = pickle.load(file)

        probability_a = calibrator.predict(probability_a_raw)

    else:
        raise ValueError(
            f"Unknown frozen calibration method: {calibration_method!r}"
        )

    result = upcoming[
        [
            column
            for column in [
                "fight_id",
                "event_id",
                "event_name",
                "event_date",
                "weight_class",
                "fighter_a_id",
                "fighter_a_name",
                "fighter_b_id",
                "fighter_b_name",
            ]
            if column in upcoming.columns
        ]
    ].copy()

    result["probability_fighter_a_raw"] = probability_a_raw
    result["probability_fighter_a"] = probability_a
    result["probability_fighter_b"] = 1.0 - probability_a
    result["predicted_target"] = (probability_a >= 0.5).astype(int)

    result["predicted_winner"] = np.where(
        result["predicted_target"].eq(1),
        result["fighter_a_name"],
        result["fighter_b_name"],
    )

    result["confidence"] = np.maximum(
        result["probability_fighter_a"],
        result["probability_fighter_b"],
    )

    result = result.sort_values(
        ["event_date", "event_name", "fight_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_csv = output_dir / f"upcoming_predictions_{timestamp}.csv"
    latest_csv = output_dir / "upcoming_predictions.csv"
    latest_parquet = output_dir / "upcoming_predictions.parquet"

    result.to_csv(timestamped_csv, index=False)
    result.to_csv(latest_csv, index=False)
    result.to_parquet(latest_parquet, index=False)

    print()
    print("=" * 100)
    print("UPCOMING UFC PREDICTIONS")
    print("=" * 100)

    display_columns = [
        "event_date",
        "event_name",
        "fighter_a_name",
        "fighter_b_name",
        "probability_fighter_a",
        "probability_fighter_b",
        "predicted_winner",
        "confidence",
    ]

    print(
        result[display_columns].to_string(
            index=False,
            formatters={
                "probability_fighter_a": lambda x: f"{x:.3f}",
                "probability_fighter_b": lambda x: f"{x:.3f}",
                "confidence": lambda x: f"{x:.3f}",
            },
        )
    )

    print()
    print(f"Fights classified: {len(result):,}")
    print(f"Calibration method: {calibration_method}")
    print(f"Latest CSV: {latest_csv}")
    print(f"Parquet: {latest_parquet}")
    print(f"Timestamped copy: {timestamped_csv}")


if __name__ == "__main__":
    main()
