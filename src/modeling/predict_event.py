"""Generate predictions for an upcoming UFC event with the trained models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/processed/ufcstats/upcoming_fight_dataset.parquet"
DEFAULT_OUTPUT = "artifacts/modeling/predictions/upcoming_event_predictions.parquet"


def find_project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for start in candidates:
        for path in [start, *start.parents]:
            if (path / "src").exists() and (path / "data").exists():
                return path
    raise FileNotFoundError(
        "Could not locate the UFC_PREDICTOR project root. "
        "Run the module from the project root."
    )


def apply_modeling_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reuse add_context_features() from the training pipeline when available.

    This preserves compatibility when training adds context features after
    fight_dataset.parquet is built.
    """
    try:
        from src.modeling.ufc_modeling import add_context_features
    except (ImportError, AttributeError):
        return df.copy()

    enriched = add_context_features(df.copy())
    if not isinstance(enriched, pd.DataFrame):
        raise TypeError(
            "add_context_features() must return a pandas.DataFrame."
        )
    return enriched


def filter_event(
    df: pd.DataFrame,
    event_id: str | None,
    event_name: str | None,
) -> pd.DataFrame:
    out = df.copy()

    if event_id:
        out = out[out["event_id"].astype("string") == str(event_id)]

    if event_name:
        out = out[
            out["event_name"]
            .astype("string")
            .str.contains(event_name, case=False, na=False, regex=False)
        ]

    if out.empty:
        raise ValueError("The selected filter contains no fights.")

    return out


def required_feature_names_from_sklearn(model) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]

    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for step in named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in names]

    return None


def prepare_frame_for_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    categorical_columns: Iterable[str] = (),
) -> pd.DataFrame:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Features required by the model are missing:\n  - "
            + "\n  - ".join(missing)
        )

    X = df[columns].copy()
    categorical = set(categorical_columns)

    for col in X.columns:
        if col in categorical:
            X[col] = X[col].astype("string").fillna("Unknown").astype(str)
        elif pd.api.types.is_bool_dtype(X[col].dtype):
            X[col] = X[col].astype(float)
        elif pd.api.types.is_numeric_dtype(X[col].dtype):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    return X


def predict_random_forest(
    df: pd.DataFrame,
    model_path: Path,
) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {model_path}\n"
            "Run python -m src.modeling.train_models first"
        )

    model = joblib.load(model_path)

    required = required_feature_names_from_sklearn(model)
    if required:
        X = prepare_frame_for_columns(df, required)
    else:
        # ColumnTransformer selects the required columns from the full DataFrame.
        X = df.copy()

    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Random Forest predict_proba() did not return two classes.")

    classes = getattr(model, "classes_", None)
    if classes is None and getattr(model, "named_steps", None):
        last_step = list(model.named_steps.values())[-1]
        classes = getattr(last_step, "classes_", None)

    if classes is not None:
        classes = list(classes)
        if 1 not in classes:
            raise ValueError(f"The model does not contain positive class 1: {classes}")
        positive_idx = classes.index(1)
    else:
        positive_idx = 1

    return np.asarray(proba[:, positive_idx], dtype=float)


def predict_catboost(
    df: pd.DataFrame,
    model_path: Path,
) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(
            f"CatBoost model not found: {model_path}\n"
            "Run python -m src.modeling.train_models first"
        )

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    required = [str(x) for x in model.feature_names_]
    if not required:
        raise ValueError(
            "The CatBoost model has no feature_names_; "
            "training feature order cannot be verified."
        )

    cat_indices = list(model.get_cat_feature_indices())
    cat_columns = [required[i] for i in cat_indices]

    X = prepare_frame_for_columns(
        df,
        required,
        categorical_columns=cat_columns,
    )

    # CatBoost requires strings for categorical values and float/NaN for numerics.
    for i, col in enumerate(required):
        if i not in cat_indices:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("CatBoost predict_proba() did not return two classes.")

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"The model does not contain positive class 1: {classes}")

    positive_idx = classes.index(1)
    return np.asarray(proba[:, positive_idx], dtype=float)


def build_prediction_frame(
    df: pd.DataFrame,
    probability_a: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    metadata_cols = [
        c
        for c in [
            "fight_id",
            "event_id",
            "event_name",
            "event_date",
            "fight_order",
            "weight_class",
            "fighter_a_id",
            "fighter_a_name",
            "fighter_b_id",
            "fighter_b_name",
        ]
        if c in df.columns
    ]

    out = df[metadata_cols].copy()
    out["probability_fighter_a"] = probability_a
    out["probability_fighter_b"] = 1.0 - probability_a
    out["prediction"] = (out["probability_fighter_a"] >= 0.5).astype(int)
    out["predicted_winner_name"] = np.where(
        out["prediction"].eq(1),
        out["fighter_a_name"],
        out["fighter_b_name"],
    )
    out["predicted_loser_name"] = np.where(
        out["prediction"].eq(1),
        out["fighter_b_name"],
        out["fighter_a_name"],
    )
    out["predicted_winner_probability"] = np.maximum(
        out["probability_fighter_a"],
        out["probability_fighter_b"],
    )
    out["model"] = model_name

    return out.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, True],
        na_position="last",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict an upcoming UFC event with the trained models."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Upcoming-fight dataset. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "random_forest", "ensemble"],
        default="catboost",
        help="Model to use. Default: catboost.",
    )
    parser.add_argument("--event-id", default=None)
    parser.add_argument(
        "--event-name",
        default=None,
        help="Text filter for the event name.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Parquet path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    root = find_project_root()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not input_path.exists():
        raise FileNotFoundError(
            f"Upcoming-fight dataset not found: {input_path}\n"
            "Create it first with:\n"
            "  python -m src.features.build_upcoming_fight_dataset"
        )

    print(f"Reading upcoming-fight dataset: {input_path}")
    df = pd.read_parquet(input_path)

    required_metadata = {
        "fighter_a_name",
        "fighter_b_name",
        "fighter_a_id",
        "fighter_b_id",
        "event_name",
    }
    missing_metadata = sorted(required_metadata - set(df.columns))
    if missing_metadata:
        raise ValueError(
            "The upcoming-fight dataset is missing required columns:\n  - "
            + "\n  - ".join(missing_metadata)
        )

    df = filter_event(df, args.event_id, args.event_name)

    # Apply the same context transformation used during training.
    model_df = apply_modeling_context(df)

    model_dir = root / "artifacts" / "modeling" / "models"
    rf_path = model_dir / "random_forest_pipeline.joblib"
    cat_path = model_dir / "catboost_model.cbm"

    if args.model == "random_forest":
        probability_a = predict_random_forest(model_df, rf_path)
        model_name = "random_forest"

    elif args.model == "catboost":
        probability_a = predict_catboost(model_df, cat_path)
        model_name = "catboost"

    else:
        p_rf = predict_random_forest(model_df, rf_path)
        p_cat = predict_catboost(model_df, cat_path)
        probability_a = (p_rf + p_cat) / 2.0
        model_name = "ensemble_rf_catboost"

    predictions = build_prediction_frame(
        df=df,
        probability_a=probability_a,
        model_name=model_name,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    predictions.to_csv(csv_path, index=False)

    print("\nPredictions:")
    for _, row in predictions.iterrows():
        print(
            f"- {row['fighter_a_name']} vs {row['fighter_b_name']}: "
            f"{row['predicted_winner_name']} "
            f"({row['predicted_winner_probability'] * 100:.1f}%)"
        )

    print(f"\nModel: {model_name}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
