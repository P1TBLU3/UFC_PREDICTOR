"""Shared utilities for feature preparation, temporal splitting, model training, evaluation, and artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "target"
DEFAULT_DATASET_RELATIVE_PATH = Path(
    "data/processed/ufcstats/fight_dataset.parquet"
)
DEFAULT_ARTIFACTS_RELATIVE_DIR = Path("artifacts/modeling")

TRAIN_END = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2024-01-01")

CATEGORICAL_FEATURES = [
    "weight_class",
    "stance_matchup",
    "same_stance",
]

ENGINEERED_NUMERIC_FEATURES = [
    "total_fights_before",
    "min_fights_before",
    "max_fights_before",
    "total_fights_with_stats_before",
    "min_fights_with_stats_before",
    "total_stats_fight_time_seconds_before",
    "min_stats_coverage_before",
    "min_control_stats_coverage_before",
    "both_debutants",
    "one_debutant",
]


@dataclass(frozen=True)
class SplitConfig:
    train_end: str = "2022-01-01"
    validation_end: str = "2024-01-01"


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 800
    max_depth: int | None = 12
    min_samples_leaf: int = 4
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 5
    l2_leaf_reg: float = 10.0
    random_seed: int = 42
    early_stopping_rounds: int = 150
    verbose: int = 100


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root from the dataset path."""
    current = (start or Path.cwd()).resolve()

    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / DEFAULT_DATASET_RELATIVE_PATH).exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the UFC_PREDICTOR project root. Run the code from "
        "the project root or a subdirectory containing "
        f"{DEFAULT_DATASET_RELATIVE_PATH.as_posix()}."
    )


def load_dataset(project_root: Path) -> pd.DataFrame:
    """Load and validate the fight-level dataset."""
    path = project_root / DEFAULT_DATASET_RELATIVE_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    required = {
        "fight_id",
        "event_date",
        "fighter_a_id",
        "fighter_a_name",
        "fighter_b_id",
        "fighter_b_name",
        TARGET_COLUMN,
        "weight_class",
        "stance_matchup",
        "same_stance",
        "fighter_a_fights_before",
        "fighter_b_fights_before",
        "fighter_a_fights_with_stats_before",
        "fighter_b_fights_with_stats_before",
        "fighter_a_stats_fight_time_seconds_before",
        "fighter_b_stats_fight_time_seconds_before",
        "fighter_a_stats_coverage_before",
        "fighter_b_stats_coverage_before",
        "fighter_a_control_stats_coverage_before",
        "fighter_b_control_stats_coverage_before",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Required columns are missing from fight_dataset.parquet: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="raise")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype("int64")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1.")

    if df["fight_id"].duplicated().any():
        raise ValueError("fight_id must be unique in the fight-level dataset.")

    return df.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add symmetric experience and coverage features."""
    result = df.copy()

    a_fights = pd.to_numeric(
        result["fighter_a_fights_before"], errors="coerce"
    )
    b_fights = pd.to_numeric(
        result["fighter_b_fights_before"], errors="coerce"
    )

    a_stats_fights = pd.to_numeric(
        result["fighter_a_fights_with_stats_before"], errors="coerce"
    )
    b_stats_fights = pd.to_numeric(
        result["fighter_b_fights_with_stats_before"], errors="coerce"
    )

    a_stats_time = pd.to_numeric(
        result["fighter_a_stats_fight_time_seconds_before"],
        errors="coerce",
    )
    b_stats_time = pd.to_numeric(
        result["fighter_b_stats_fight_time_seconds_before"],
        errors="coerce",
    )

    a_stats_coverage = pd.to_numeric(
        result["fighter_a_stats_coverage_before"], errors="coerce"
    )
    b_stats_coverage = pd.to_numeric(
        result["fighter_b_stats_coverage_before"], errors="coerce"
    )

    a_control_coverage = pd.to_numeric(
        result["fighter_a_control_stats_coverage_before"],
        errors="coerce",
    )
    b_control_coverage = pd.to_numeric(
        result["fighter_b_control_stats_coverage_before"],
        errors="coerce",
    )

    result["total_fights_before"] = a_fights + b_fights
    result["min_fights_before"] = pd.concat(
        [a_fights, b_fights], axis=1
    ).min(axis=1)
    result["max_fights_before"] = pd.concat(
        [a_fights, b_fights], axis=1
    ).max(axis=1)

    result["total_fights_with_stats_before"] = (
        a_stats_fights + b_stats_fights
    )
    result["min_fights_with_stats_before"] = pd.concat(
        [a_stats_fights, b_stats_fights], axis=1
    ).min(axis=1)

    result["total_stats_fight_time_seconds_before"] = (
        a_stats_time + b_stats_time
    )
    result["min_stats_coverage_before"] = pd.concat(
        [a_stats_coverage, b_stats_coverage], axis=1
    ).min(axis=1)
    result["min_control_stats_coverage_before"] = pd.concat(
        [a_control_coverage, b_control_coverage], axis=1
    ).min(axis=1)

    result["both_debutants"] = (
        a_fights.eq(0) & b_fights.eq(0)
    ).astype("int64")
    result["one_debutant"] = (
        a_fights.eq(0) ^ b_fights.eq(0)
    ).astype("int64")

    return result


def build_feature_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build X from A-B differences and symmetric context features.

    Exclude identifiers, names, dates, and post-fight outcomes.
    """
    enriched = add_context_features(df)

    difference_features = sorted(
        column for column in enriched.columns
        if column.startswith("diff_")
    )

    numeric_features = [
        *difference_features,
        *ENGINEERED_NUMERIC_FEATURES,
    ]

    missing_numeric = set(numeric_features) - set(enriched.columns)
    missing_categorical = set(CATEGORICAL_FEATURES) - set(enriched.columns)

    if missing_numeric or missing_categorical:
        raise ValueError(
            "Could not build every required feature. "
            f"Missing numerical features: {sorted(missing_numeric)}; "
            f"missing categorical features: {sorted(missing_categorical)}"
        )

    X = enriched[numeric_features + CATEGORICAL_FEATURES].copy()

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce").astype(float)

    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    return X, numeric_features, CATEGORICAL_FEATURES.copy()


def temporal_split(
    df: pd.DataFrame,
    split_config: SplitConfig = SplitConfig(),
) -> dict[str, pd.Index]:
    """Return train, validation, and test indices."""
    train_end = pd.Timestamp(split_config.train_end)
    validation_end = pd.Timestamp(split_config.validation_end)

    train_idx = df.index[df["event_date"] < train_end]
    validation_idx = df.index[
        (df["event_date"] >= train_end)
        & (df["event_date"] < validation_end)
    ]
    test_idx = df.index[df["event_date"] >= validation_end]

    splits = {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx,
    }

    if any(len(index) == 0 for index in splits.values()):
        raise ValueError(
            "At least one temporal split is empty. "
            f"Sizes: { {k: len(v) for k, v in splits.items()} }"
        )

    return splits


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: RandomForestConfig = RandomForestConfig(),
) -> Pipeline:
    """Build the complete Random Forest pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight=None,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
    config: CatBoostConfig = CatBoostConfig(),
):
    """Train CatBoost with early stopping."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Run: "
            "pip install catboost"
        ) from exc

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.random_seed,
        allow_writing_files=False,
        verbose=config.verbose,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    return model


def metrics_from_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate classification and probability metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(y_true_array, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true_array, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true_array,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true_array, probabilities)
        ),
        "threshold": float(threshold),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def prediction_frame(
    source_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Create an interpretable prediction table."""
    result = source_df[
        [
            "fight_id",
            "event_date",
            "event_name",
            "weight_class",
            "fighter_a_id",
            "fighter_a_name",
            "fighter_b_id",
            "fighter_b_name",
            TARGET_COLUMN,
        ]
    ].copy()

    result["model"] = model_name
    result["probability_fighter_a"] = probabilities
    result["probability_fighter_b"] = 1.0 - probabilities
    result["prediction"] = (
        result["probability_fighter_a"] >= 0.5
    ).astype("int64")

    result["predicted_winner_name"] = np.where(
        result["prediction"].eq(1),
        result["fighter_a_name"],
        result["fighter_b_name"],
    )

    result["actual_winner_name"] = np.where(
        result[TARGET_COLUMN].eq(1),
        result["fighter_a_name"],
        result["fighter_b_name"],
    )

    result["correct"] = (
        result["prediction"] == result[TARGET_COLUMN]
    )

    return result


def save_confusion_matrix_plot(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a confusion matrix without additional dependencies."""
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["B wins", "A wins"],
        yticklabels=["B wins", "A wins"],
        xlabel="Prediction",
        ylabel="Actual outcome",
        title=title,
    )

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    output_path: Path,
    title: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Save the calibration curve and table."""
    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    rows: list[dict[str, float | int]] = []

    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue

        rows.append(
            {
                "bin": bin_id,
                "count": int(mask.sum()),
                "mean_predicted_probability": float(
                    probabilities[mask].mean()
                ),
                "observed_win_rate": float(
                    y_true_array[mask].mean()
                ),
            }
        )

    calibration = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_win_rate"],
        marker="o",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability for fighter_a",
        ylabel="Observed win frequency for fighter_a",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return calibration


def random_forest_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract feature importances from the Random Forest pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def catboost_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importances from the CatBoost model."""
    values = model.get_feature_importance()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def ensure_artifact_directories(
    project_root: Path,
) -> dict[str, Path]:
    """Create the output directories."""
    root = project_root / DEFAULT_ARTIFACTS_RELATIVE_DIR
    directories = {
        "root": root,
        "models": root / "models",
        "reports": root / "reports",
        "plots": root / "plots",
        "predictions": root / "predictions",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON with support for basic NumPy and pandas types."""

    def default(value: Any):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        raise TypeError(
            f"Unsupported JSON type: {type(value).__name__}"
        )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=default,
        ),
        encoding="utf-8",
    )


def train_and_compare(
    project_root: Path,
    rf_config: RandomForestConfig = RandomForestConfig(),
    cat_config: CatBoostConfig = CatBoostConfig(),
    split_config: SplitConfig = SplitConfig(),
) -> dict[str, Any]:
    """
    Run the complete workflow: load data, build features, split, train,
    evaluate, and save artifacts.
    """
    df = load_dataset(project_root)
    X, numeric_features, categorical_features = build_feature_frame(df)
    y = df[TARGET_COLUMN]
    splits = temporal_split(df, split_config)
    dirs = ensure_artifact_directories(project_root)

    X_train = X.loc[splits["train"]]
    y_train = y.loc[splits["train"]]

    X_validation = X.loc[splits["validation"]]
    y_validation = y.loc[splits["validation"]]

    X_test = X.loc[splits["test"]]
    y_test = y.loc[splits["test"]]

    print("Training Random Forest...")
    random_forest = build_random_forest_pipeline(
        numeric_features,
        categorical_features,
        rf_config,
    )
    random_forest.fit(X_train, y_train)

    rf_validation_prob = random_forest.predict_proba(
        X_validation
    )[:, 1]

    print("Training CatBoost...")
    catboost = train_catboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_features,
        cat_config,
    )
    cat_validation_prob = catboost.predict_proba(
        X_validation
    )[:, 1]

    validation_metrics = {
        "random_forest": metrics_from_probabilities(
            y_validation,
            rf_validation_prob,
        ),
        "catboost": metrics_from_probabilities(
            y_validation,
            cat_validation_prob,
        ),
    }

    selected_model = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["log_loss"],
    )

    # Evaluate the test set only after validation-based model selection.
    rf_test_prob = random_forest.predict_proba(X_test)[:, 1]
    cat_test_prob = catboost.predict_proba(X_test)[:, 1]

    test_metrics = {
        "random_forest": metrics_from_probabilities(
            y_test,
            rf_test_prob,
        ),
        "catboost": metrics_from_probabilities(
            y_test,
            cat_test_prob,
        ),
    }

    metrics_rows: list[dict[str, Any]] = []
    for split_name, group in [
        ("validation", validation_metrics),
        ("test", test_metrics),
    ]:
        for model_name, metrics in group.items():
            metrics_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    **metrics,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        dirs["reports"] / "model_metrics.csv",
        index=False,
    )

    validation_predictions = pd.concat(
        [
            prediction_frame(
                df.loc[splits["validation"]],
                rf_validation_prob,
                "random_forest",
            ),
            prediction_frame(
                df.loc[splits["validation"]],
                cat_validation_prob,
                "catboost",
            ),
        ],
        ignore_index=True,
    )
    validation_predictions.to_parquet(
        dirs["predictions"] / "validation_predictions.parquet",
        index=False,
    )

    test_predictions = pd.concat(
        [
            prediction_frame(
                df.loc[splits["test"]],
                rf_test_prob,
                "random_forest",
            ),
            prediction_frame(
                df.loc[splits["test"]],
                cat_test_prob,
                "catboost",
            ),
        ],
        ignore_index=True,
    )
    test_predictions.to_parquet(
        dirs["predictions"] / "test_predictions.parquet",
        index=False,
    )

    joblib.dump(
        random_forest,
        dirs["models"] / "random_forest_pipeline.joblib",
    )
    catboost.save_model(
        dirs["models"] / "catboost_model.cbm"
    )

    rf_importance = random_forest_feature_importance(
        random_forest
    )
    rf_importance.to_csv(
        dirs["reports"]
        / "feature_importance_random_forest.csv",
        index=False,
    )

    cat_importance = catboost_feature_importance(
        catboost,
        list(X.columns),
    )
    cat_importance.to_csv(
        dirs["reports"] / "feature_importance_catboost.csv",
        index=False,
    )

    for model_name, probabilities in [
        ("random_forest", rf_test_prob),
        ("catboost", cat_test_prob),
    ]:
        save_confusion_matrix_plot(
            y_test,
            probabilities,
            dirs["plots"]
            / f"confusion_matrix_{model_name}_test.png",
            f"Confusion matrix — {model_name}",
        )

        calibration = save_calibration_plot(
            y_test,
            probabilities,
            dirs["plots"]
            / f"calibration_{model_name}_test.png",
            f"Calibration — {model_name}",
        )
        calibration.to_csv(
            dirs["reports"]
            / f"calibration_{model_name}_test.csv",
            index=False,
        )

    metadata = {
        "dataset_rows": int(len(df)),
        "dataset_date_min": df["event_date"].min(),
        "dataset_date_max": df["event_date"].max(),
        "split_config": asdict(split_config),
        "split_sizes": {
            name: int(len(index))
            for name, index in splits.items()
        },
        "split_target_rates": {
            name: float(y.loc[index].mean())
            for name, index in splits.items()
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_forest_config": asdict(rf_config),
        "catboost_config": asdict(cat_config),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_model_by_validation_log_loss": selected_model,
        "target_definition": (
            "target=1: fighter_a wins; target=0: fighter_b wins"
        ),
        "selection_rule": (
            "The model is selected solely by validation log loss "
            "(2022–2023)."
        ),
    }
    save_json(
        metadata,
        dirs["root"] / "model_metadata.json",
    )

    print()
    print("Training complete.")
    print(f"Selected model: {selected_model}")
    print()
    print(metrics_df.to_string(index=False))
    print()
    print(f"Artifacts saved to: {dirs['root']}")

    return {
        "dataframe": df,
        "features": X,
        "splits": splits,
        "random_forest": random_forest,
        "catboost": catboost,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_model": selected_model,
        "artifact_directories": dirs,
        "metrics_dataframe": metrics_df,
    }
