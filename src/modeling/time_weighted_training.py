"""Train and compare models while varying the influence of older fights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.modeling.ufc_modeling import (
    CatBoostConfig,
    DEFAULT_ARTIFACTS_RELATIVE_DIR,
    RandomForestConfig,
    SplitConfig,
    build_feature_frame,
    build_random_forest_pipeline,
    catboost_feature_importance,
    load_dataset,
    metrics_from_probabilities,
    prediction_frame,
    random_forest_feature_importance,
    save_calibration_plot,
    save_confusion_matrix_plot,
    save_json,
    temporal_split,
)

TARGET_COLUMN = "target"


@dataclass(frozen=True)
class TimeWeightConfig:
    """Configuration for the temporal influence of each training fight."""

    mode: str = "none"
    cutoff_date: str = "2005-01-01"
    pre_cutoff_weight: float = 0.25
    half_life_years: float = 8.0
    minimum_weight: float = 0.10


def prepare_time_weighted_training_data(
    df: pd.DataFrame,
    train_index: pd.Index,
    config: TimeWeightConfig,
) -> tuple[pd.Index, pd.Series, dict[str, Any]]:
    """Filter training rows or calculate sample weights from fight dates."""

    allowed = {"none", "exclude_before", "step", "exponential"}
    if config.mode not in allowed:
        raise ValueError(
            f"Unknown mode: {config.mode}. Options: {sorted(allowed)}"
        )
    if not 0 < config.pre_cutoff_weight <= 1:
        raise ValueError("pre_cutoff_weight must be in (0, 1].")
    if config.half_life_years <= 0:
        raise ValueError("half_life_years must be greater than zero.")
    if not 0 < config.minimum_weight <= 1:
        raise ValueError("minimum_weight must be in (0, 1].")

    original_index = pd.Index(train_index)
    original_dates = pd.to_datetime(df.loc[original_index, "event_date"])
    cutoff = pd.Timestamp(config.cutoff_date)

    effective_index = original_index
    weights = pd.Series(1.0, index=effective_index, dtype="float64")

    if config.mode == "exclude_before":
        keep = original_dates.ge(cutoff).to_numpy()
        effective_index = original_index[keep]
        weights = pd.Series(1.0, index=effective_index, dtype="float64")

    elif config.mode == "step":
        weights.loc[original_dates.lt(cutoff)] = config.pre_cutoff_weight

    elif config.mode == "exponential":
        reference_date = original_dates.max()
        age_years = (
            (reference_date - original_dates).dt.days / 365.2425
        ).clip(lower=0.0)
        raw = np.power(0.5, age_years / config.half_life_years)
        weights = pd.Series(
            np.maximum(raw, config.minimum_weight),
            index=original_index,
            dtype="float64",
        )

    if len(effective_index) == 0:
        raise ValueError("The weighting strategy produces an empty training set.")

    y_effective = df.loc[effective_index, TARGET_COLUMN]
    if y_effective.nunique() < 2:
        raise ValueError("The effective training set does not contain both classes.")

    values = weights.to_numpy(dtype=float)
    effective_sample_size = float(
        values.sum() ** 2 / np.square(values).sum()
    )
    effective_dates = pd.to_datetime(df.loc[effective_index, "event_date"])

    summary: dict[str, Any] = {
        "config": asdict(config),
        "original_train_rows": int(len(original_index)),
        "effective_train_rows": int(len(effective_index)),
        "excluded_rows": int(len(original_index) - len(effective_index)),
        "effective_date_min": effective_dates.min(),
        "effective_date_max": effective_dates.max(),
        "weight_min": float(values.min()),
        "weight_mean": float(values.mean()),
        "weight_max": float(values.max()),
        "weight_sum": float(values.sum()),
        "effective_sample_size": effective_sample_size,
        "effective_target_rate": float(y_effective.mean()),
    }

    if config.mode == "step":
        summary["rows_before_cutoff"] = int(original_dates.lt(cutoff).sum())
        summary["rows_from_cutoff"] = int(original_dates.ge(cutoff).sum())
    if config.mode == "exponential":
        summary["reference_date"] = original_dates.max()

    return effective_index, weights, summary


def _artifact_directories(
    project_root: Path,
    experiment_name: str | None,
) -> dict[str, Path]:
    root = project_root / DEFAULT_ARTIFACTS_RELATIVE_DIR
    if experiment_name:
        safe = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_"
            for c in experiment_name
        )
        root = root / "experiments" / safe

    dirs = {
        "root": root,
        "models": root / "models",
        "reports": root / "reports",
        "plots": root / "plots",
        "predictions": root / "predictions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _train_catboost_weighted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
    config: CatBoostConfig,
):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install CatBoost: pip install catboost") from exc

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
        sample_weight=sample_weight.to_numpy(),
        cat_features=categorical_features,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    return model


def train_time_weighted_experiment(
    project_root: Path,
    time_weight_config: TimeWeightConfig,
    rf_config: RandomForestConfig = RandomForestConfig(),
    cat_config: CatBoostConfig = CatBoostConfig(),
    split_config: SplitConfig = SplitConfig(),
    evaluate_test: bool = False,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Train Random Forest and CatBoost with weights applied only to training data."""

    df = load_dataset(project_root)
    X, numeric_features, categorical_features = build_feature_frame(df)
    y = df[TARGET_COLUMN]
    splits = temporal_split(df, split_config)

    train_index, train_weights, weight_summary = (
        prepare_time_weighted_training_data(
            df, splits["train"], time_weight_config
        )
    )

    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    X_validation = X.loc[splits["validation"]]
    y_validation = y.loc[splits["validation"]]
    dirs = _artifact_directories(project_root, experiment_name)

    print("Temporal weighting:")
    print(
        f"  mode={time_weight_config.mode}; "
        f"original train={weight_summary['original_train_rows']:,}; "
        f"effective train={weight_summary['effective_train_rows']:,}; "
        f"mean weight={weight_summary['weight_mean']:.4f}; "
        f"effective sample size={weight_summary['effective_sample_size']:.1f}"
    )

    print("Training Random Forest...")
    random_forest = build_random_forest_pipeline(
        numeric_features, categorical_features, rf_config
    )
    random_forest.fit(
        X_train,
        y_train,
        model__sample_weight=train_weights.to_numpy(),
    )
    rf_val_prob = random_forest.predict_proba(X_validation)[:, 1]

    print("Training CatBoost...")
    catboost = _train_catboost_weighted(
        X_train,
        y_train,
        train_weights,
        X_validation,
        y_validation,
        categorical_features,
        cat_config,
    )
    cat_val_prob = catboost.predict_proba(X_validation)[:, 1]

    validation_metrics = {
        "random_forest": metrics_from_probabilities(
            y_validation, rf_val_prob
        ),
        "catboost": metrics_from_probabilities(
            y_validation, cat_val_prob
        ),
    }
    selected_model = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["log_loss"],
    )

    rows: list[dict[str, Any]] = []
    for model_name, metrics in validation_metrics.items():
        rows.append({
            "split": "validation",
            "model": model_name,
            **metrics,
        })

    validation_predictions = pd.concat([
        prediction_frame(
            df.loc[splits["validation"]], rf_val_prob, "random_forest"
        ),
        prediction_frame(
            df.loc[splits["validation"]], cat_val_prob, "catboost"
        ),
    ], ignore_index=True)
    validation_predictions.to_parquet(
        dirs["predictions"] / "validation_predictions.parquet",
        index=False,
    )

    test_metrics = None
    if evaluate_test:
        X_test = X.loc[splits["test"]]
        y_test = y.loc[splits["test"]]
        rf_test_prob = random_forest.predict_proba(X_test)[:, 1]
        cat_test_prob = catboost.predict_proba(X_test)[:, 1]
        test_metrics = {
            "random_forest": metrics_from_probabilities(
                y_test, rf_test_prob
            ),
            "catboost": metrics_from_probabilities(
                y_test, cat_test_prob
            ),
        }
        for model_name, metrics in test_metrics.items():
            rows.append({"split": "test", "model": model_name, **metrics})

        test_predictions = pd.concat([
            prediction_frame(
                df.loc[splits["test"]], rf_test_prob, "random_forest"
            ),
            prediction_frame(
                df.loc[splits["test"]], cat_test_prob, "catboost"
            ),
        ], ignore_index=True)
        test_predictions.to_parquet(
            dirs["predictions"] / "test_predictions.parquet",
            index=False,
        )

        for model_name, probabilities in [
            ("random_forest", rf_test_prob),
            ("catboost", cat_test_prob),
        ]:
            save_confusion_matrix_plot(
                y_test,
                probabilities,
                dirs["plots"] / f"confusion_matrix_{model_name}_test.png",
                f"Confusion matrix — {model_name}",
            )
            calibration = save_calibration_plot(
                y_test,
                probabilities,
                dirs["plots"] / f"calibration_{model_name}_test.png",
                f"Calibration — {model_name}",
            )
            calibration.to_csv(
                dirs["reports"] / f"calibration_{model_name}_test.csv",
                index=False,
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(dirs["reports"] / "model_metrics.csv", index=False)

    joblib.dump(
        random_forest,
        dirs["models"] / "random_forest_pipeline.joblib",
    )
    catboost.save_model(dirs["models"] / "catboost_model.cbm")

    random_forest_feature_importance(random_forest).to_csv(
        dirs["reports"] / "feature_importance_random_forest.csv",
        index=False,
    )
    catboost_feature_importance(catboost, list(X.columns)).to_csv(
        dirs["reports"] / "feature_importance_catboost.csv",
        index=False,
    )

    metadata = {
        "time_weight_config": asdict(time_weight_config),
        "time_weight_summary": weight_summary,
        "split_config": asdict(split_config),
        "evaluate_test": evaluate_test,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_model_by_validation_log_loss": selected_model,
        "random_forest_config": asdict(rf_config),
        "catboost_config": asdict(cat_config),
    }
    save_json(metadata, dirs["root"] / "model_metadata.json")

    print("\nTraining complete.")
    print(f"Selected model: {selected_model}")
    print(metrics_df.to_string(index=False))
    print(f"\nArtifacts: {dirs['root']}")

    return {
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_model": selected_model,
        "weighting_summary": weight_summary,
        "metrics_dataframe": metrics_df,
        "artifact_directories": dirs,
        "random_forest": random_forest,
        "catboost": catboost,
    }
