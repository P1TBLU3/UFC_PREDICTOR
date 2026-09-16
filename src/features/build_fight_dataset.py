"""Build a leakage-safe supervised dataset by combining pre-fight snapshots with stable fighter attributes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "ufcstats"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ufcstats"

FIGHTS_PATH = INTERIM_DIR / "fights_clean.parquet"
FIGHTERS_PATH = INTERIM_DIR / "fighters_clean.parquet"
SNAPSHOTS_PATH = PROCESSED_DIR / "fighter_snapshots.parquet"
ELO_SNAPSHOTS_PATH = PROCESSED_DIR / "fighter_elo_snapshots.parquet"
ELO_REPORT_PATH = PROCESSED_DIR / "fighter_elo_snapshots_report.json"

OUTPUT_PATH = PROCESSED_DIR / "fight_dataset.parquet"
REPORT_PATH = PROCESSED_DIR / "fight_dataset_report.json"


# Historical values available before each fight.
SNAPSHOT_FEATURE_COLUMNS = [
    "fights_before",
    "wins_before",
    "losses_before",
    "draws_before",
    "no_contests_before",
    "win_rate_before",
    "current_win_streak_before",
    "current_loss_streak_before",
    "days_since_last_fight",
    "total_fight_time_seconds_before",
    "fights_with_stats_before",
    "stats_fight_time_seconds_before",
    "stats_coverage_before",
    "fights_with_control_stats_before",
    "control_stats_coverage_before",
    "sig_str_landed_before",
    "sig_str_attempted_before",
    "sig_str_absorbed_before",
    "sig_str_absorbed_attempted_before",
    "sig_str_landed_per_min_before",
    "sig_str_absorbed_per_min_before",
    "sig_str_accuracy_before",
    "sig_str_defense_before",
    "takedowns_landed_before",
    "takedowns_attempted_before",
    "takedowns_conceded_before",
    "takedowns_faced_before",
    "takedown_accuracy_before",
    "takedown_defense_before",
    "control_seconds_before",
    "knockdowns_before",
    "knockdowns_absorbed_before",
    "submission_attempts_before",
    "reversals_before",
]

# Stable physical attributes do not require pre-fight/current variants.
PHYSICAL_NUMERIC_FEATURES = [
    "height_cm",
    "reach_cm",
    "age_years",
]

PHYSICAL_CATEGORICAL_FEATURES = [
    "stance",
]

ELO_FEATURE_COLUMNS = [
    "is_debutant",
    "elo_confidence",
    "elo_striking_confidence",
    "elo_grappling_offense_confidence",
    "elo_grappling_defense_confidence",
    "elo_days_inactive_before",
    "elo_general_raw_before",
    "elo_striking_raw_before",
    "elo_grappling_offense_raw_before",
    "elo_grappling_defense_raw_before",
]

ELO_CONFIDENCE_PRIOR_FIGHTS = 3.0
EXPECTED_ELO_CONFIG_VERSION = "elo_v2_final_raw"

ELO_DIFF_FEATURE_BASES = [
    column for column in ELO_FEATURE_COLUMNS if column != "is_debutant"
]

DIFF_FEATURE_BASES = [
    *SNAPSHOT_FEATURE_COLUMNS,
    *PHYSICAL_NUMERIC_FEATURES,
    *ELO_DIFF_FEATURE_BASES,
]

METADATA_COLUMNS = [
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

CATEGORICAL_MODEL_FEATURES = [
    "weight_class",
    "fighter_a_stance",
    "fighter_b_stance",
    "stance_matchup",
    "same_stance",
]


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Check that a DataFrame contains the required columns."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {dataframe_name}: "
            f"{sorted(missing_columns)}"
        )


def validate_elo_report_version() -> None:
    """Fail fast when Elo snapshots were generated with an obsolete configuration."""

    if not ELO_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Elo report: {ELO_REPORT_PATH}. "
            "Run build_elo_snapshots.py before build_fight_dataset.py."
        )

    report = json.loads(ELO_REPORT_PATH.read_text(encoding="utf-8"))
    version = report.get("config_version")
    if version != EXPECTED_ELO_CONFIG_VERSION:
        raise ValueError(
            "fighter_elo_snapshots.parquet was not generated with the final "
            f"production Elo configuration. Expected {EXPECTED_ELO_CONFIG_VERSION!r}, "
            f"found {version!r}. Re-run build_elo_snapshots.py."
        )


def normalize_result(value: object) -> str | None:
    """Normalize W, L, D, and NC; return None for missing values."""

    if pd.isna(value):
        return None

    normalized = str(value).strip().upper()

    aliases = {
        "W": "W",
        "L": "L",
        "D": "D",
        "DRAW": "D",
        "NC": "NC",
        "N/C": "NC",
        "NO CONTEST": "NC",
    }

    return aliases.get(normalized, normalized)


def calculate_age_years(
    event_date: pd.Series,
    date_of_birth: pd.Series,
) -> pd.Series:
    """
    Calculate age on the fight date.

    Use 365.2425 days as the mean Gregorian year length.
    """

    age_days = (
        event_date.dt.normalize()
        - date_of_birth.dt.normalize()
    ).dt.days

    age_years = age_days / 365.2425

    return age_years.astype("Float64")


def prefix_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Select columns and add a prefix to their names."""

    renamed = {
        column: f"{prefix}{column}"
        for column in columns
    }

    return dataframe[columns].rename(columns=renamed)


def prepare_decisive_fights(
    fights: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Keep decisive fights and assign A/B by sorted fighter_id.

    This prevents UFCStats' winner-first ordering from leaking the outcome.
    """

    result = fights.copy()

    result["fighter_1_result_normalized"] = (
        result["fighter_1_result"].map(normalize_result)
    )
    result["fighter_2_result_normalized"] = (
        result["fighter_2_result"].map(normalize_result)
    )

    valid_pairs = (
        (
            result["fighter_1_result_normalized"].eq("W")
            & result["fighter_2_result_normalized"].eq("L")
        )
        |
        (
            result["fighter_1_result_normalized"].eq("L")
            & result["fighter_2_result_normalized"].eq("W")
        )
    )

    decisive = result.loc[valid_pairs].copy()

    # Validate the winner before applying the deterministic A/B ordering.
    original_expected_winner = decisive["fighter_1_id"].where(
        decisive["fighter_1_result_normalized"].eq("W"),
        decisive["fighter_2_id"],
    )

    winner_mismatch = (
        decisive["winner_id"].astype("string")
        != original_expected_winner.astype("string")
    )

    if winner_mismatch.any():
        examples = decisive.loc[
            winner_mismatch,
            [
                "fight_id",
                "fighter_1_id",
                "fighter_2_id",
                "winner_id",
                "fighter_1_result",
                "fighter_2_result",
            ],
        ].head(10)

        raise ValueError(
            "winner_id does not match the recorded outcomes for decisive "
            "fights:\n"
            f"{examples.to_string(index=False)}"
        )

    missing_ids = decisive[
        ["fighter_1_id", "fighter_2_id"]
    ].isna().any(axis=1)

    if missing_ids.any():
        examples = decisive.loc[
            missing_ids,
            [
                "fight_id",
                "fighter_1_id",
                "fighter_2_id",
            ],
        ].head(10)

        raise ValueError(
            "Decisive fights contain missing fighter_id values:\n"
            f"{examples.to_string(index=False)}"
        )

    same_fighter = (
        decisive["fighter_1_id"].astype("string")
        == decisive["fighter_2_id"].astype("string")
    )

    if same_fighter.any():
        raise ValueError(
            "Decisive fights contain the same fighter "
            "on both sides."
        )

    fighter_1_is_a = (
        decisive["fighter_1_id"].astype("string")
        < decisive["fighter_2_id"].astype("string")
    )

    decisive["fighter_a_id"] = decisive[
        "fighter_1_id"
    ].where(
        fighter_1_is_a,
        decisive["fighter_2_id"],
    )

    decisive["fighter_a_name"] = decisive[
        "fighter_1_name"
    ].where(
        fighter_1_is_a,
        decisive["fighter_2_name"],
    )

    decisive["fighter_b_id"] = decisive[
        "fighter_2_id"
    ].where(
        fighter_1_is_a,
        decisive["fighter_1_id"],
    )

    decisive["fighter_b_name"] = decisive[
        "fighter_2_name"
    ].where(
        fighter_1_is_a,
        decisive["fighter_1_name"],
    )

    decisive["target"] = (
        decisive["fighter_a_id"].astype("string")
        == decisive["winner_id"].astype("string")
    ).astype("Int64")

    original_fighter_1_win_rate = float(
        decisive["fighter_1_result_normalized"]
        .eq("W")
        .mean()
    )

    exclusion_report: dict[str, object] = {
        "total_fights": int(len(result)),
        "decisive_fights": int(valid_pairs.sum()),
        "excluded_non_decisive_fights": int((~valid_pairs).sum()),
        "excluded_draws": int(
            (
                result["fighter_1_result_normalized"].eq("D")
                | result["fighter_2_result_normalized"].eq("D")
            ).sum()
        ),
        "excluded_no_contests": int(
            (
                result["fighter_1_result_normalized"].eq("NC")
                | result["fighter_2_result_normalized"].eq("NC")
            ).sum()
        ),
        "excluded_other_or_missing_results": int(
            (
                ~valid_pairs
                & ~(
                    result["fighter_1_result_normalized"].isin(
                        ["D", "NC"]
                    )
                    | result["fighter_2_result_normalized"].isin(
                        ["D", "NC"]
                    )
                )
            ).sum()
        ),
        "original_fighter_1_win_rate": (
            original_fighter_1_win_rate
        ),
        "side_assignment_rule": (
            "fighter_a=min(fighter_1_id,fighter_2_id); "
            "fighter_b=max(fighter_1_id,fighter_2_id)"
        ),
        "fighter_a_from_original_fighter_1": int(
            fighter_1_is_a.sum()
        ),
        "fighter_a_from_original_fighter_2": int(
            (~fighter_1_is_a).sum()
        ),
    }

    return decisive, exclusion_report

def merge_snapshot_side(
    fights: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """Join the correct snapshot for fighter_a or fighter_b."""

    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'.")

    opposite_side = "b" if side == "a" else "a"
    prefix = f"fighter_{side}_"

    snapshot_columns = [
        "fight_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        *SNAPSHOT_FEATURE_COLUMNS,
    ]

    side_snapshots = snapshots[snapshot_columns].copy()

    side_snapshots = side_snapshots.rename(
        columns={
            "fighter_id": f"{prefix}snapshot_fighter_id",
            "opponent_id": f"{prefix}snapshot_opponent_id",
            "event_date": f"{prefix}snapshot_event_date",
            **{
                column: f"{prefix}{column}"
                for column in SNAPSHOT_FEATURE_COLUMNS
            },
        }
    )

    merged = fights.merge(
        side_snapshots,
        left_on=[
            "fight_id",
            f"fighter_{side}_id",
        ],
        right_on=[
            "fight_id",
            f"{prefix}snapshot_fighter_id",
        ],
        how="left",
        validate="one_to_one",
        indicator=f"_merge_snapshot_{side}",
    )

    unmatched = (
        merged[f"_merge_snapshot_{side}"] != "both"
    )

    if unmatched.any():
        examples = merged.loc[
            unmatched,
            [
                "fight_id",
                f"fighter_{side}_id",
                f"fighter_{side}_name",
            ],
        ].head(10)

        raise ValueError(
            f"Missing snapshots for fighter_{side}:\n"
            f"{examples.to_string(index=False)}"
        )

    fighter_mismatch = (
        merged[f"{prefix}snapshot_fighter_id"].astype("string")
        != merged[f"fighter_{side}_id"].astype("string")
    )

    opponent_mismatch = (
        merged[f"{prefix}snapshot_opponent_id"].astype("string")
        != merged[f"fighter_{opposite_side}_id"].astype("string")
    )

    date_mismatch = (
        pd.to_datetime(
            merged[f"{prefix}snapshot_event_date"]
        )
        != pd.to_datetime(merged["event_date"])
    )

    if fighter_mismatch.any():
        raise ValueError(
            f"The snapshot for fighter_{side} does not match "
            "the expected fighter."
        )

    if opponent_mismatch.any():
        examples = merged.loc[
            opponent_mismatch,
            [
                "fight_id",
                f"fighter_{side}_id",
                f"fighter_{opposite_side}_id",
                f"{prefix}snapshot_opponent_id",
            ],
        ].head(10)

        raise ValueError(
            f"The snapshot opponent for fighter_{side} does not match:\n"
            f"{examples.to_string(index=False)}"
        )

    if date_mismatch.any():
        raise ValueError(
            f"The snapshot date for fighter_{side} does not match the fight date."
        )

    columns_to_drop = [
        f"{prefix}snapshot_fighter_id",
        f"{prefix}snapshot_opponent_id",
        f"{prefix}snapshot_event_date",
        f"_merge_snapshot_{side}",
    ]

    return merged.drop(columns=columns_to_drop)

def merge_elo_side(
    dataset: pd.DataFrame,
    elo_snapshots: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """Join the leakage-safe pre-fight Elo snapshot for one side."""

    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'.")

    opposite_side = "b" if side == "a" else "a"
    prefix = f"fighter_{side}_"

    columns = [
        "fight_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        *ELO_FEATURE_COLUMNS,
    ]
    side_elo = elo_snapshots[columns].copy()
    side_elo = side_elo.rename(
        columns={
            "fighter_id": f"{prefix}elo_snapshot_fighter_id",
            "opponent_id": f"{prefix}elo_snapshot_opponent_id",
            "event_date": f"{prefix}elo_snapshot_event_date",
            **{column: f"{prefix}{column}" for column in ELO_FEATURE_COLUMNS},
        }
    )

    merged = dataset.merge(
        side_elo,
        left_on=["fight_id", f"fighter_{side}_id"],
        right_on=["fight_id", f"{prefix}elo_snapshot_fighter_id"],
        how="left",
        validate="one_to_one",
        indicator=f"_merge_elo_{side}",
    )

    unmatched = merged[f"_merge_elo_{side}"] != "both"
    if unmatched.any():
        examples = merged.loc[
            unmatched,
            ["fight_id", f"fighter_{side}_id", f"fighter_{side}_name"],
        ].head(10)
        raise ValueError(
            f"Missing Elo snapshots for fighter_{side}:\n"
            f"{examples.to_string(index=False)}"
        )

    opponent_mismatch = (
        merged[f"{prefix}elo_snapshot_opponent_id"].astype("string")
        != merged[f"fighter_{opposite_side}_id"].astype("string")
    )
    if opponent_mismatch.any():
        raise ValueError(
            f"The Elo snapshot opponent for fighter_{side} does not match."
        )

    date_mismatch = (
        pd.to_datetime(merged[f"{prefix}elo_snapshot_event_date"])
        != pd.to_datetime(merged["event_date"])
    )
    if date_mismatch.any():
        raise ValueError(
            f"The Elo snapshot date for fighter_{side} does not match the fight date."
        )

    return merged.drop(
        columns=[
            f"{prefix}elo_snapshot_fighter_id",
            f"{prefix}elo_snapshot_opponent_id",
            f"{prefix}elo_snapshot_event_date",
            f"_merge_elo_{side}",
        ]
    )


def merge_physical_side(
    dataset: pd.DataFrame,
    fighters: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """Add height, reach, stance, and date of birth."""

    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'.")

    prefix = f"fighter_{side}_"

    profile_columns = [
        "fighter_id",
        "height_cm",
        "reach_cm",
        "stance",
        "date_of_birth",
    ]

    profiles = fighters[profile_columns].copy()

    profiles = profiles.rename(
        columns={
            "fighter_id": f"{prefix}profile_fighter_id",
            "height_cm": f"{prefix}height_cm",
            "reach_cm": f"{prefix}reach_cm",
            "stance": f"{prefix}stance",
            "date_of_birth": f"{prefix}date_of_birth",
        }
    )

    merged = dataset.merge(
        profiles,
        left_on=f"fighter_{side}_id",
        right_on=f"{prefix}profile_fighter_id",
        how="left",
        validate="many_to_one",
        indicator=f"_merge_profile_{side}",
    )

    missing_profile = (
        merged[f"_merge_profile_{side}"] != "both"
    )

    if missing_profile.any():
        examples = merged.loc[
            missing_profile,
            [
                "fight_id",
                f"fighter_{side}_id",
                f"fighter_{side}_name",
            ],
        ].drop_duplicates().head(10)

        raise ValueError(
            f"Missing profiles for fighter_{side}:\n"
            f"{examples.to_string(index=False)}"
        )

    merged[f"{prefix}age_years"] = calculate_age_years(
        pd.to_datetime(merged["event_date"]),
        pd.to_datetime(merged[f"{prefix}date_of_birth"]),
    )

    invalid_age = (
        merged[f"{prefix}age_years"].notna()
        & ~merged[f"{prefix}age_years"].between(
            10,
            80,
            inclusive="both",
        )
    )

    if invalid_age.any():
        examples = merged.loc[
            invalid_age,
            [
                "fight_id",
                f"fighter_{side}_name",
                "event_date",
                f"{prefix}date_of_birth",
                f"{prefix}age_years",
            ],
        ].head(10)

        raise ValueError(
            f"Invalid ages were calculated for fighter_{side}:\n"
            f"{examples.to_string(index=False)}"
        )

    return merged.drop(
        columns=[
            f"{prefix}profile_fighter_id",
            f"{prefix}date_of_birth",
            f"_merge_profile_{side}",
        ]
    )

def add_difference_features(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Add fighter_a minus fighter_b differences."""

    result = dataset.copy()

    for base_name in DIFF_FEATURE_BASES:
        fighter_a_column = f"fighter_a_{base_name}"
        fighter_b_column = f"fighter_b_{base_name}"
        difference_column = f"diff_{base_name}"

        if fighter_a_column not in result.columns:
            raise ValueError(
                f"Missing column: {fighter_a_column}."
            )

        if fighter_b_column not in result.columns:
            raise ValueError(
                f"Missing column: {fighter_b_column}."
            )

        result[difference_column] = (
            result[fighter_a_column]
            - result[fighter_b_column]
        )

    return result


def add_stance_features(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Create categorical stance-matchup features."""

    result = dataset.copy()

    stance_a = (
        result["fighter_a_stance"]
        .astype("string")
        .fillna("Unknown")
    )
    stance_b = (
        result["fighter_b_stance"]
        .astype("string")
        .fillna("Unknown")
    )

    result["fighter_a_stance"] = stance_a
    result["fighter_b_stance"] = stance_b

    result["stance_matchup"] = (
        stance_a + "_vs_" + stance_b
    ).astype("string")

    result["same_stance"] = (
        stance_a.eq(stance_b)
        & stance_a.ne("Unknown")
    ).astype("boolean")

    return result


def validate_output(
    dataset: pd.DataFrame,
    decisive_fights: pd.DataFrame,
) -> None:
    """Validate the output structure, target, and basic leakage constraints."""

    if len(dataset) != len(decisive_fights):
        raise ValueError(
            f"Rows generated: {len(dataset)}; "
            f"decisive fights: {len(decisive_fights)}."
        )

    if dataset["fight_id"].duplicated().any():
        duplicated = dataset.loc[
            dataset["fight_id"].duplicated(keep=False),
            ["fight_id"],
        ].head(10)

        raise ValueError(
            "The final dataset contains duplicate fights:\n"
            f"{duplicated.to_string(index=False)}"
        )

    if not dataset["target"].dropna().isin([0, 1]).all():
        raise ValueError(
            "The target column contains values other than 0 and 1."
        )

    if dataset["target"].isna().any():
        raise ValueError("The target column contains missing values.")

    same_fighter = (
        dataset["fighter_a_id"]
        == dataset["fighter_b_id"]
    )

    if same_fighter.any():
        raise ValueError(
            "At least one fight contains the same fighter on both sides."
        )

    invalid_ab_order = (
        dataset["fighter_a_id"].astype("string")
        > dataset["fighter_b_id"].astype("string")
    )
    if invalid_ab_order.any():
        raise ValueError(
            "A/B assignment is not deterministic: fighter_a_id must be <= fighter_b_id."
        )

    sorted_keys = (
        dataset[["event_date", "event_id", "fight_order"]]
        .sort_values(
            ["event_date", "event_id", "fight_order"],
            ascending=[True, True, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    current_keys = dataset[["event_date", "event_id", "fight_order"]].reset_index(drop=True)
    if not current_keys.equals(sorted_keys):
        raise ValueError("The final dataset is not in stable chronological order.")

    forbidden_suffix_columns = [
        column
        for column in dataset.columns
        if column.endswith("_current")
    ]

    if forbidden_suffix_columns:
        raise ValueError(
            "The dataset contains prohibited current-state metrics: "
            f"{forbidden_suffix_columns}"
        )

    forbidden_post_fight_columns = {
        "winner_id",
        "fighter_1_result",
        "fighter_2_result",
        "method",
        "method_details",
        "finish_round",
        "finish_time",
    }

    leaked_columns = (
        forbidden_post_fight_columns
        & set(dataset.columns)
    )

    if leaked_columns:
        raise ValueError(
            "The dataset contains post-fight columns: "
            f"{sorted(leaked_columns)}"
        )

    forbidden_legacy_elo_columns = [
        column
        for column in dataset.columns
        if "elo_" in column
        and ("effective" in column or "decay_fraction" in column)
    ]
    if forbidden_legacy_elo_columns:
        raise ValueError(
            "Legacy Effective/decay Elo columns are present in the final dataset: "
            f"{forbidden_legacy_elo_columns}"
        )

    for side in ("a", "b"):
        debutant_column = f"fighter_{side}_is_debutant"
        if not dataset[debutant_column].dropna().isin([0, 1]).all():
            raise ValueError(f"{debutant_column} contains values other than 0/1.")

        fights_before = pd.to_numeric(
            dataset[f"fighter_{side}_fights_before"], errors="coerce"
        )
        debutant = pd.to_numeric(dataset[debutant_column], errors="coerce")
        expected_debutant = fights_before.eq(0).astype(int)
        if not debutant.eq(expected_debutant).all():
            raise ValueError(
                f"{debutant_column} is inconsistent with fights_before."
            )

        record_sum = (
            pd.to_numeric(dataset[f"fighter_{side}_wins_before"], errors="coerce")
            + pd.to_numeric(dataset[f"fighter_{side}_losses_before"], errors="coerce")
            + pd.to_numeric(dataset[f"fighter_{side}_draws_before"], errors="coerce")
            + pd.to_numeric(dataset[f"fighter_{side}_no_contests_before"], errors="coerce")
        )
        if not record_sum.eq(fights_before).all():
            raise ValueError(
                f"fighter_{side} historical record does not sum to fights_before."
            )

        expected_confidence = fights_before / (
            fights_before + ELO_CONFIDENCE_PRIOR_FIGHTS
        )
        actual_confidence = pd.to_numeric(
            dataset[f"fighter_{side}_elo_confidence"], errors="coerce"
        )
        if not np.allclose(
            actual_confidence.to_numpy(dtype=float),
            expected_confidence.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"fighter_{side}_elo_confidence is inconsistent with "
                "fights_before and the production confidence formula."
            )

        stats_days = pd.to_numeric(
            dataset[f"fighter_{side}_days_since_last_fight"], errors="coerce"
        )
        elo_days = pd.to_numeric(
            dataset[f"fighter_{side}_elo_days_inactive_before"], errors="coerce"
        )
        same_days = stats_days.eq(elo_days) | (stats_days.isna() & elo_days.isna())
        if not same_days.all():
            raise ValueError(
                f"fighter_{side} inactivity differs between statistical and Elo snapshots."
            )

        for rating_name in [
            "elo_general_raw_before",
            "elo_striking_raw_before",
            "elo_grappling_offense_raw_before",
            "elo_grappling_defense_raw_before",
        ]:
            values = pd.to_numeric(
                dataset[f"fighter_{side}_{rating_name}"], errors="coerce"
            )
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"fighter_{side}_{rating_name} contains missing/non-finite values."
                )

    elo_bounded_columns = [
        column
        for column in dataset.columns
        if (
            "elo_confidence" in column
            or "elo_striking_confidence" in column
            or "elo_grappling_offense_confidence" in column
            or "elo_grappling_defense_confidence" in column
        )
        and not column.startswith("diff_")
    ]
    for column in elo_bounded_columns:
        values = dataset[column].dropna()
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"Values outside [0,1] were found in {column}.")

    bounded_columns = [
        column
        for column in dataset.columns
        if (
            column.endswith("win_rate_before")
            or column.endswith("stats_coverage_before")
            or column.endswith("control_stats_coverage_before")
            or column.endswith("sig_str_accuracy_before")
            or column.endswith("sig_str_defense_before")
            or column.endswith("takedown_accuracy_before")
            or column.endswith("takedown_defense_before")
        )
        and not column.startswith("diff_")
    ]

    for column in bounded_columns:
        values = dataset[column].dropna()

        invalid = ~values.between(
            0.0,
            1.0,
            inclusive="both",
        )

        if invalid.any():
            raise ValueError(
                f"Values outside [0, 1] were found in {column}."
            )

    # Sample difference features to detect sign errors.
    for base_name in DIFF_FEATURE_BASES:
        difference_column = f"diff_{base_name}"
        expected = (
            dataset[f"fighter_a_{base_name}"]
            - dataset[f"fighter_b_{base_name}"]
        )

        mismatch = ~(
            dataset[difference_column].eq(expected)
            | (
                dataset[difference_column].isna()
                & expected.isna()
            )
        )

        if mismatch.any():
            raise ValueError(
                f"Difference feature {difference_column} "
                "does not equal fighter_a - fighter_b."
            )


def set_output_dtypes(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Assign stable pandas dtypes before writing Parquet."""

    result = dataset.copy()

    string_columns = [
        "fight_id",
        "event_id",
        "event_name",
        "weight_class",
        "fighter_a_id",
        "fighter_a_name",
        "fighter_b_id",
        "fighter_b_name",
        "fighter_a_stance",
        "fighter_b_stance",
        "stance_matchup",
    ]

    integer_columns = [
        "fight_order",
        "target",
    ]

    for side in ("fighter_a_", "fighter_b_"):
        integer_columns.extend(
            [
                f"{side}{column}"
                for column in SNAPSHOT_FEATURE_COLUMNS
                if column not in {
                    "win_rate_before",
                    "stats_coverage_before",
                    "control_stats_coverage_before",
                    "sig_str_landed_per_min_before",
                    "sig_str_absorbed_per_min_before",
                    "sig_str_accuracy_before",
                    "sig_str_defense_before",
                    "takedown_accuracy_before",
                    "takedown_defense_before",
                }
            ]
        )

    float_columns = [
        column
        for column in result.columns
        if (
            column.startswith("diff_")
            or column
            in {
                f"fighter_{side}_{feature}"
                for side in ("a", "b")
                for feature in [
                    "win_rate_before",
                    "stats_coverage_before",
                    "control_stats_coverage_before",
                    "sig_str_landed_per_min_before",
                    "sig_str_absorbed_per_min_before",
                    "sig_str_accuracy_before",
                    "sig_str_defense_before",
                    "takedown_accuracy_before",
                    "takedown_defense_before",
                    "height_cm",
                    "reach_cm",
                    "age_years",
                ]
            }
        )
    ]

    for column in string_columns:
        if column in result.columns:
            result[column] = result[column].astype("string")

    for column in integer_columns:
        if column in result.columns:
            result[column] = result[column].astype("Int64")

    for column in float_columns:
        if column in result.columns:
            result[column] = result[column].astype("Float64")

    result["same_stance"] = result["same_stance"].astype(
        "boolean"
    )

    result["event_date"] = pd.to_datetime(
        result["event_date"]
    )

    return result


def build_feature_manifest(
    dataset: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Separate metadata, model features, and the target.

    The manifest prevents accidental training on names, identifiers,
    or post-fight columns.
    """

    prefixed_numeric_features = [
        column
        for column in dataset.columns
        if (
            column.startswith("fighter_a_")
            or column.startswith("fighter_b_")
            or column.startswith("diff_")
        )
        and column not in {
            "fighter_a_id",
            "fighter_a_name",
            "fighter_b_id",
            "fighter_b_name",
            "fighter_a_stance",
            "fighter_b_stance",
        }
    ]

    numeric_model_features = [
        column
        for column in prefixed_numeric_features
        if pd.api.types.is_numeric_dtype(dataset[column].dtype)
    ]

    categorical_model_features = [
        column
        for column in CATEGORICAL_MODEL_FEATURES
        if column in dataset.columns
    ]

    return {
        "metadata_columns": METADATA_COLUMNS,
        "numeric_model_features": numeric_model_features,
        "categorical_model_features": categorical_model_features,
        "target_column": ["target"],
    }


def build_fight_dataset() -> pd.DataFrame:
    """Build one supervised row per decisive fight."""

    print(f"Reading fights: {FIGHTS_PATH}")
    fights = pd.read_parquet(FIGHTS_PATH)

    print(f"Reading snapshots: {SNAPSHOTS_PATH}")
    snapshots = pd.read_parquet(SNAPSHOTS_PATH)

    print(f"Reading Elo snapshots: {ELO_SNAPSHOTS_PATH}")
    elo_snapshots = pd.read_parquet(ELO_SNAPSHOTS_PATH)

    validate_elo_report_version()

    print(f"Reading fighter profiles: {FIGHTERS_PATH}")
    fighters = pd.read_parquet(FIGHTERS_PATH)

    require_columns(
        fights,
        {
            "fight_id",
            "event_id",
            "event_name",
            "event_date",
            "fight_order",
            "weight_class",
            "fighter_1_id",
            "fighter_1_name",
            "fighter_1_result",
            "fighter_2_id",
            "fighter_2_name",
            "fighter_2_result",
            "winner_id",
        },
        "fights_clean.parquet",
    )

    require_columns(
        snapshots,
        {
            "fight_id",
            "fighter_id",
            "opponent_id",
            "event_date",
            *SNAPSHOT_FEATURE_COLUMNS,
        },
        "fighter_snapshots.parquet",
    )

    require_columns(
        elo_snapshots,
        {
            "fight_id",
            "fighter_id",
            "opponent_id",
            "event_date",
            *ELO_FEATURE_COLUMNS,
        },
        "fighter_elo_snapshots.parquet",
    )

    require_columns(
        fighters,
        {
            "fighter_id",
            "height_cm",
            "reach_cm",
            "stance",
            "date_of_birth",
        },
        "fighters_clean.parquet",
    )

    duplicate_profiles = fighters["fighter_id"].duplicated(
        keep=False
    )

    if duplicate_profiles.any():
        examples = fighters.loc[
            duplicate_profiles,
            ["fighter_id", "fighter_name"],
        ].head(10)

        raise ValueError(
            "fighters_clean.parquet contains duplicate fighter_id values:\n"
            f"{examples.to_string(index=False)}"
        )

    duplicate_snapshots = snapshots.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicate_snapshots.any():
        examples = snapshots.loc[
            duplicate_snapshots,
            ["fight_id", "fighter_id"],
        ].head(10)

        raise ValueError(
            "fighter_snapshots.parquet contains duplicate keys:\n"
            f"{examples.to_string(index=False)}"
        )

    duplicate_elo_snapshots = elo_snapshots.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicate_elo_snapshots.any():
        examples = elo_snapshots.loc[
            duplicate_elo_snapshots,
            ["fight_id", "fighter_id"],
        ].head(10)
        raise ValueError(
            "fighter_elo_snapshots.parquet contains duplicate keys:\n"
            f"{examples.to_string(index=False)}"
        )

    decisive_fights, exclusion_report = (
        prepare_decisive_fights(fights)
    )

    base_columns = [
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
        "target",
    ]

    dataset = decisive_fights[base_columns].copy()

    dataset = merge_snapshot_side(
        dataset,
        snapshots,
        side="a",
    )

    dataset = merge_snapshot_side(
        dataset,
        snapshots,
        side="b",
    )

    dataset = merge_elo_side(
        dataset,
        elo_snapshots,
        side="a",
    )

    dataset = merge_elo_side(
        dataset,
        elo_snapshots,
        side="b",
    )

    dataset = merge_physical_side(
        dataset,
        fighters,
        side="a",
    )

    dataset = merge_physical_side(
        dataset,
        fighters,
        side="b",
    )

    dataset = add_stance_features(dataset)
    dataset = add_difference_features(dataset)

    # Use stable chronological ordering for temporal splits.
    dataset = dataset.sort_values(
        by=[
            "event_date",
            "event_id",
            "fight_order",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="stable",
    ).reset_index(drop=True)

    dataset = set_output_dtypes(dataset)

    validate_output(
        dataset=dataset,
        decisive_fights=decisive_fights,
    )

    manifest = build_feature_manifest(dataset)

    ordered_columns = [
        *METADATA_COLUMNS,
        *[
            f"fighter_a_{column}"
            for column in SNAPSHOT_FEATURE_COLUMNS
        ],
        *[
            f"fighter_b_{column}"
            for column in SNAPSHOT_FEATURE_COLUMNS
        ],
        *[
            f"fighter_a_{column}"
            for column in ELO_FEATURE_COLUMNS
        ],
        *[
            f"fighter_b_{column}"
            for column in ELO_FEATURE_COLUMNS
        ],
        "fighter_a_height_cm",
        "fighter_a_reach_cm",
        "fighter_a_age_years",
        "fighter_a_stance",
        "fighter_b_height_cm",
        "fighter_b_reach_cm",
        "fighter_b_age_years",
        "fighter_b_stance",
        "stance_matchup",
        "same_stance",
        *[
            f"diff_{column}"
            for column in DIFF_FEATURE_BASES
        ],
        "target",
    ]

    missing_ordered_columns = (
        set(ordered_columns) - set(dataset.columns)
    )

    if missing_ordered_columns:
        raise ValueError(
            "Columns required for output ordering are missing: "
            f"{sorted(missing_ordered_columns)}"
        )

    dataset = dataset[ordered_columns]

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    target_rate = float(dataset["target"].mean())

    missingness = (
        dataset.isna()
        .mean()
        .sort_values(ascending=False)
    )

    report = {
        **exclusion_report,
        "rows_written": int(len(dataset)),
        "columns_written": int(len(dataset.columns)),
        "fighter_a_win_rate": target_rate,
        "fighter_b_win_rate": 1.0 - target_rate,
        "date_min": dataset["event_date"].min().date().isoformat(),
        "date_max": dataset["event_date"].max().date().isoformat(),
        "weight_classes": int(
            dataset["weight_class"].nunique(dropna=True)
        ),
        "feature_manifest": manifest,
        "highest_missingness": {
            str(column): float(rate)
            for column, rate in missingness.head(20).items()
        },
        "notes": [
            "fighter_a is the lexicographically smaller fighter_id.",
            "fighter_b is the lexicographically larger fighter_id.",
            "The A/B assignment is deterministic and independent of the outcome.",
            "target=1 indicates a fighter_a win.",
            "Draws and no contests are excluded from the binary target.",
            "Draws and no contests still contribute to historical snapshots.",
            "Columns ending in _current are excluded.",
            "Method, final round, and final time are excluded from the features.",
            "Differences are calculated as fighter_a - fighter_b.",
            "event_date must be used for chronological train/validation/test splits.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Fight-level dataset generated successfully.")
    print(f"Total fights: {len(fights):,}")
    print(
        "Decisive fights included: "
        f"{len(dataset):,}"
    )
    print(
        "Non-decisive fights excluded: "
        f"{exclusion_report['excluded_non_decisive_fights']:,}"
    )
    print(
        "Original fighter_1 win rate: "
        f"{exclusion_report['original_fighter_1_win_rate']:.2%}"
    )
    print(
        "fighter_a assigned from fighter_1: "
        f"{exclusion_report['fighter_a_from_original_fighter_1']:,}"
    )
    print(
        "fighter_a assigned from fighter_2: "
        f"{exclusion_report['fighter_a_from_original_fighter_2']:,}"
    )
    print(
        "fighter_a win rate: "
        f"{target_rate:.2%}"
    )

    if target_rate < 0.45 or target_rate > 0.55:
        print(
            "WARNING: fighter_a/fighter_b position is strongly associated "
            "with the outcome. Review the assignment before training."
        )

    print(f"Columns generated: {len(dataset.columns):,}")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")

    return dataset


def main() -> None:
    build_fight_dataset()


if __name__ == "__main__":
    main()
