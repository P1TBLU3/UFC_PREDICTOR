"""Clean, validate, and convert raw UFCStats CSV files to Parquet.

Strict relational checks prevent incomplete historical fights from entering the
modeling datasets while allowing genuinely upcoming fights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = "1.2.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ufcstats"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "ufcstats"

INPUT_FILES = {
    "events": "events.csv",
    "fights": "fights.csv",
    "fight_details": "fight_details.csv",
    "fight_totals": "fight_totals.csv",
    "round_stats": "round_stats.csv",
    "fighters": "fighters.csv",
}

OUTPUT_FILES = {
    "events": "events_clean.parquet",
    "fights": "fights_clean.parquet",
    "fight_details": "fight_details_clean.parquet",
    "fight_totals": "fight_totals_clean.parquet",
    "round_stats": "round_stats_clean.parquet",
    "fighters": "fighters_clean.parquet",
}

MISSING_TOKENS = {
    "",
    "--",
    "---",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
}

RESULT_MAPPING = {
    "w": "W",
    "win": "W",
    "winner": "W",
    "l": "L",
    "loss": "L",
    "loser": "L",
    "d": "D",
    "draw": "D",
    "nc": "NC",
    "no contest": "NC",
    "no-contest": "NC",
}

METHOD_MAPPING = {
    "U-DEC": "U-DEC",
    "UNANIMOUS DECISION": "U-DEC",
    "S-DEC": "S-DEC",
    "SPLIT DECISION": "S-DEC",
    "M-DEC": "M-DEC",
    "MAJORITY DECISION": "M-DEC",
    "KO/TKO": "KO/TKO",
    "TKO": "KO/TKO",
    "KO": "KO/TKO",
    "SUB": "SUB",
    "SUBMISSION": "SUB",
    "DQ": "DQ",
    "DISQUALIFICATION": "DQ",
    "NC": "NC",
    "NO CONTEST": "NC",
    "OVERTURNED": "OVERTURNED",
    "OTHER": "OTHER",
}

STANCE_MAPPING = {
    "orthodox": "Orthodox",
    "southpaw": "Southpaw",
    "switch": "Switch",
    "open stance": "Open Stance",
    "sideways": "Sideways",
}

COUNT_STAT_COLUMNS = [
    "knockdowns",
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "submission_attempts",
    "reversals",
    "control_seconds",
    "head_landed",
    "head_attempted",
    "body_landed",
    "body_attempted",
    "leg_landed",
    "leg_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
]

LANDED_ATTEMPTED_PAIRS = [
    ("sig_str_landed", "sig_str_attempted"),
    ("total_str_landed", "total_str_attempted"),
    ("takedowns_landed", "takedowns_attempted"),
    ("head_landed", "head_attempted"),
    ("body_landed", "body_attempted"),
    ("leg_landed", "leg_attempted"),
    ("distance_landed", "distance_attempted"),
    ("clinch_landed", "clinch_attempted"),
    ("ground_landed", "ground_attempted"),
]


@dataclass
class ValidationReport:
    """Collect validation findings before raising structural errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def raise_if_errors(self) -> None:
        if not self.errors:
            return

        formatted = "\n".join(
            f"  {index}. {message}"
            for index, message in enumerate(self.errors, start=1)
        )
        raise ValueError(
            "Validation found structural errors:\n"
            f"{formatted}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean, type, and validate the raw UFCStats datasets, "
            "and write Parquet files to data/interim/ufcstats."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing the six raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the cleaned Parquet files.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow fight_details to omit completed fights. "
            "Upcoming fights without results may omit fight_details "
            "without enabling this option."
        ),
    )
    return parser.parse_args()


def clean_whitespace(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    return " ".join(str(value).split())


def normalize_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize text and convert sentinel values to missing data."""

    result = dataframe.copy()

    for column in result.columns:
        series = result[column].astype("string").map(clean_whitespace)
        missing_mask = series.str.casefold().isin(MISSING_TOKENS)
        result[column] = series.mask(missing_mask, pd.NA)

    return result


def require_columns(
    dataframe: pd.DataFrame,
    table_name: str,
    columns: Iterable[str],
) -> None:
    missing = set(columns) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def read_raw_table(raw_dir: Path, table_name: str) -> pd.DataFrame:
    path = raw_dir / INPUT_FILES[table_name]

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        encoding="utf-8-sig",
    )
    dataframe.columns = [clean_whitespace(column) for column in dataframe.columns]

    if dataframe.columns.duplicated().any():
        duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise ValueError(
            f"{path.name} contains duplicate columns: {duplicated}"
        )

    return normalize_missing_values(dataframe)


def to_string_dtype(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("string")


def to_nullable_integer(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in dataframe.columns:
            continue

        original_non_null = dataframe[column].notna()
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        failed = original_non_null & numeric.isna()

        if failed.any():
            examples = dataframe.loc[failed, column].head(5).tolist()
            raise ValueError(
                f"{column} contains unparseable numbers: {examples}"
            )

        non_integer = numeric.notna() & ((numeric % 1).abs() > 1e-9)
        if non_integer.any():
            examples = dataframe.loc[non_integer, column].head(5).tolist()
            raise ValueError(
                f"{column} contains non-integer values: {examples}"
            )

        dataframe[column] = numeric.round().astype("Int64")


def to_nullable_float(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in dataframe.columns:
            continue

        original_non_null = dataframe[column].notna()
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        failed = original_non_null & numeric.isna()

        if failed.any():
            examples = dataframe.loc[failed, column].head(5).tolist()
            raise ValueError(
                f"{column} contains unparseable numbers: {examples}"
            )

        dataframe[column] = numeric.astype("Float64")


def to_date(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in dataframe.columns:
            original_non_null = dataframe[column].notna()
            parsed = pd.to_datetime(dataframe[column], errors="coerce")
            failed = original_non_null & parsed.isna()
            if failed.any():
                examples = dataframe.loc[failed, column].head(5).tolist()
                raise ValueError(
                    f"{column} contains unparseable dates: {examples}"
                )
            dataframe[column] = parsed


def to_utc_datetime(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in dataframe.columns:
            original_non_null = dataframe[column].notna()
            parsed = pd.to_datetime(
                dataframe[column],
                errors="coerce",
                utc=True,
            )
            failed = original_non_null & parsed.isna()
            if failed.any():
                examples = dataframe.loc[failed, column].head(5).tolist()
                raise ValueError(
                    f"{column} contains unparseable UTC datetimes: {examples}"
                )
            dataframe[column] = parsed


def to_nullable_boolean(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        return

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    normalized = dataframe[column].astype("string").str.casefold()
    unknown = dataframe[column].notna() & ~normalized.isin(mapping)

    if unknown.any():
        examples = dataframe.loc[unknown, column].head(5).tolist()
        raise ValueError(
            f"{column} contains unrecognized Boolean values: {examples}"
        )

    dataframe[column] = normalized.map(mapping).astype("boolean")


def normalize_result_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    mapped = normalized.str.casefold().map(RESULT_MAPPING)
    return mapped.fillna(normalized.str.upper()).astype("string")


def normalize_method_series(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )
    return normalized.map(METHOD_MAPPING).fillna(normalized).astype("string")


def normalize_stance_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.casefold().str.strip()
    mapped = normalized.map(STANCE_MAPPING)
    fallback = normalized.str.title()
    return mapped.fillna(fallback).astype("string")


def normalize_category_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        dataframe,
        "events.csv",
        [
            "event_id",
            "event_name",
            "event_date",
            "event_location",
            "event_url",
        ],
    )

    result = dataframe.copy()
    to_string_dtype(
        result,
        ["event_id", "event_name", "event_location", "event_url"],
    )
    to_date(result, ["event_date"])
    return result


def clean_fights(dataframe: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        dataframe,
        "fights.csv",
        [
            "fight_id",
            "event_id",
            "event_date",
            "fight_url",
            "fighter_1_id",
            "fighter_1_name",
            "fighter_1_result",
            "fighter_2_id",
            "fighter_2_name",
            "fighter_2_result",
            "winner_id",
            "weight_class",
            "method",
            "finish_round",
            "finish_time",
        ],
    )

    result = dataframe.copy()
    to_string_dtype(result, result.columns)
    to_date(result, ["event_date"])
    to_nullable_integer(
        result,
        [
            "fight_order",
            "finish_round",
            "fighter_1_knockdowns",
            "fighter_2_knockdowns",
            "fighter_1_sig_strikes",
            "fighter_2_sig_strikes",
            "fighter_1_takedowns",
            "fighter_2_takedowns",
            "fighter_1_submission_attempts",
            "fighter_2_submission_attempts",
        ],
    )

    result["fighter_1_result"] = normalize_result_series(
        result["fighter_1_result"]
    )
    result["fighter_2_result"] = normalize_result_series(
        result["fighter_2_result"]
    )
    result["method"] = normalize_method_series(result["method"])
    result["weight_class"] = normalize_category_series(result["weight_class"])

    return result


def clean_fight_details(dataframe: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        dataframe,
        "fight_details.csv",
        [
            "fight_id",
            "event_id",
            "event_date",
            "fighter_1_id",
            "fighter_1_result",
            "fighter_2_id",
            "fighter_2_result",
            "winner_id",
            "method",
            "finish_round",
            "finish_time_seconds",
            "scheduled_rounds",
            "total_fight_time_seconds",
            "stats_available",
            "rounds_with_stats",
            "scraped_at_utc",
        ],
    )

    result = dataframe.copy()
    to_string_dtype(result, result.columns)
    to_date(result, ["event_date"])
    to_utc_datetime(result, ["scraped_at_utc"])
    to_nullable_integer(
        result,
        [
            "finish_round",
            "finish_time_seconds",
            "scheduled_rounds",
            "total_fight_time_seconds",
            "rounds_with_stats",
        ],
    )
    to_nullable_boolean(result, "stats_available")

    result["fighter_1_result"] = normalize_result_series(
        result["fighter_1_result"]
    )
    result["fighter_2_result"] = normalize_result_series(
        result["fighter_2_result"]
    )
    result["method"] = normalize_method_series(result["method"])
    if "bout_type" in result.columns:
        result["bout_type"] = normalize_category_series(result["bout_type"])

    return result


def clean_stats_table(
    dataframe: pd.DataFrame,
    table_name: str,
    include_round: bool,
) -> pd.DataFrame:
    required = [
        "fight_id",
        "event_id",
        "event_date",
        "fighter_id",
        "fighter_result",
        "fighter_order",
        "opponent_id",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "control_seconds",
    ]
    if include_round:
        required.append("round")

    require_columns(dataframe, table_name, required)

    result = dataframe.copy()
    to_string_dtype(result, result.columns)
    to_date(result, ["event_date"])
    to_nullable_integer(
        result,
        ["fighter_order", "round", *COUNT_STAT_COLUMNS],
    )
    to_nullable_float(result, ["sig_str_pct", "takedown_pct"])
    result["fighter_result"] = normalize_result_series(
        result["fighter_result"]
    )

    return result


def clean_fighters(dataframe: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        dataframe,
        "fighters.csv",
        [
            "fighter_id",
            "fighter_name",
            "fighter_url",
            "height_cm",
            "weight_kg",
            "reach_cm",
            "stance",
            "date_of_birth",
            "profile_status",
            "scraped_at_utc",
        ],
    )

    result = dataframe.copy()

    to_string_dtype(
        result,
        result.columns,
    )

    to_date(
        result,
        ["date_of_birth"],
    )

    to_utc_datetime(
        result,
        ["scraped_at_utc"],
    )

    to_nullable_integer(
        result,
        [
            "wins_current",
            "losses_current",
            "draws_current",
            "no_contests_current",
        ],
    )

    to_nullable_float(
        result,
        [
            "height_cm",
            "weight_lbs",
            "weight_kg",
            "reach_inches",
            "reach_cm",
            "slpm_current",
            "striking_accuracy_current",
            "sapm_current",
            "striking_defense_current",
            "takedown_average_current",
            "takedown_accuracy_current",
            "takedown_defense_current",
            "submission_average_current",
        ],
    )

    # Treat implausible weights as unknown.
    invalid_weight = (
        result["weight_kg"].notna()
        & ~result["weight_kg"].between(
            40,
            180,
            inclusive="both",
        )
    )

    result.loc[
        invalid_weight,
        ["weight_lbs", "weight_kg"],
    ] = pd.NA

    # Missing stance is a genuine categorical state for modeling purposes.
    # Keep numerical/temporal profile fields (height, reach, DOB, etc.) missing
    # here; their imputation belongs to the downstream training preprocessing
    # pipeline so that statistics are learned from the training split only.
    result["stance"] = (
        normalize_stance_series(result["stance"])
        .fillna("Unknown")
    )

    result["profile_status"] = (
        result["profile_status"]
        .astype("string")
        .str.casefold()
    )

    return result

def sample_values(series: pd.Series, limit: int = 5) -> list[str]:
    return [str(value) for value in series.dropna().astype(str).head(limit).tolist()]


def check_primary_key(
    dataframe: pd.DataFrame,
    table_name: str,
    columns: list[str],
    report: ValidationReport,
) -> None:
    null_mask = dataframe[columns].isna().any(axis=1)
    if null_mask.any():
        report.error(
            f"{table_name}: {int(null_mask.sum())} rows have a null primary "
            f"key in {columns}."
        )

    duplicated = dataframe.duplicated(columns, keep=False)
    if duplicated.any():
        examples = (
            dataframe.loc[duplicated, columns]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .to_dict(orient="records")
        )
        report.error(
            f"{table_name}: {int(duplicated.sum())} rows are involved in "
            f"duplicate keys {columns}. Examples: {examples}"
        )


def check_foreign_key(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    relationship_name: str,
    report: ValidationReport,
) -> None:
    parent_values = set(parent[parent_column].dropna().astype(str))
    child_values = child[child_column].dropna().astype(str)
    orphan_mask = ~child_values.isin(parent_values)

    if orphan_mask.any():
        examples = child_values[orphan_mask].drop_duplicates().head(10).tolist()
        report.error(
            f"{relationship_name}: {int(orphan_mask.sum())} references do not "
            f"exist in the parent table. Examples: {examples}"
        )


def check_url_contains(
    dataframe: pd.DataFrame,
    column: str,
    fragment: str,
    table_name: str,
    report: ValidationReport,
) -> None:
    invalid = dataframe[column].notna() & ~dataframe[column].str.contains(
        fragment,
        regex=False,
    )
    if invalid.any():
        report.error(
            f"{table_name}.{column}: {int(invalid.sum())} URLs do not contain "
            f"{fragment}. Examples: {sample_values(dataframe.loc[invalid, column])}"
        )


def check_non_negative(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
    report: ValidationReport,
) -> None:
    for column in columns:
        if column not in dataframe.columns:
            continue
        invalid = dataframe[column].notna() & (dataframe[column] < 0)
        if invalid.any():
            report.error(
                f"{table_name}.{column}: {int(invalid.sum())} negative values."
            )


def check_range(
    dataframe: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
    table_name: str,
    report: ValidationReport,
    *,
    severity: str = "error",
) -> None:
    if column not in dataframe.columns:
        return

    invalid = dataframe[column].notna() & ~dataframe[column].between(
        lower,
        upper,
        inclusive="both",
    )
    if not invalid.any():
        return

    message = (
        f"{table_name}.{column}: {int(invalid.sum())} values outside "
        f"[{lower}, {upper}]. Examples: "
        f"{sample_values(dataframe.loc[invalid, column])}"
    )
    if severity == "warning":
        report.warning(message)
    else:
        report.error(message)


def check_landed_not_above_attempted(
    dataframe: pd.DataFrame,
    table_name: str,
    report: ValidationReport,
) -> None:
    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        if landed not in dataframe.columns or attempted not in dataframe.columns:
            continue
        invalid = (
            dataframe[landed].notna()
            & dataframe[attempted].notna()
            & (dataframe[landed] > dataframe[attempted])
        )
        if invalid.any():
            report.error(
                f"{table_name}: {int(invalid.sum())} rows have "
                f"{landed} > {attempted}."
            )


def get_upcoming_fight_mask(fights: pd.DataFrame) -> pd.Series:
    """
    Identify upcoming fights from the event date, empty results, and empty winner_id.

    Requiring every condition keeps incomplete historical fights invalid.
    """

    today = pd.Timestamp.now().normalize()

    event_date = pd.to_datetime(
        fights["event_date"],
        errors="coerce",
    )

    return (
        event_date.notna()
        & event_date.ge(today)
        & fights["fighter_1_result"].isna()
        & fights["fighter_2_result"].isna()
        & fights["winner_id"].isna()
    )


def check_result_logic(fights: pd.DataFrame, report: ValidationReport) -> None:
    """
    Validate completed-fight results while allowing empty results for upcoming fights.
    """

    allowed_resolved_pairs = {
        ("W", "L"),
        ("L", "W"),
        ("D", "D"),
        ("NC", "NC"),
    }

    upcoming_mask = get_upcoming_fight_mask(fights)

    fighter_1_result = fights["fighter_1_result"]
    fighter_2_result = fights["fighter_2_result"]

    both_missing = (
        fighter_1_result.isna()
        & fighter_2_result.isna()
    )

    partially_missing = (
        fighter_1_result.isna()
        ^ fighter_2_result.isna()
    )

    if partially_missing.any():
        examples = fights.loc[
            partially_missing,
            [
                "fight_id",
                "event_date",
                "fighter_1_result",
                "fighter_2_result",
                "winner_id",
            ],
        ].head(5).to_dict(orient="records")

        report.error(
            "fights: some fights have a result for only one fighter. "
            f"Rows: {int(partially_missing.sum())}. Examples: {examples}"
        )

    resolved_mask = ~both_missing & ~partially_missing

    invalid_resolved_indices: list[int] = []

    for index in fights.index[resolved_mask]:
        pair = (
            fights.at[index, "fighter_1_result"],
            fights.at[index, "fighter_2_result"],
        )

        if pair not in allowed_resolved_pairs:
            invalid_resolved_indices.append(index)

    if invalid_resolved_indices:
        examples = fights.loc[
            invalid_resolved_indices[:5],
            [
                "fight_id",
                "event_date",
                "fighter_1_result",
                "fighter_2_result",
            ],
        ].to_dict(orient="records")

        report.error(
            f"fights: {len(invalid_resolved_indices)} invalid result "
            f"combinations among completed fights. Examples: {examples}"
        )

    # An (NA, NA) pair is valid only for an upcoming fight without winner_id.
    invalid_unresolved = both_missing & ~upcoming_mask

    if invalid_unresolved.any():
        examples = fights.loc[
            invalid_unresolved,
            [
                "fight_id",
                "event_name",
                "event_date",
                "fighter_1_result",
                "fighter_2_result",
                "winner_id",
            ],
        ].head(5).to_dict(orient="records")

        report.error(
            "fights: some result-less fights do not meet the "
            "upcoming-fight conditions. "
            f"Rows: {int(invalid_unresolved.sum())}. "
            f"Examples: {examples}"
        )

    expected_winner = pd.Series(
        pd.NA,
        index=fights.index,
        dtype="string",
    )

    expected_winner = expected_winner.mask(
        fights["fighter_1_result"].eq("W"),
        fights["fighter_1_id"],
    )

    expected_winner = expected_winner.mask(
        fights["fighter_2_result"].eq("W"),
        fights["fighter_2_id"],
    )

    winner_mismatch = (
        expected_winner.notna()
        & fights["winner_id"].astype("string").ne(expected_winner)
    )

    if winner_mismatch.any():
        examples = fights.loc[
            winner_mismatch,
            [
                "fight_id",
                "fighter_1_id",
                "fighter_1_result",
                "fighter_2_id",
                "fighter_2_result",
                "winner_id",
            ],
        ].head(5).to_dict(orient="records")

        report.error(
            f"fights: {int(winner_mismatch.sum())} winner_id values do not "
            f"match W/L. Examples: {examples}"
        )

    # Draws, no contests, and upcoming fights must not have a winner.
    should_have_no_winner = (
        expected_winner.isna()
        & fights["winner_id"].notna()
    )

    if should_have_no_winner.any():
        examples = fights.loc[
            should_have_no_winner,
            [
                "fight_id",
                "event_date",
                "fighter_1_result",
                "fighter_2_result",
                "winner_id",
            ],
        ].head(5).to_dict(orient="records")

        report.error(
            "fights: some fights without a valid winner contain winner_id. "
            f"Rows: {int(should_have_no_winner.sum())}. "
            f"Examples: {examples}"
        )

    report.note(
        "Accepted upcoming fights without results: "
        f"{int(upcoming_mask.sum())}."
    )

def compare_shared_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    columns: Iterable[str],
    relationship_name: str,
    report: ValidationReport,
) -> None:
    available = [
        column
        for column in columns
        if column in left.columns and column in right.columns
    ]
    if not available:
        return

    merged = left[[key, *available]].merge(
        right[[key, *available]],
        on=key,
        how="inner",
        suffixes=("_left", "_right"),
    )

    for column in available:
        left_values = merged[f"{column}_left"]
        right_values = merged[f"{column}_right"]
        mismatch = ~(
            left_values.eq(right_values)
            | (left_values.isna() & right_values.isna())
        )
        if mismatch.any():
            examples = merged.loc[
                mismatch,
                [key, f"{column}_left", f"{column}_right"],
            ].head(5).to_dict(orient="records")
            report.error(
                f"{relationship_name}: {int(mismatch.sum())} values for "
                f"{column} do not match. Examples: {examples}"
            )


def validate_all(
    tables: dict[str, pd.DataFrame],
    report: ValidationReport,
    allow_partial: bool,
) -> None:
    events = tables["events"]
    fights = tables["fights"]
    details = tables["fight_details"]
    totals = tables["fight_totals"]
    rounds = tables["round_stats"]
    fighters = tables["fighters"]

    check_primary_key(events, "events", ["event_id"], report)
    check_primary_key(fights, "fights", ["fight_id"], report)
    check_primary_key(details, "fight_details", ["fight_id"], report)
    check_primary_key(
        totals,
        "fight_totals",
        ["fight_id", "fighter_id"],
        report,
    )
    check_primary_key(
        rounds,
        "round_stats",
        ["fight_id", "fighter_id", "round"],
        report,
    )
    check_primary_key(fighters, "fighters", ["fighter_id"], report)

    check_url_contains(events, "event_url", "/event-details/", "events", report)
    check_url_contains(fights, "fight_url", "/fight-details/", "fights", report)
    check_url_contains(fighters, "fighter_url", "/fighter-details/", "fighters", report)

    check_foreign_key(
        fights,
        "event_id",
        events,
        "event_id",
        "fights.event_id → events.event_id",
        report,
    )
    for column in ["fighter_1_id", "fighter_2_id"]:
        check_foreign_key(
            fights,
            column,
            fighters,
            "fighter_id",
            f"fights.{column} → fighters.fighter_id",
            report,
        )

    for table_name in ["fight_details", "fight_totals", "round_stats"]:
        check_foreign_key(
            tables[table_name],
            "fight_id",
            fights,
            "fight_id",
            f"{table_name}.fight_id → fights.fight_id",
            report,
        )

    for table_name in ["fight_totals", "round_stats"]:
        check_foreign_key(
            tables[table_name],
            "fighter_id",
            fighters,
            "fighter_id",
            f"{table_name}.fighter_id → fighters.fighter_id",
            report,
        )
        check_foreign_key(
            tables[table_name],
            "opponent_id",
            fighters,
            "fighter_id",
            f"{table_name}.opponent_id → fighters.fighter_id",
            report,
        )

    same_fighter = fights["fighter_1_id"].eq(fights["fighter_2_id"])
    if same_fighter.any():
        report.error(
            f"fights: {int(same_fighter.sum())} fights contain the same "
            "fighter on both sides."
        )

    check_result_logic(fights, report)

    detail_ids = set(
        details["fight_id"]
        .dropna()
        .astype(str)
    )

    fight_ids = set(
        fights["fight_id"]
        .dropna()
        .astype(str)
    )

    upcoming_mask = get_upcoming_fight_mask(fights)

    upcoming_fight_ids = set(
        fights.loc[
            upcoming_mask,
            "fight_id",
        ]
        .dropna()
        .astype(str)
    )

    # Require fight_details for completed fights, not upcoming fights.
    required_detail_ids = fight_ids - upcoming_fight_ids

    missing_required_details = (
        required_detail_ids
        - detail_ids
    )

    extra_details = (
        detail_ids
        - fight_ids
    )

    if missing_required_details:
        message = (
            "fight_details does not contain "
            f"{len(missing_required_details)} fight_id values for completed "
            "fights. "
            f"Examples: {sorted(missing_required_details)[:10]}"
        )

        if allow_partial:
            report.warning(message)
        else:
            report.error(message)

    if extra_details:
        report.error(
            "fight_details contains "
            f"{len(extra_details)} fight_id values not present in fights."
        )

    upcoming_with_details = (
        upcoming_fight_ids
        & detail_ids
    )

    if upcoming_with_details:
        report.warning(
            "fight_details contains rows for "
            f"{len(upcoming_with_details)} fights that do not yet have a result. "
            "Check whether UFCStats has begun publishing information "
            "for those fights or the rows contain placeholders. "
            f"Examples: {sorted(upcoming_with_details)[:10]}"
        )

    report.note(
        "Upcoming fights that do not require fight_details: "
        f"{len(upcoming_fight_ids)}."
    )

    compare_shared_columns(
        fights,
        details,
        "fight_id",
        [
            "event_id",
            "event_date",
            "fighter_1_id",
            "fighter_1_result",
            "fighter_2_id",
            "fighter_2_result",
            "winner_id",
            "finish_round",
        ],
        "fights ↔ fight_details",
        report,
    )

    stats_available = details["stats_available"].fillna(False)
    stats_ids = set(details.loc[stats_available, "fight_id"].astype(str))
    no_stats_ids = set(details.loc[~stats_available, "fight_id"].astype(str))
    total_ids = set(totals["fight_id"].dropna().astype(str))
    round_ids = set(rounds["fight_id"].dropna().astype(str))

    if total_ids != stats_ids:
        report.error(
            "The fight_id values in fight_totals do not exactly match "
            "fights marked stats_available=True. "
            f"Missing={len(stats_ids - total_ids)}, extra={len(total_ids - stats_ids)}."
        )
    if round_ids != stats_ids:
        report.error(
            "The fight_id values in round_stats do not exactly match "
            "fights marked stats_available=True. "
            f"Missing={len(stats_ids - round_ids)}, extra={len(round_ids - stats_ids)}."
        )
    if (total_ids | round_ids) & no_stats_ids:
        report.error(
            "Some stats_available=False fights contain statistic rows."
        )

    totals_per_fight = totals.groupby("fight_id", observed=True).size()
    invalid_totals_count = ~totals_per_fight.eq(2)
    if invalid_totals_count.any():
        examples = totals_per_fight[invalid_totals_count].head(10).to_dict()
        report.error(
            "fight_totals must contain exactly two rows per fight. "
            f"Examples: {examples}"
        )

    invalid_totals_fighters: list[str] = []
    fight_lookup = fights.set_index("fight_id")[["fighter_1_id", "fighter_2_id"]]
    for fight_id, group in totals.groupby("fight_id", observed=True):
        if fight_id not in fight_lookup.index:
            continue
        expected = set(fight_lookup.loc[fight_id].dropna().astype(str))
        actual = set(group["fighter_id"].dropna().astype(str))
        if actual != expected:
            invalid_totals_fighters.append(str(fight_id))
            if len(invalid_totals_fighters) >= 10:
                break
    if invalid_totals_fighters:
        report.error(
            "fight_totals contains fighters that do not match fights. "
            f"Example fight_id values: {invalid_totals_fighters}"
        )

    invalid_opponents = totals["fighter_id"].eq(totals["opponent_id"])
    if invalid_opponents.any():
        report.error(
            f"fight_totals: {int(invalid_opponents.sum())} rows have "
            "fighter_id == opponent_id."
        )

    rounds_per_fight = rounds.groupby("fight_id", observed=True).size()
    expected_round_rows = (
        details.loc[stats_available, ["fight_id", "rounds_with_stats"]]
        .set_index("fight_id")["rounds_with_stats"]
        .astype("Int64")
        * 2
    )
    aligned_actual = rounds_per_fight.reindex(expected_round_rows.index).fillna(0)
    bad_round_counts = aligned_actual.astype("Int64").ne(expected_round_rows)
    if bad_round_counts.any():
        examples = pd.DataFrame(
            {
                "expected": expected_round_rows[bad_round_counts],
                "actual": aligned_actual[bad_round_counts],
            }
        ).head(10).to_dict(orient="index")
        report.error(
            "round_stats does not contain 2 × rounds_with_stats rows per fight. "
            f"Examples: {examples}"
        )

    fighters_per_round = rounds.groupby(
        ["fight_id", "round"],
        observed=True,
    )["fighter_id"].nunique()
    invalid_rounds = ~fighters_per_round.eq(2)
    if invalid_rounds.any():
        examples = fighters_per_round[invalid_rounds].head(10).to_dict()
        report.error(
            "Each round must contain exactly two fighters. "
            f"Examples: {examples}"
        )

    check_landed_not_above_attempted(totals, "fight_totals", report)
    check_landed_not_above_attempted(rounds, "round_stats", report)
    check_non_negative(totals, COUNT_STAT_COLUMNS, "fight_totals", report)
    check_non_negative(rounds, ["round", *COUNT_STAT_COLUMNS], "round_stats", report)

    for table_name, dataframe in [
        ("fight_totals", totals),
        ("round_stats", rounds),
    ]:
        check_range(dataframe, "sig_str_pct", 0, 100, table_name, report)
        check_range(dataframe, "takedown_pct", 0, 100, table_name, report)
        check_range(dataframe, "fighter_order", 1, 2, table_name, report)

    check_range(fighters, "height_cm", 120, 230, "fighters", report, severity="warning")
    check_range(fighters, "weight_kg", 40, 180, "fighters", report, severity="warning")
    check_range(fighters, "reach_cm", 120, 250, "fighters", report, severity="warning")

    for column in [
        "striking_accuracy_current",
        "striking_defense_current",
        "takedown_accuracy_current",
        "takedown_defense_current",
    ]:
        check_range(fighters, column, 0, 1, "fighters", report)

    future_dob = (
        fighters["date_of_birth"].notna()
        & (fighters["date_of_birth"] > pd.Timestamp.now().normalize())
    )
    if future_dob.any():
        report.error(
            f"fighters: {int(future_dob.sum())} future dates of birth."
        )

    report.note(
        "Fights with fight_details but no detailed statistics: "
        f"{len(no_stats_ids)}."
    )
    report.note(
        "Missing fighter stance values are normalized to the explicit "
        "categorical level 'Unknown'."
    )
    report.note(
        "Missing fighter height, reach, weight, and date_of_birth values are "
        "preserved as missing at the cleaning stage. Any weight-class-based "
        "imputation must be fitted on the training split downstream to avoid "
        "data leakage."
    )
    report.note(
        "Fighter columns ending in _current are current snapshots "
        "and require temporal handling before historical modeling."
    )


def dataframe_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "null_counts": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
            if int(count) > 0
        },
    }


def ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "The pyarrow dependency is required to write Parquet files. "
            "Install it with: python -m pip install pyarrow"
        ) from error


def write_parquet_atomic(dataframe: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    dataframe.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    os.replace(temporary, target)


def write_json_atomic(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def main() -> None:
    args = parse_arguments()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Input directory: {raw_dir}")
    print(f"Output directory:  {output_dir}")
    print()

    print("Reading raw CSV files...")
    raw_tables = {
        table_name: read_raw_table(raw_dir, table_name)
        for table_name in INPUT_FILES
    }

    print("Cleaning data and assigning types...")
    clean_tables = {
        "events": clean_events(raw_tables["events"]),
        "fights": clean_fights(raw_tables["fights"]),
        "fight_details": clean_fight_details(raw_tables["fight_details"]),
        "fight_totals": clean_stats_table(
            raw_tables["fight_totals"],
            "fight_totals.csv",
            include_round=False,
        ),
        "round_stats": clean_stats_table(
            raw_tables["round_stats"],
            "round_stats.csv",
            include_round=True,
        ),
        "fighters": clean_fighters(raw_tables["fighters"]),
    }

    print("Validating keys, relationships, and statistical consistency...")
    report = ValidationReport()
    validate_all(
        clean_tables,
        report,
        allow_partial=args.allow_partial,
    )
    report.raise_if_errors()

    ensure_parquet_support()

    print("Writing cleaned Parquet files...")
    for table_name, dataframe in clean_tables.items():
        target = output_dir / OUTPUT_FILES[table_name]
        write_parquet_atomic(dataframe, target)
        print(f"  {table_name}: {len(dataframe):,} rows → {target.name}")

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_directory": str(raw_dir),
        "output_directory": str(output_dir),
        "allow_partial": bool(args.allow_partial),
        "tables": {
            table_name: dataframe_summary(dataframe)
            for table_name, dataframe in clean_tables.items()
        },
        "validation": {
            "status": "passed",
            "errors": report.errors,
            "warnings": report.warnings,
            "notes": report.notes,
        },
    }
    report_path = output_dir / "cleaning_report.json"
    write_json_atomic(report_payload, report_path)

    print()
    print("Cleaning completed successfully.")
    print(f"Report: {report_path}")

    if report.warnings:
        print()
        print(f"Warnings: {len(report.warnings)}")
        for warning in report.warnings:
            print(f"- {warning}")

    if report.notes:
        print()
        print("Notes:")
        for note in report.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
