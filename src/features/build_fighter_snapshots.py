"""Build each fighter's historical state immediately before every fight without using future information."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "ufcstats"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ufcstats"

FIGHTS_PATH = INPUT_DIR / "fights_clean.parquet"
FIGHT_DETAILS_PATH = INPUT_DIR / "fight_details_clean.parquet"
FIGHT_TOTALS_PATH = INPUT_DIR / "fight_totals_clean.parquet"

OUTPUT_PATH = OUTPUT_DIR / "fighter_snapshots.parquet"


@dataclass
class FighterState:
    """Accumulated historical state for one fighter."""

    # Fight record.
    fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    no_contests: int = 0

    current_win_streak: int = 0
    current_loss_streak: int = 0

    # Sum of durations whose values are actually known.
    total_fight_time_seconds: int = 0
    # A missing duration makes the exact historical total unknown.
    fights_without_duration: int = 0
    last_fight_date: pd.Timestamp | None = None

    # Detailed-statistics coverage.
    fights_with_stats: int = 0
    stats_fight_time_seconds: int = 0

    # Control time may be missing even when all other statistics are available.
    fights_with_control_stats: int = 0

    # Striking.
    sig_str_landed: int = 0
    sig_str_attempted: int = 0
    sig_str_absorbed: int = 0
    sig_str_absorbed_attempted: int = 0

    # Grappling.
    takedowns_landed: int = 0
    takedowns_attempted: int = 0
    takedowns_conceded: int = 0
    takedowns_faced: int = 0

    control_seconds: int = 0
    knockdowns: int = 0
    knockdowns_absorbed: int = 0
    submission_attempts: int = 0
    reversals: int = 0


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


def normalize_result(value: object, fight_id: str) -> str:
    """
    Normalize a fighter result.

    Accepted values:
    - W: win
    - L: loss
    - D: draw
    - NC: no contest
    """

    if pd.isna(value):
        raise ValueError(
            f"Missing result for fight {fight_id}."
        )

    normalized = str(value).strip().upper()

    aliases = {
        "W": "W",
        "L": "L",
        "D": "D",
        "NC": "NC",
        "N/C": "NC",
        "NO CONTEST": "NC",
    }

    if normalized not in aliases:
        raise ValueError(
            f"Unknown result in {fight_id}: {value!r}"
        )

    return aliases[normalized]


def normalize_optional_string(value: object) -> str | None:
    """Convert blank or missing values to None."""

    if pd.isna(value):
        return None

    normalized = str(value).strip()

    return normalized if normalized else None


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float | Any:
    """Calculate a ratio and return NA when the denominator is zero."""

    if denominator == 0:
        return pd.NA

    return numerator / denominator


def safe_per_minute(
    total: int,
    total_seconds: int,
) -> float | Any:
    """Calculate a per-minute rate using only time with available statistics."""

    if total_seconds == 0:
        return pd.NA

    return total / (total_seconds / 60.0)


def safe_defense_rate(
    conceded: int,
    faced: int,
) -> float | Any:
    """
    Calculate defense as the proportion of attempts avoided.

    Example:
        1 - significant strikes absorbed / opponent attempts
    """

    if faced == 0:
        return pd.NA

    return 1.0 - (conceded / faced)


def validate_fight_result(
    fight_id: str,
    fighter_1_id: str,
    fighter_2_id: str,
    fighter_1_result: str,
    fighter_2_result: str,
    winner_id: str | None,
) -> None:
    """Validate that both fighter results are compatible."""

    valid_result_pairs = {
        ("W", "L"),
        ("L", "W"),
        ("D", "D"),
        ("NC", "NC"),
    }

    result_pair = (fighter_1_result, fighter_2_result)

    if result_pair not in valid_result_pairs:
        raise ValueError(
            f"Inconsistent result pair in {fight_id}: "
            f"{result_pair}"
        )

    if fighter_1_result == "W" and winner_id != fighter_1_id:
        raise ValueError(
            f"Inconsistent winner_id in {fight_id}: "
            f"expected {fighter_1_id}"
        )

    if fighter_2_result == "W" and winner_id != fighter_2_id:
        raise ValueError(
            f"Inconsistent winner_id in {fight_id}: "
            f"expected {fighter_2_id}"
        )

    if fighter_1_result in {"D", "NC"} and winner_id is not None:
        raise ValueError(
            f"Fight {fight_id} should not have a winner."
        )


def create_snapshot(
    *,
    fight_id: str,
    event_id: str,
    event_name: str,
    event_date: pd.Timestamp,
    fight_order: object,
    weight_class: object,
    fighter_id: str,
    fighter_name: str,
    opponent_id: str,
    opponent_name: str,
    state: FighterState,
) -> dict[str, object]:
    """Create a fighter snapshot immediately before the current fight."""

    completed_results = (
        state.wins
        + state.losses
        + state.draws
    )

    win_rate = safe_ratio(
        state.wins,
        completed_results,
    )

    if state.last_fight_date is None:
        days_since_last_fight: int | Any = pd.NA
    else:
        days_since_last_fight = (
            event_date - state.last_fight_date
        ).days

        if days_since_last_fight < 0:
            raise ValueError(
                f"Negative time interval for {fighter_id} in {fight_id}: "
                f"{days_since_last_fight} days."
            )

    stats_coverage = safe_ratio(
        state.fights_with_stats,
        state.fights,
    )

    control_stats_coverage = safe_ratio(
        state.fights_with_control_stats,
        state.fights,
    )

    return {
        # Snapshot identifiers.
        "fight_id": fight_id,
        "event_id": event_id,
        "event_name": event_name,
        "event_date": event_date,
        "fight_order": fight_order,
        "weight_class": weight_class,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,

        # Prior fight record.
        "fights_before": state.fights,
        "wins_before": state.wins,
        "losses_before": state.losses,
        "draws_before": state.draws,
        "no_contests_before": state.no_contests,
        "win_rate_before": win_rate,
        "current_win_streak_before": state.current_win_streak,
        "current_loss_streak_before": state.current_loss_streak,
        "days_since_last_fight": days_since_last_fight,
        "total_fight_time_seconds_before": (
            state.total_fight_time_seconds
            if state.fights_without_duration == 0
            else pd.NA
        ),

        # Prior statistics coverage.
        "fights_with_stats_before": state.fights_with_stats,
        "stats_fight_time_seconds_before": (
            state.stats_fight_time_seconds
        ),
        "stats_coverage_before": stats_coverage,
        "fights_with_control_stats_before": (
            state.fights_with_control_stats
        ),
        "control_stats_coverage_before": (
            control_stats_coverage
        ),

        # Prior striking totals.
        "sig_str_landed_before": state.sig_str_landed,
        "sig_str_attempted_before": state.sig_str_attempted,
        "sig_str_absorbed_before": state.sig_str_absorbed,
        "sig_str_absorbed_attempted_before": (
            state.sig_str_absorbed_attempted
        ),
        "sig_str_landed_per_min_before": safe_per_minute(
            state.sig_str_landed,
            state.stats_fight_time_seconds,
        ),
        "sig_str_absorbed_per_min_before": safe_per_minute(
            state.sig_str_absorbed,
            state.stats_fight_time_seconds,
        ),
        "sig_str_accuracy_before": safe_ratio(
            state.sig_str_landed,
            state.sig_str_attempted,
        ),
        "sig_str_defense_before": safe_defense_rate(
            state.sig_str_absorbed,
            state.sig_str_absorbed_attempted,
        ),

        # Prior takedown totals.
        "takedowns_landed_before": state.takedowns_landed,
        "takedowns_attempted_before": state.takedowns_attempted,
        "takedowns_conceded_before": state.takedowns_conceded,
        "takedowns_faced_before": state.takedowns_faced,
        "takedown_accuracy_before": safe_ratio(
            state.takedowns_landed,
            state.takedowns_attempted,
        ),
        "takedown_defense_before": safe_defense_rate(
            state.takedowns_conceded,
            state.takedowns_faced,
        ),

        # Prior grappling and control totals.
        "control_seconds_before": state.control_seconds,
        "knockdowns_before": state.knockdowns,
        "knockdowns_absorbed_before": (
            state.knockdowns_absorbed
        ),
        "submission_attempts_before": (
            state.submission_attempts
        ),
        "reversals_before": state.reversals,
    }


def update_competitive_state(
    state: FighterState,
    *,
    result: str,
    event_date: pd.Timestamp,
    fight_time_seconds: int | None,
) -> None:
    """
    Update the fight record after saving the snapshot.

    This update runs whether or not detailed statistics are available.
    """

    state.fights += 1
    if fight_time_seconds is None:
        # Preserve the result and the chronological history; never assume
        # that unknown duration means a zero-second fight.
        state.fights_without_duration += 1
    else:
        state.total_fight_time_seconds += fight_time_seconds
    state.last_fight_date = event_date

    if result == "W":
        state.wins += 1
        state.current_win_streak += 1
        state.current_loss_streak = 0

    elif result == "L":
        state.losses += 1
        state.current_loss_streak += 1
        state.current_win_streak = 0

    elif result == "D":
        state.draws += 1
        state.current_win_streak = 0
        state.current_loss_streak = 0

    elif result == "NC":
        state.no_contests += 1
        state.current_win_streak = 0
        state.current_loss_streak = 0

    else:
        raise ValueError(f"Unsupported result: {result}")


def update_statistical_state(
    state: FighterState,
    *,
    own_stats: Any,
    opponent_stats: Any,
    fight_time_seconds: int,
) -> None:
    """
    Update performance statistics after saving the snapshot.

    Absorbed or conceded metrics are taken from the opponent row.
    """

    state.fights_with_stats += 1
    state.stats_fight_time_seconds += fight_time_seconds

    # Offensive striking.
    state.sig_str_landed += int(own_stats.sig_str_landed)
    state.sig_str_attempted += int(own_stats.sig_str_attempted)

    # Opponent striking.
    state.sig_str_absorbed += int(opponent_stats.sig_str_landed)
    state.sig_str_absorbed_attempted += int(
        opponent_stats.sig_str_attempted
    )

    # Offensive takedowns.
    state.takedowns_landed += int(own_stats.takedowns_landed)
    state.takedowns_attempted += int(
        own_stats.takedowns_attempted
    )

    # Opponent takedowns.
    state.takedowns_conceded += int(
        opponent_stats.takedowns_landed
    )
    state.takedowns_faced += int(
        opponent_stats.takedowns_attempted
    )

    # Missing UFCStats control time means unknown, not zero.
    if pd.notna(own_stats.control_seconds):
        state.fights_with_control_stats += 1
        state.control_seconds += int(own_stats.control_seconds)

    # Remaining grappling metrics.
    state.knockdowns += int(own_stats.knockdowns)
    state.knockdowns_absorbed += int(
        opponent_stats.knockdowns
    )
    state.submission_attempts += int(
        own_stats.submission_attempts
    )
    state.reversals += int(own_stats.reversals)


def validate_repeated_fighters_in_event(
    fights: pd.DataFrame,
) -> int:
    """
    Detect fighters with multiple fights in the same event.

    This can occur in older tournament-format events.
    """

    fighter_1_rows = fights[
        ["fight_id", "event_id", "fighter_1_id"]
    ].rename(
        columns={"fighter_1_id": "fighter_id"}
    )

    fighter_2_rows = fights[
        ["fight_id", "event_id", "fighter_2_id"]
    ].rename(
        columns={"fighter_2_id": "fighter_id"}
    )

    participants = pd.concat(
        [fighter_1_rows, fighter_2_rows],
        ignore_index=True,
    )

    group_sizes = (
        participants
        .groupby(["event_id", "fighter_id"])["fight_id"]
        .size()
    )

    repeated_groups = group_sizes[group_sizes > 1]

    if not repeated_groups.empty:
        repeated_pairs = repeated_groups.reset_index()[
            ["event_id", "fighter_id"]
        ]

        repeated_fights = fights.merge(
            repeated_pairs,
            left_on=["event_id", "fighter_1_id"],
            right_on=["event_id", "fighter_id"],
            how="inner",
        )

        repeated_fights_2 = fights.merge(
            repeated_pairs,
            left_on=["event_id", "fighter_2_id"],
            right_on=["event_id", "fighter_id"],
            how="inner",
        )

        repeated_fight_ids = pd.concat(
            [
                repeated_fights["fight_id"],
                repeated_fights_2["fight_id"],
            ],
            ignore_index=True,
        ).unique()

        missing_order = fights.loc[
            fights["fight_id"].isin(repeated_fight_ids),
            "fight_order",
        ].isna()

        if missing_order.any():
            raise ValueError(
                "Some fighters have multiple fights in one "
                "event but no fight_order."
            )

    return int(len(repeated_groups))


def validate_fight_totals(
    fights: pd.DataFrame,
    fight_totals: pd.DataFrame,
) -> None:
    """Validate fight_totals coverage, keys, and consistency."""

    duplicated = fight_totals.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicated.any():
        examples = fight_totals.loc[
            duplicated,
            ["fight_id", "fighter_id"],
        ].head(10)

        raise ValueError(
            "fight_totals contains duplicate rows for "
            "fight_id + fighter_id:\n"
            f"{examples.to_string(index=False)}"
        )

    rows_per_fight = fight_totals.groupby("fight_id").size()

    invalid_group_sizes = rows_per_fight[
        rows_per_fight != 2
    ]

    if not invalid_group_sizes.empty:
        raise ValueError(
            "Some fights in fight_totals do not contain exactly two rows:\n"
            f"{invalid_group_sizes.head(10)}"
        )

    unknown_fights = (
        set(fight_totals["fight_id"].astype(str))
        - set(fights["fight_id"].astype(str))
    )

    if unknown_fights:
        raise ValueError(
            "fight_totals contains unknown fights. "
            f"Examples: {sorted(unknown_fights)[:10]}"
        )

    counts = (
        fight_totals.groupby("fight_id")
        .size()
        .rename("totals_rows")
    )

    coverage = fights[
        ["fight_id", "stats_available"]
    ].merge(
        counts,
        left_on="fight_id",
        right_index=True,
        how="left",
    )

    coverage["totals_rows"] = (
        coverage["totals_rows"]
        .fillna(0)
        .astype("int64")
    )

    expected_rows = coverage["stats_available"].map(
        {True: 2, False: 0}
    )

    inconsistent_coverage = (
        coverage["totals_rows"] != expected_rows
    )

    if inconsistent_coverage.any():
        examples = coverage.loc[
            inconsistent_coverage,
            [
                "fight_id",
                "stats_available",
                "totals_rows",
            ],
        ].head(10)

        raise ValueError(
            "stats_available does not match fight_totals:\n"
            f"{examples.to_string(index=False)}"
        )

    participant_lookup = {
        str(row.fight_id): {
            str(row.fighter_1_id): str(row.fighter_2_id),
            str(row.fighter_2_id): str(row.fighter_1_id),
        }
        for row in fights.itertuples(index=False)
    }

    invalid_participants: list[dict[str, str]] = []

    for row in fight_totals.itertuples(index=False):
        fight_id = str(row.fight_id)
        fighter_id = str(row.fighter_id)
        opponent_id = str(row.opponent_id)

        expected_opponent = participant_lookup[
            fight_id
        ].get(fighter_id)

        if expected_opponent != opponent_id:
            invalid_participants.append(
                {
                    "fight_id": fight_id,
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "expected_opponent_id": str(expected_opponent),
                }
            )

            if len(invalid_participants) >= 10:
                break

    if invalid_participants:
        raise ValueError(
            "Inconsistent participants or opponents were found in "
            "fight_totals:\n"
            f"{pd.DataFrame(invalid_participants).to_string(index=False)}"
        )

    non_negative_columns = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
    ]

    null_counts = (
        fight_totals[non_negative_columns]
        .isna()
        .sum()
    )
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        raise ValueError(
            "Required statistics contain missing values:\n"
            f"{null_counts}"
        )

    negative_values = (
        fight_totals[non_negative_columns] < 0
    ).any()

    negative_columns = negative_values[
        negative_values
    ].index.tolist()

    if negative_columns:
        raise ValueError(
            "Negative statistics were found in these columns: "
            f"{negative_columns}"
        )

    # control_seconds may be missing but cannot be negative.
    invalid_control = (
        fight_totals["control_seconds"]
        .dropna()
        .lt(0)
    )

    if invalid_control.any():
        raise ValueError(
            "Negative control_seconds values were found."
        )

    invalid_sig_str = (
        fight_totals["sig_str_landed"]
        > fight_totals["sig_str_attempted"]
    )

    if invalid_sig_str.any():
        raise ValueError(
            "Landed significant strikes exceed "
            "attempted significant strikes."
        )

    invalid_takedowns = (
        fight_totals["takedowns_landed"]
        > fight_totals["takedowns_attempted"]
    )

    if invalid_takedowns.any():
        raise ValueError(
            "Landed takedowns exceed attempted takedowns."
        )


def validate_snapshots(
    snapshots: pd.DataFrame,
    fights: pd.DataFrame,
) -> None:
    """Run the main output validations."""

    expected_rows = len(fights) * 2

    if len(snapshots) != expected_rows:
        raise ValueError(
            f"Incorrect snapshot count: {len(snapshots)}. "
            f"Expected: {expected_rows}."
        )

    duplicated = snapshots.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicated.any():
        examples = snapshots.loc[
            duplicated,
            ["fight_id", "fighter_id"],
        ].head(10)

        raise ValueError(
            "Duplicate fight_id + fighter_id keys were found:\n"
            f"{examples.to_string(index=False)}"
        )

    rows_per_fight = snapshots.groupby("fight_id").size()
    invalid_rows_per_fight = rows_per_fight[
        rows_per_fight != 2
    ]

    if not invalid_rows_per_fight.empty:
        raise ValueError(
            "Some fights do not contain exactly two snapshots:\n"
            f"{invalid_rows_per_fight.head(10)}"
        )

    self_opponents = (
        snapshots["fighter_id"]
        == snapshots["opponent_id"]
    )

    if self_opponents.any():
        raise ValueError(
            "Some fighters appear as their own opponent."
        )

    historical_results = (
        snapshots["wins_before"]
        + snapshots["losses_before"]
        + snapshots["draws_before"]
        + snapshots["no_contests_before"]
    )

    inconsistent_history = (
        historical_results
        != snapshots["fights_before"]
    )

    if inconsistent_history.any():
        raise ValueError(
            "The W-L-D-NC record does not match fights_before."
        )

    invalid_stats_counts = (
        snapshots["fights_with_stats_before"]
        > snapshots["fights_before"]
    )

    if invalid_stats_counts.any():
        raise ValueError(
            "fights_with_stats_before exceeds fights_before."
        )

    invalid_control_counts = (
        snapshots["fights_with_control_stats_before"]
        > snapshots["fights_with_stats_before"]
    )

    if invalid_control_counts.any():
        raise ValueError(
            "fights_with_control_stats_before exceeds "
            "fights_with_stats_before."
        )

    negative_days = (
        snapshots["days_since_last_fight"]
        .dropna()
        .lt(0)
    )

    if negative_days.any():
        raise ValueError(
            "Negative days_since_last_fight values were found."
        )

    count_pairs = [
        (
            "sig_str_landed_before",
            "sig_str_attempted_before",
        ),
        (
            "sig_str_absorbed_before",
            "sig_str_absorbed_attempted_before",
        ),
        (
            "takedowns_landed_before",
            "takedowns_attempted_before",
        ),
        (
            "takedowns_conceded_before",
            "takedowns_faced_before",
        ),
    ]

    for landed_column, attempted_column in count_pairs:
        invalid = (
            snapshots[landed_column]
            > snapshots[attempted_column]
        )

        if invalid.any():
            raise ValueError(
                f"{landed_column} exceeds {attempted_column}."
            )

    bounded_rate_columns = [
        "win_rate_before",
        "stats_coverage_before",
        "sig_str_accuracy_before",
        "sig_str_defense_before",
        "takedown_accuracy_before",
        "takedown_defense_before",
    ]

    for column in bounded_rate_columns:
        non_null_values = snapshots[column].dropna()

        invalid = ~non_null_values.between(
            0.0,
            1.0,
            inclusive="both",
        )

        if invalid.any():
            raise ValueError(
                f"Values outside [0, 1] were found in {column}."
            )

    non_negative_rate_columns = [
        "sig_str_landed_per_min_before",
        "sig_str_absorbed_per_min_before",
    ]

    for column in non_negative_rate_columns:
        invalid = snapshots[column].dropna().lt(0)

        if invalid.any():
            raise ValueError(
                f"Negative values were found in {column}."
            )

    invalid_sequences: list[str] = []

    for fighter_id, fighter_history in snapshots.groupby(
        "fighter_id",
        sort=False,
    ):
        actual_sequence = (
            fighter_history["fights_before"]
            .astype("int64")
            .tolist()
        )

        expected_sequence = list(
            range(len(fighter_history))
        )

        if actual_sequence != expected_sequence:
            invalid_sequences.append(str(fighter_id))

            if len(invalid_sequences) >= 10:
                break

    if invalid_sequences:
        raise ValueError(
            "fights_before does not follow the 0, 1, 2... sequence "
            f"for these fighters: {invalid_sequences}"
        )

    first_fights = snapshots[
        snapshots["fights_before"] == 0
    ]

    first_fight_count_columns = [
        "wins_before",
        "losses_before",
        "draws_before",
        "no_contests_before",
        "fights_with_stats_before",
        "stats_fight_time_seconds_before",
        "fights_with_control_stats_before",
        "sig_str_landed_before",
        "sig_str_attempted_before",
        "sig_str_absorbed_before",
        "sig_str_absorbed_attempted_before",
        "takedowns_landed_before",
        "takedowns_attempted_before",
        "takedowns_conceded_before",
        "takedowns_faced_before",
        "control_seconds_before",
        "knockdowns_before",
        "knockdowns_absorbed_before",
        "submission_attempts_before",
        "reversals_before",
    ]

    if (
        first_fights[first_fight_count_columns]
        .ne(0)
        .any()
        .any()
    ):
        raise ValueError(
            "At least one fighter's first snapshot contains prior statistics."
        )


def build_fighter_snapshots() -> pd.DataFrame:
    """Build all historical pre-fight snapshots."""

    print(f"Reading fights: {FIGHTS_PATH}")
    fights = pd.read_parquet(FIGHTS_PATH)

    print(f"Reading fight details: {FIGHT_DETAILS_PATH}")
    fight_details = pd.read_parquet(FIGHT_DETAILS_PATH)

    print(f"Reading fight totals: {FIGHT_TOTALS_PATH}")
    fight_totals = pd.read_parquet(FIGHT_TOTALS_PATH)

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
        fight_details,
        {
            "fight_id",
            "total_fight_time_seconds",
            "stats_available",
        },
        "fight_details_clean.parquet",
    )

    require_columns(
        fight_totals,
        {
            "fight_id",
            "fighter_id",
            "opponent_id",
            "knockdowns",
            "sig_str_landed",
            "sig_str_attempted",
            "takedowns_landed",
            "takedowns_attempted",
            "submission_attempts",
            "reversals",
            "control_seconds",
        },
        "fight_totals_clean.parquet",
    )

    details_for_merge = fight_details[
        [
            "fight_id",
            "total_fight_time_seconds",
            "stats_available",
        ]
    ].copy()

    fights = fights.merge(
        details_for_merge,
        on="fight_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_fights = fights["_merge"] != "both"

    if unmatched_fights.any():
        raise ValueError(
            "Some fights have no matching row in "
            "fight_details_clean.parquet."
        )

    fights = fights.drop(columns="_merge")


    # ---------------------------------------------------------------------
    # Historical snapshots must only use fights that have already happened.
    #
    # fights_clean.parquet may also contain upcoming UFC fights. Those rows
    # have no result and naturally have no fight duration or statistics yet.
    # They must not be included when reconstructing historical fighter state.
    # ---------------------------------------------------------------------

    fighter_1_result_blank = (
        fights["fighter_1_result"].isna()
        | fights["fighter_1_result"]
            .astype("string")
            .str.strip()
            .eq("")
    )

    fighter_2_result_blank = (
        fights["fighter_2_result"].isna()
        | fights["fighter_2_result"]
            .astype("string")
            .str.strip()
            .eq("")
    )

    # A fight is unresolved when neither fighter has a recorded result.
    unresolved_fights = (
        fighter_1_result_blank
        & fighter_2_result_blank
    )

    # If only one fighter has a result, the data are inconsistent rather
    # than simply representing an upcoming fight.
    partially_resolved_fights = (
        fighter_1_result_blank
        ^ fighter_2_result_blank
    )

    if partially_resolved_fights.any():
        examples = fights.loc[
            partially_resolved_fights,
            [
                "fight_id",
                "event_date",
                "fighter_1_name",
                "fighter_1_result",
                "fighter_2_name",
                "fighter_2_result",
            ],
        ].head(10)

        raise ValueError(
            "Some fights contain a result for only one fighter:\n"
            f"{examples.to_string(index=False)}"
        )

    unresolved_fight_count = int(unresolved_fights.sum())

    # Keep only completed fights for historical snapshot reconstruction.
    fights = fights.loc[~unresolved_fights].copy()

    print(
        "Unresolved/upcoming fights excluded from historical snapshots: "
        f"{unresolved_fight_count:,}"
    )



    

    required_non_null_columns = [
        "fight_id",
        "event_id",
        "event_date",
        "fighter_1_id",
        "fighter_2_id",
        "stats_available",
    ]

    null_counts = (
        fights[required_non_null_columns]
        .isna()
        .sum()
    )

    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        raise ValueError(
            "Required columns contain missing values:\n"
            f"{null_counts}"
        )

    # The cleaning stage may preserve completed fights whose detail pages
    # have not supplied a duration. Keep their W/L/D/NC history, but never
    # manufacture their timing or use it in per-minute statistics.
    missing_duration = fights["total_fight_time_seconds"].isna()
    missing_duration_count = int(missing_duration.sum())
    if missing_duration_count:
        print(
            "Completed fights with unknown duration (history preserved): "
            f"{missing_duration_count:,}"
        )
        preview_columns = [
            "fight_id", "event_date", "fighter_1_name",
            "fighter_2_name", "stats_available",
        ]
        print(fights.loc[missing_duration, preview_columns].head(10).to_string(index=False))

        # If detailed stats exist, unknown duration would make the rates
        # statistically invalid. Stop instead of silently corrupting them.
        missing_duration_with_stats = (
            missing_duration & fights["stats_available"].eq(True)
        )
        if missing_duration_with_stats.any():
            examples = fights.loc[
                missing_duration_with_stats, preview_columns
            ].head(10)
            raise ValueError(
                "Some completed fights have detailed statistics but no "
                "duration. Refresh their UFCStats detail pages before "
                "building per-minute features:\n"
                f"{examples.to_string(index=False)}"
            )

    invalid_duration = (
        fights["total_fight_time_seconds"] < 0
    )

    if invalid_duration.any():
        raise ValueError(
            "Negative fight durations were found."
        )

    validate_fight_totals(
        fights=fights,
        fight_totals=fight_totals,
    )

    repeated_event_fighters = (
        validate_repeated_fighters_in_event(fights)
    )

    # UFCStats usually assigns fight_order=1 to the main event, so higher
    # fight_order values are processed first to reconstruct chronological order.
    fights = fights.sort_values(
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
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    stats_lookup = {
        (
            str(row.fight_id),
            str(row.fighter_id),
        ): row
        for row in fight_totals.itertuples(index=False)
    }

    fighter_states: dict[str, FighterState] = {}
    snapshot_records: list[dict[str, object]] = []

    for fight in fights.itertuples(index=False):
        fight_id = str(fight.fight_id)
        event_id = str(fight.event_id)
        event_date = pd.Timestamp(fight.event_date)

        fighter_1_id = str(fight.fighter_1_id)
        fighter_2_id = str(fight.fighter_2_id)

        if fighter_1_id == fighter_2_id:
            raise ValueError(
                f"Fight {fight_id} contains fighter {fighter_1_id} twice."
            )

        fighter_1_result = normalize_result(
            fight.fighter_1_result,
            fight_id,
        )

        fighter_2_result = normalize_result(
            fight.fighter_2_result,
            fight_id,
        )

        winner_id = normalize_optional_string(
            fight.winner_id
        )

        validate_fight_result(
            fight_id=fight_id,
            fighter_1_id=fighter_1_id,
            fighter_2_id=fighter_2_id,
            fighter_1_result=fighter_1_result,
            fighter_2_result=fighter_2_result,
            winner_id=winner_id,
        )

        fighter_1_state = fighter_states.setdefault(
            fighter_1_id,
            FighterState(),
        )

        fighter_2_state = fighter_states.setdefault(
            fighter_2_id,
            FighterState(),
        )

        # Save both pre-fight states before updating either fighter.
        snapshot_records.append(
            create_snapshot(
                fight_id=fight_id,
                event_id=event_id,
                event_name=str(fight.event_name),
                event_date=event_date,
                fight_order=fight.fight_order,
                weight_class=fight.weight_class,
                fighter_id=fighter_1_id,
                fighter_name=str(fight.fighter_1_name),
                opponent_id=fighter_2_id,
                opponent_name=str(fight.fighter_2_name),
                state=fighter_1_state,
            )
        )

        snapshot_records.append(
            create_snapshot(
                fight_id=fight_id,
                event_id=event_id,
                event_name=str(fight.event_name),
                event_date=event_date,
                fight_order=fight.fight_order,
                weight_class=fight.weight_class,
                fighter_id=fighter_2_id,
                fighter_name=str(fight.fighter_2_name),
                opponent_id=fighter_1_id,
                opponent_name=str(fight.fighter_1_name),
                state=fighter_2_state,
            )
        )

        fight_time_seconds = (
            None
            if pd.isna(fight.total_fight_time_seconds)
            else int(fight.total_fight_time_seconds)
        )

        # Update both fight records after the snapshots are stored.
        update_competitive_state(
            fighter_1_state,
            result=fighter_1_result,
            event_date=event_date,
            fight_time_seconds=fight_time_seconds,
        )

        update_competitive_state(
            fighter_2_state,
            result=fighter_2_result,
            event_date=event_date,
            fight_time_seconds=fight_time_seconds,
        )

        # Update performance metrics only when both statistics rows are available.
        if bool(fight.stats_available):
            if fight_time_seconds is None:
                raise ValueError(
                    "Cannot update rate statistics without duration for "
                    f"fight {fight_id}."
                )
            fighter_1_stats = stats_lookup[
                (fight_id, fighter_1_id)
            ]
            fighter_2_stats = stats_lookup[
                (fight_id, fighter_2_id)
            ]

            update_statistical_state(
                fighter_1_state,
                own_stats=fighter_1_stats,
                opponent_stats=fighter_2_stats,
                fight_time_seconds=fight_time_seconds,
            )

            update_statistical_state(
                fighter_2_state,
                own_stats=fighter_2_stats,
                opponent_stats=fighter_1_stats,
                fight_time_seconds=fight_time_seconds,
            )

    snapshots = pd.DataFrame(snapshot_records)

    string_columns = [
        "fight_id",
        "event_id",
        "event_name",
        "weight_class",
        "fighter_id",
        "fighter_name",
        "opponent_id",
        "opponent_name",
    ]

    integer_columns = [
        "fight_order",
        "fights_before",
        "wins_before",
        "losses_before",
        "draws_before",
        "no_contests_before",
        "current_win_streak_before",
        "current_loss_streak_before",
        "days_since_last_fight",
        "total_fight_time_seconds_before",
        "fights_with_stats_before",
        "stats_fight_time_seconds_before",
        "fights_with_control_stats_before",
        "sig_str_landed_before",
        "sig_str_attempted_before",
        "sig_str_absorbed_before",
        "sig_str_absorbed_attempted_before",
        "takedowns_landed_before",
        "takedowns_attempted_before",
        "takedowns_conceded_before",
        "takedowns_faced_before",
        "control_seconds_before",
        "knockdowns_before",
        "knockdowns_absorbed_before",
        "submission_attempts_before",
        "reversals_before",
    ]

    float_columns = [
        "win_rate_before",
        "stats_coverage_before",
        "control_stats_coverage_before",
        "sig_str_landed_per_min_before",
        "sig_str_absorbed_per_min_before",
        "sig_str_accuracy_before",
        "sig_str_defense_before",
        "takedown_accuracy_before",
        "takedown_defense_before",
    ]

    for column in string_columns:
        snapshots[column] = snapshots[column].astype("string")

    for column in integer_columns:
        snapshots[column] = snapshots[column].astype("Int64")

    for column in float_columns:
        snapshots[column] = snapshots[column].astype("Float64")

    snapshots["event_date"] = pd.to_datetime(
        snapshots["event_date"]
    )

    validate_snapshots(
        snapshots=snapshots,
        fights=fights,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    fights_with_stats = int(
        fights["stats_available"].sum()
    )
    fights_without_stats = (
        len(fights) - fights_with_stats
    )

    missing_control_rows = int(
        fight_totals["control_seconds"].isna().sum()
    )
    missing_control_fights = int(
        fight_totals.loc[
            fight_totals["control_seconds"].isna(),
            "fight_id",
        ].nunique()
    )

    print()
    print("Historical snapshots generated successfully.")
    print(f"Fights processed: {len(fights):,}")
    print(f"Completed fights with unknown duration: {missing_duration_count:,}")
    print(f"Snapshots generated: {len(snapshots):,}")
    print(
        "Distinct fighters: "
        f"{snapshots['fighter_id'].nunique():,}"
    )
    print(
        "Fights with statistics: "
        f"{fights_with_stats:,}"
    )
    print(
        "Fights without statistics: "
        f"{fights_without_stats:,}"
    )
    print(
        "Rows without control time: "
        f"{missing_control_rows:,}"
    )
    print(
        "Fights affected by missing control time: "
        f"{missing_control_fights:,}"
    )
    print(
        "Fighters with multiple fights in one event: "
        f"{repeated_event_fighters:,}"
    )
    print(f"Columns generated: {len(snapshots.columns):,}")
    print(f"Output file: {OUTPUT_PATH}")

    return snapshots


def main() -> None:
    build_fighter_snapshots()


if __name__ == "__main__":
    main()
