"""Build leakage-safe pre-fight Elo snapshots for UFC fighters.

The module maintains four independent ratings:
- General Elo: overall ability to win UFC fights.
- Striking Elo: striking performance, with a conservative KO/TKO boost.
- Grappling offense Elo: ability to impose takedowns/control/submission threats.
- Grappling defense Elo: ability to prevent or escape an opponent's grappling.

Every snapshot is saved BEFORE the current fight updates any rating.
Unresolved future fights receive snapshots but never update the ratings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "ufcstats"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ufcstats"

FIGHTS_PATH = INTERIM_DIR / "fights_clean.parquet"
FIGHT_DETAILS_PATH = INTERIM_DIR / "fight_details_clean.parquet"
FIGHT_TOTALS_PATH = INTERIM_DIR / "fight_totals_clean.parquet"

OUTPUT_PATH = PROCESSED_DIR / "fighter_elo_snapshots.parquet"
REPORT_PATH = PROCESSED_DIR / "fighter_elo_snapshots_report.json"


# ---------------------------------------------------------------------------
# Elo configuration
# ---------------------------------------------------------------------------
ELO_CONFIG_VERSION = "elo_v2_final_raw"
BASE_ELO = 1500.0
ELO_SCALE = 300.0

# Conservative K schedule based on UFC fights completed before the current bout.
K_SCHEDULE = (
    (3, 30.0),   # 0-2 prior UFC fights
    (6, 28.0),   # 3-5
    (10, 24.0),  # 6-9
    (math.inf, 20.0),  # 10+
)

# Confidence in the rating grows with the amount of UFC evidence available.
CONFIDENCE_PRIOR_FIGHTS = 3.0

# Final model decision: inactivity remains a separate feature and does not
# shrink the Elo ratings. Expectations are always computed from Raw Elo.

# Domain-specific update speeds.
STRIKING_K_MULTIPLIER = 1.00
GRAPPLING_K_MULTIPLIER = 0.75

# Striking performance.
STRIKING_SIG_SHARE_WEIGHT = 0.75
STRIKING_KD_SHARE_WEIGHT = 0.25
KO_MIN_WINNER_SCORE = 0.90
KO_BASE_MULTIPLIER = 1.05
KO_MAX_QUALITY_BONUS = 0.05

# Grappling performance (weights are renormalized over available components).
GRAPPLE_TD_WEIGHT = 0.30
GRAPPLE_CONTROL_WEIGHT = 0.25
GRAPPLE_SUB_WEIGHT = 0.30
GRAPPLE_RECOVERY_WEIGHT = 0.15
GRAPPLE_REVERSAL_PENALTY = 0.04
RECOVERY_SCALE_SECONDS = 60.0
SUBMISSION_THREAT_SCALE = 1.0
SUB_MIN_WINNER_SCORE = 0.85
SUB_BASE_MULTIPLIER = 1.05
SUB_MAX_QUALITY_BONUS = 0.05


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


@dataclass
class FighterEloState:
    """Persistent pre-/post-fight Elo state for one fighter."""

    general: float = BASE_ELO
    striking: float = BASE_ELO
    grappling_offense: float = BASE_ELO
    grappling_defense: float = BASE_ELO

    ufc_fights: int = 0
    striking_updates: int = 0
    grappling_offense_updates: int = 0
    grappling_defense_updates: int = 0

    last_fight_date: pd.Timestamp | None = None


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {dataframe_name}: {sorted(missing)}"
        )


def normalize_optional_string(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None


def normalize_result(value: object) -> str | None:
    value = normalize_optional_string(value)
    if value is None:
        return None

    normalized = value.upper()
    aliases = {
        "W": "W",
        "WIN": "W",
        "L": "L",
        "LOSS": "L",
        "D": "D",
        "DRAW": "D",
        "NC": "NC",
        "N/C": "NC",
        "NO CONTEST": "NC",
        "NO_CONTEST": "NC",
    }
    return aliases.get(normalized, normalized)


def normalize_method(value: object) -> str:
    value = normalize_optional_string(value)
    return "" if value is None else value.upper()


def is_ko_tko(method: str) -> bool:
    text = normalize_method(method)
    return "KO" in text or "TKO" in text


def is_submission(method: str) -> bool:
    text = normalize_method(method)
    return text == "SUB" or "SUBMISSION" in text or text.startswith("SUB ")


def k_factor(ufc_fights_before: int) -> float:
    for upper_bound, value in K_SCHEDULE:
        if ufc_fights_before < upper_bound:
            return value
    raise RuntimeError("Invalid K schedule.")


def elo_confidence(observations: int) -> float:
    """Conservative confidence in [0, 1), based on informative observations."""

    if observations <= 0:
        return 0.0
    return float(observations / (observations + CONFIDENCE_PRIOR_FIGHTS))


def expected_score(rating_a: float, rating_b: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE)))


def inactivity_days(
    state: FighterEloState,
    event_date: pd.Timestamp,
) -> int | None:
    if state.ufc_fights == 0 or state.last_fight_date is None:
        return None
    days = int((event_date.normalize() - state.last_fight_date.normalize()).days)
    if days < 0:
        raise ValueError(f"Negative Elo inactivity interval: {days} days.")
    return days


def opponent_quality(rating: float) -> float:
    """Quality bonus in [0,1], only for opponents above the baseline."""

    return float(np.clip((rating - BASE_ELO) / ELO_SCALE, 0.0, 1.0))


def weighted_average(parts: list[tuple[float, float]]) -> float | None:
    if not parts:
        return None
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return None
    return float(sum(value * weight for value, weight in parts) / total_weight)


def numeric(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def striking_scores(
    stats_a: Any | None,
    stats_b: Any | None,
) -> tuple[float | None, float | None]:
    """Return complementary striking-performance scores for A and B."""

    parts_a: list[tuple[float, float]] = []
    parts_b: list[tuple[float, float]] = []

    if stats_a is not None and stats_b is not None:
        sig_a = numeric(stats_a.sig_str_landed)
        sig_b = numeric(stats_b.sig_str_landed)
        total_sig = sig_a + sig_b
        if total_sig > 0:
            share_a = sig_a / total_sig
            parts_a.append((share_a, STRIKING_SIG_SHARE_WEIGHT))
            parts_b.append((1.0 - share_a, STRIKING_SIG_SHARE_WEIGHT))

        kd_a = numeric(stats_a.knockdowns)
        kd_b = numeric(stats_b.knockdowns)
        total_kd = kd_a + kd_b
        if total_kd > 0:
            share_a = kd_a / total_kd
            parts_a.append((share_a, STRIKING_KD_SHARE_WEIGHT))
            parts_b.append((1.0 - share_a, STRIKING_KD_SHARE_WEIGHT))

    score_a = weighted_average(parts_a)
    score_b = weighted_average(parts_b)
    return score_a, score_b


def grappling_offense_score(
    own_stats: Any | None,
    opponent_stats: Any | None,
    fight_time_seconds: int,
) -> float | None:
    """Score one fighter's offense against the opponent's defense.

    Defensive information is encoded naturally as the complement of this score:
    failed takedowns, little control after a takedown, and opponent reversals all
    improve the opponent's grappling-defense result.
    """

    if own_stats is None or opponent_stats is None:
        return None

    td_landed = numeric(own_stats.takedowns_landed)
    td_attempted = numeric(own_stats.takedowns_attempted)
    control_value = pd.to_numeric(own_stats.control_seconds, errors="coerce")
    control_known = pd.notna(control_value)
    control_seconds = 0.0 if not control_known else float(control_value)
    sub_attempts = numeric(own_stats.submission_attempts)
    opponent_reversals = numeric(opponent_stats.reversals)

    evidence = (
        td_attempted > 0
        or control_seconds > 0
        or sub_attempts > 0
        or opponent_reversals > 0
    )
    if not evidence:
        return None

    parts: list[tuple[float, float]] = []

    if td_attempted > 0:
        parts.append((td_landed / td_attempted, GRAPPLE_TD_WEIGHT))

    if control_known:
        # Controlling roughly half or more of the fight saturates this component.
        denominator = max(float(fight_time_seconds) * 0.5, 60.0)
        control_score = float(np.clip(control_seconds / denominator, 0.0, 1.0))
        parts.append((control_score, GRAPPLE_CONTROL_WEIGHT))

    # A submission attempt is meaningful but should saturate quickly.
    sub_score = 1.0 - math.exp(-sub_attempts / SUBMISSION_THREAT_SCALE)
    parts.append((sub_score, GRAPPLE_SUB_WEIGHT))

    # Proxy for the defender's ability to get back up after conceding a takedown.
    # Short control-per-takedown => good recovery => lower offense score.
    if td_landed > 0 and control_known:
        control_per_td = control_seconds / td_landed
        retention_score = 1.0 - math.exp(
            -control_per_td / RECOVERY_SCALE_SECONDS
        )
        parts.append((retention_score, GRAPPLE_RECOVERY_WEIGHT))

    score = weighted_average(parts)
    if score is None:
        return None

    # Reversals are explicit evidence that the defender escaped/flipped position.
    reversal_penalty = GRAPPLE_REVERSAL_PENALTY * (
        1.0 - math.exp(-opponent_reversals)
    )

    return float(np.clip(score - reversal_penalty, 0.0, 1.0))


def create_snapshot(
    *,
    fight: Any,
    fighter_id: str,
    fighter_name: str,
    opponent_id: str,
    opponent_name: str,
    state: FighterEloState,
) -> dict[str, object]:
    event_date = pd.Timestamp(fight.event_date)
    days_inactive = inactivity_days(state, event_date)

    return {
        "fight_id": str(fight.fight_id),
        "event_id": str(fight.event_id),
        "event_name": str(fight.event_name),
        "event_date": event_date,
        "fight_order": fight.fight_order,
        "weight_class": fight.weight_class,
        "fighter_id": fighter_id,
        "fighter_name": fighter_name,
        "opponent_id": opponent_id,
        "opponent_name": opponent_name,
        "is_debutant": int(state.ufc_fights == 0),
        "elo_confidence": elo_confidence(state.ufc_fights),
        "elo_striking_confidence": elo_confidence(state.striking_updates),
        "elo_grappling_offense_confidence": elo_confidence(
            state.grappling_offense_updates
        ),
        "elo_grappling_defense_confidence": elo_confidence(
            state.grappling_defense_updates
        ),
        "elo_days_inactive_before": (
            pd.NA if days_inactive is None else days_inactive
        ),
        "elo_general_raw_before": state.general,
        "elo_striking_raw_before": state.striking,
        "elo_grappling_offense_raw_before": state.grappling_offense,
        "elo_grappling_defense_raw_before": state.grappling_defense,
    }


def finish_multiplier(
    opponent_rating: float,
    *,
    base_multiplier: float,
    max_quality_bonus: float,
) -> float:
    return float(
        base_multiplier
        + max_quality_bonus * opponent_quality(opponent_rating)
    )


def general_actual_scores(result_a: str, result_b: str) -> tuple[float, float]:
    pair = (result_a, result_b)
    if pair == ("W", "L"):
        return 1.0, 0.0
    if pair == ("L", "W"):
        return 0.0, 1.0
    if pair == ("D", "D"):
        return 0.5, 0.5
    raise ValueError(f"Unsupported decisive/draw result pair: {pair}")


def build_method_column(
    fights: pd.DataFrame,
    details: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a best-effort finish-method column without assuming its source."""

    result = fights.copy()
    if "method" in result.columns:
        result["elo_method"] = result["method"]
        return result

    if "method" in details.columns:
        method_rows = details[["fight_id", "method"]].drop_duplicates("fight_id")
        result = result.merge(
            method_rows,
            on="fight_id",
            how="left",
            validate="one_to_one",
        )
        result = result.rename(columns={"method": "elo_method"})
        return result

    result["elo_method"] = ""
    return result


def build_elo_snapshots() -> pd.DataFrame:
    print(f"Reading fights: {FIGHTS_PATH}")
    fights = pd.read_parquet(FIGHTS_PATH)
    print(f"Reading fight details: {FIGHT_DETAILS_PATH}")
    details = pd.read_parquet(FIGHT_DETAILS_PATH)
    print(f"Reading fight totals: {FIGHT_TOTALS_PATH}")
    totals = pd.read_parquet(FIGHT_TOTALS_PATH)

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
        details,
        {"fight_id", "total_fight_time_seconds", "stats_available"},
        "fight_details_clean.parquet",
    )
    require_columns(
        totals,
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

    detail_cols = ["fight_id", "total_fight_time_seconds", "stats_available"]
    fights = fights.merge(
        details[detail_cols].drop_duplicates("fight_id"),
        on="fight_id",
        how="left",
        validate="one_to_one",
    )
    fights = build_method_column(fights, details)
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="raise")

    duplicate_totals = totals.duplicated(["fight_id", "fighter_id"], keep=False)
    if duplicate_totals.any():
        examples = totals.loc[
            duplicate_totals, ["fight_id", "fighter_id"]
        ].head(10)
        raise ValueError(
            "Duplicate fight_totals rows for fight_id + fighter_id:\n"
            f"{examples.to_string(index=False)}"
        )

    # Same chronological convention as build_fighter_snapshots.py.
    fights = fights.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    stats_lookup = {
        (str(row.fight_id), str(row.fighter_id)): row
        for row in totals.itertuples(index=False)
    }

    states: dict[str, FighterEloState] = {}
    records: list[dict[str, object]] = []

    resolved_fights = 0
    unresolved_fights = 0
    ko_tko_fights = 0
    submission_fights = 0
    striking_updates = 0
    grappling_directional_updates = 0

    for fight in fights.itertuples(index=False):
        fight_id = str(fight.fight_id)
        event_date = pd.Timestamp(fight.event_date)
        f1_id = str(fight.fighter_1_id)
        f2_id = str(fight.fighter_2_id)

        if f1_id == f2_id:
            raise ValueError(f"Fight {fight_id} contains the same fighter twice.")

        s1 = states.setdefault(f1_id, FighterEloState())
        s2 = states.setdefault(f2_id, FighterEloState())

        records.append(
            create_snapshot(
                fight=fight,
                fighter_id=f1_id,
                fighter_name=str(fight.fighter_1_name),
                opponent_id=f2_id,
                opponent_name=str(fight.fighter_2_name),
                state=s1,
            )
        )
        records.append(
            create_snapshot(
                fight=fight,
                fighter_id=f2_id,
                fighter_name=str(fight.fighter_2_name),
                opponent_id=f1_id,
                opponent_name=str(fight.fighter_1_name),
                state=s2,
            )
        )

        r1 = normalize_result(fight.fighter_1_result)
        r2 = normalize_result(fight.fighter_2_result)
        winner_id = normalize_optional_string(fight.winner_id)

        # Fully unresolved scheduled fights get a pre-fight snapshot only.
        if r1 is None and r2 is None and winner_id is None:
            unresolved_fights += 1
            continue

        if r1 is None or r2 is None:
            raise ValueError(
                f"Partially resolved result in {fight_id}: {r1!r}, {r2!r}."
            )

        valid_pairs = {("W", "L"), ("L", "W"), ("D", "D"), ("NC", "NC")}
        if (r1, r2) not in valid_pairs:
            raise ValueError(f"Invalid result pair in {fight_id}: {(r1, r2)}")

        if r1 == "W" and winner_id != f1_id:
            raise ValueError(f"winner_id mismatch in {fight_id}.")
        if r2 == "W" and winner_id != f2_id:
            raise ValueError(f"winner_id mismatch in {fight_id}.")
        if r1 in {"D", "NC"} and winner_id is not None:
            raise ValueError(f"Fight {fight_id} should not have winner_id.")

        resolved_fights += 1
        method = normalize_method(fight.elo_method)
        ko_finish = is_ko_tko(method)
        sub_finish = is_submission(method)
        ko_tko_fights += int(ko_finish)
        submission_fights += int(sub_finish)

        # Final production design: expectations use Raw Elo directly.
        raw1_general, raw2_general = s1.general, s2.general
        raw1_striking, raw2_striking = s1.striking, s2.striking
        raw1_goff, raw2_goff = s1.grappling_offense, s2.grappling_offense
        raw1_gdef, raw2_gdef = s1.grappling_defense, s2.grappling_defense

        k1 = k_factor(s1.ufc_fights)
        k2 = k_factor(s2.ufc_fights)
        striking_k1 = k1 * STRIKING_K_MULTIPLIER
        striking_k2 = k2 * STRIKING_K_MULTIPLIER
        grappling_k1 = k1 * GRAPPLING_K_MULTIPLIER
        grappling_k2 = k2 * GRAPPLING_K_MULTIPLIER

        # --------------------------- General Elo ---------------------------
        if r1 != "NC":
            actual1, actual2 = general_actual_scores(r1, r2)
            expected1 = expected_score(raw1_general, raw2_general)
            expected2 = 1.0 - expected1
            s1.general += k1 * (actual1 - expected1)
            s2.general += k2 * (actual2 - expected2)

        stats1 = stats_lookup.get((fight_id, f1_id))
        stats2 = stats_lookup.get((fight_id, f2_id))
        stats_available = bool(fight.stats_available)
        if stats_available and (stats1 is None or stats2 is None):
            raise ValueError(
                f"stats_available=True but fight_totals rows are missing in {fight_id}."
            )

        # Domain ratings are not updated for no contests.
        if r1 != "NC":
            # ------------------------- Striking Elo ------------------------
            strike1, strike2 = striking_scores(
                stats1 if stats_available else None,
                stats2 if stats_available else None,
            )

            if ko_finish and winner_id in {f1_id, f2_id}:
                if winner_id == f1_id:
                    strike1 = max(KO_MIN_WINNER_SCORE, strike1 or 0.0)
                    strike2 = 1.0 - strike1
                    multiplier = finish_multiplier(
                        raw2_striking,
                        base_multiplier=KO_BASE_MULTIPLIER,
                        max_quality_bonus=KO_MAX_QUALITY_BONUS,
                    )
                else:
                    strike2 = max(KO_MIN_WINNER_SCORE, strike2 or 0.0)
                    strike1 = 1.0 - strike2
                    multiplier = finish_multiplier(
                        raw1_striking,
                        base_multiplier=KO_BASE_MULTIPLIER,
                        max_quality_bonus=KO_MAX_QUALITY_BONUS,
                    )
            else:
                multiplier = 1.0

            if strike1 is not None and strike2 is not None:
                exp1 = expected_score(raw1_striking, raw2_striking)
                exp2 = 1.0 - exp1
                old1, old2 = s1.striking, s2.striking
                s1.striking = old1 + striking_k1 * multiplier * (strike1 - exp1)
                s2.striking = old2 + striking_k2 * multiplier * (strike2 - exp2)
                s1.striking_updates += 1
                s2.striking_updates += 1
                striking_updates += 1

            # ---------------------- Grappling Elo -------------------------
            fight_time_seconds = int(fight.total_fight_time_seconds)

            score_1_offense = grappling_offense_score(
                stats1 if stats_available else None,
                stats2 if stats_available else None,
                fight_time_seconds,
            )
            score_2_offense = grappling_offense_score(
                stats2 if stats_available else None,
                stats1 if stats_available else None,
                fight_time_seconds,
            )

            mult_1 = 1.0
            mult_2 = 1.0

            if sub_finish and winner_id == f1_id:
                score_1_offense = max(
                    SUB_MIN_WINNER_SCORE,
                    score_1_offense or 0.0,
                )
                mult_1 = finish_multiplier(
                    raw2_gdef,
                    base_multiplier=SUB_BASE_MULTIPLIER,
                    max_quality_bonus=SUB_MAX_QUALITY_BONUS,
                )
            elif sub_finish and winner_id == f2_id:
                score_2_offense = max(
                    SUB_MIN_WINNER_SCORE,
                    score_2_offense or 0.0,
                )
                mult_2 = finish_multiplier(
                    raw1_gdef,
                    base_multiplier=SUB_BASE_MULTIPLIER,
                    max_quality_bonus=SUB_MAX_QUALITY_BONUS,
                )

            # Compute both directional deltas from the same pre-fight ratings.
            old1_off, old1_def = s1.grappling_offense, s1.grappling_defense
            old2_off, old2_def = s2.grappling_offense, s2.grappling_defense

            delta_1_off = delta_2_def = 0.0
            delta_2_off = delta_1_def = 0.0

            if score_1_offense is not None:
                exp_1_off = expected_score(raw1_goff, raw2_gdef)
                delta_1_off = grappling_k1 * mult_1 * (score_1_offense - exp_1_off)
                delta_2_def = grappling_k2 * mult_1 * (
                    (1.0 - score_1_offense) - (1.0 - exp_1_off)
                )
                s1.grappling_offense_updates += 1
                s2.grappling_defense_updates += 1
                grappling_directional_updates += 1

            if score_2_offense is not None:
                exp_2_off = expected_score(raw2_goff, raw1_gdef)
                delta_2_off = grappling_k2 * mult_2 * (score_2_offense - exp_2_off)
                delta_1_def = grappling_k1 * mult_2 * (
                    (1.0 - score_2_offense) - (1.0 - exp_2_off)
                )
                s2.grappling_offense_updates += 1
                s1.grappling_defense_updates += 1
                grappling_directional_updates += 1

            s1.grappling_offense = old1_off + delta_1_off
            s1.grappling_defense = old1_def + delta_1_def
            s2.grappling_offense = old2_off + delta_2_off
            s2.grappling_defense = old2_def + delta_2_def

        # All resolved UFC fights, including NC, count as experience and activity.
        s1.ufc_fights += 1
        s2.ufc_fights += 1
        s1.last_fight_date = event_date
        s2.last_fight_date = event_date

    snapshots = pd.DataFrame(records)

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
        "is_debutant",
        "elo_days_inactive_before",
    ]
    float_columns = [c for c in ELO_FEATURE_COLUMNS if c not in {
        "is_debutant", "elo_days_inactive_before"
    }]

    for column in string_columns:
        snapshots[column] = snapshots[column].astype("string")
    for column in integer_columns:
        snapshots[column] = snapshots[column].astype("Int64")
    for column in float_columns:
        snapshots[column] = snapshots[column].astype("Float64")
    snapshots["event_date"] = pd.to_datetime(snapshots["event_date"])

    expected_rows = len(fights) * 2
    if len(snapshots) != expected_rows:
        raise ValueError(
            f"Generated {len(snapshots)} Elo snapshots; expected {expected_rows}."
        )
    if snapshots.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("Duplicate fight_id + fighter_id Elo snapshots were generated.")
    if not snapshots["is_debutant"].isin([0, 1]).all():
        raise ValueError("is_debutant contains values other than 0/1.")

    for column in [
        "elo_confidence",
        "elo_striking_confidence",
        "elo_grappling_offense_confidence",
        "elo_grappling_defense_confidence",
    ]:
        values = snapshots[column].dropna()
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{column} contains values outside [0,1].")

    # Debutants must be untouched by prior UFC information.
    debutants = snapshots["is_debutant"].eq(1)
    if snapshots.loc[debutants, "elo_days_inactive_before"].notna().any():
        raise ValueError("At least one Elo debutant has prior inactivity days.")

    for rating_column in [
        "elo_general_raw_before",
        "elo_striking_raw_before",
        "elo_grappling_offense_raw_before",
        "elo_grappling_defense_raw_before",
    ]:
        values = pd.to_numeric(snapshots[rating_column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.astype(float)).all():
            raise ValueError(f"{rating_column} contains missing/non-finite values.")
        if snapshots.loc[debutants, rating_column].ne(BASE_ELO).any():
            raise ValueError(
                f"Debutants do not start at BASE_ELO in {rating_column}."
            )

    if snapshots.loc[debutants, "elo_confidence"].ne(0.0).any():
        raise ValueError("At least one debutant has non-zero General Elo confidence.")

    inactivity = snapshots["elo_days_inactive_before"].dropna()
    if (inactivity < 0).any():
        raise ValueError("Negative Elo inactivity days were generated.")

    forbidden_legacy_columns = [
        column for column in snapshots.columns
        if "effective" in column or "decay_fraction" in column
    ]
    if forbidden_legacy_columns:
        raise ValueError(
            "Legacy Effective/decay Elo columns are still present: "
            f"{forbidden_legacy_columns}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(OUTPUT_PATH, index=False)

    report = {
        "config_version": ELO_CONFIG_VERSION,
        "rows_written": int(len(snapshots)),
        "fights_seen": int(len(fights)),
        "resolved_fights": int(resolved_fights),
        "unresolved_fights_snapshot_only": int(unresolved_fights),
        "ko_tko_fights": int(ko_tko_fights),
        "submission_fights": int(submission_fights),
        "striking_fight_updates": int(striking_updates),
        "grappling_directional_updates": int(grappling_directional_updates),
        "base_elo": BASE_ELO,
        "elo_scale": ELO_SCALE,
        "k_schedule": [
            {"prior_fights_lt": None if math.isinf(limit) else int(limit), "k": k}
            for limit, k in K_SCHEDULE
        ],
        "confidence_formula": f"n / (n + {CONFIDENCE_PRIOR_FIGHTS:g})",
        "inactivity": {
            "elo_decay_enabled": False,
            "feature": "elo_days_inactive_before",
            "note": (
                "Inactivity is kept as an independent model feature and does "
                "not shrink the Elo ratings."
            ),
        },
        "striking": {
            "sig_str_share_weight": STRIKING_SIG_SHARE_WEIGHT,
            "knockdown_share_weight": STRIKING_KD_SHARE_WEIGHT,
            "ko_min_winner_score": KO_MIN_WINNER_SCORE,
            "ko_base_multiplier": KO_BASE_MULTIPLIER,
            "ko_max_opponent_quality_bonus": KO_MAX_QUALITY_BONUS,
            "k_multiplier": STRIKING_K_MULTIPLIER,
        },
        "grappling": {
            "offense_vs_defense": True,
            "td_weight": GRAPPLE_TD_WEIGHT,
            "control_weight": GRAPPLE_CONTROL_WEIGHT,
            "submission_threat_weight": GRAPPLE_SUB_WEIGHT,
            "recovery_proxy_weight": GRAPPLE_RECOVERY_WEIGHT,
            "reversal_penalty": GRAPPLE_REVERSAL_PENALTY,
            "recovery_proxy": "opponent control seconds per takedown conceded",
            "submission_min_winner_score": SUB_MIN_WINNER_SCORE,
            "submission_base_multiplier": SUB_BASE_MULTIPLIER,
            "submission_max_opponent_quality_bonus": SUB_MAX_QUALITY_BONUS,
            "k_multiplier": GRAPPLING_K_MULTIPLIER,
        },
        "feature_columns": ELO_FEATURE_COLUMNS,
        "notes": [
            "Every Elo snapshot is taken before the current fight is processed.",
            "Unresolved future fights receive snapshots but do not update ratings.",
            "Opponent strength enters every Elo update through expected score.",
            "KO/TKO and submission finishes get only a conservative extra boost.",
            "Raw Elo is used directly for expectations and model features.",
            "Effective Elo and Elo decay were removed after temporal validation.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nElo snapshots generated successfully.")
    print(f"Fights seen: {len(fights):,}")
    print(f"Resolved fights: {resolved_fights:,}")
    print(f"Unresolved snapshot-only fights: {unresolved_fights:,}")
    print(f"Snapshots: {len(snapshots):,}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")

    return snapshots


def main() -> None:
    build_elo_snapshots()


if __name__ == "__main__":
    main()
