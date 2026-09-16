"""Build pre-fight features for upcoming UFC events using the same schema as the historical training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = "data/processed/ufcstats/upcoming_fight_dataset.parquet"

ELO_CONFIDENCE_PRIOR_FIGHTS = 3.0
EXPECTED_ELO_CONFIG_VERSION = "elo_v2_final_raw"



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


def is_blank(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("")


def safe_ratio(num: float | int, den: float | int) -> float:
    if den is None or pd.isna(den) or float(den) == 0.0:
        return np.nan
    return float(num) / float(den)


def safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return 0.0
    return float(numeric.sum(skipna=True))


def normalize_result(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().upper()
    aliases = {
        "WIN": "W",
        "LOSS": "L",
        "DRAW": "D",
        "NO CONTEST": "NC",
        "NO_CONTEST": "NC",
        "N/C": "NC",
    }
    return aliases.get(value, value)


def build_long_history(
    fights: pd.DataFrame,
    details: pd.DataFrame,
    totals: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "fight_id",
        "event_id",
        "event_name",
        "event_date",
        "fight_order",
        "weight_class",
    ]

    f1 = fights[
        base_cols
        + ["fighter_1_id", "fighter_1_name", "fighter_1_result",
           "fighter_2_id", "fighter_2_name"]
    ].copy()
    f1 = f1.rename(
        columns={
            "fighter_1_id": "fighter_id",
            "fighter_1_name": "fighter_name",
            "fighter_1_result": "fighter_result",
            "fighter_2_id": "opponent_id",
            "fighter_2_name": "opponent_name",
        }
    )

    f2 = fights[
        base_cols
        + ["fighter_2_id", "fighter_2_name", "fighter_2_result",
           "fighter_1_id", "fighter_1_name"]
    ].copy()
    f2 = f2.rename(
        columns={
            "fighter_2_id": "fighter_id",
            "fighter_2_name": "fighter_name",
            "fighter_2_result": "fighter_result",
            "fighter_1_id": "opponent_id",
            "fighter_1_name": "opponent_name",
        }
    )

    history = pd.concat([f1, f2], ignore_index=True)
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")

    detail_cols = [
        c
        for c in ["fight_id", "total_fight_time_seconds", "stats_available"]
        if c in details.columns
    ]
    history = history.merge(
        details[detail_cols].drop_duplicates("fight_id"),
        on="fight_id",
        how="left",
    )

    stat_cols = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_seconds",
    ]

    own_cols = ["fight_id", "fighter_id", "opponent_id"] + [
        c for c in stat_cols if c in totals.columns
    ]
    own = totals[own_cols].copy()
    own["_has_stats_row"] = True

    opp = totals[
        ["fight_id", "fighter_id"]
        + [c for c in stat_cols if c in totals.columns]
    ].copy()
    opp = opp.rename(
        columns={
            "fighter_id": "opponent_id",
            **{c: f"opponent_{c}" for c in stat_cols if c in opp.columns},
        }
    )

    own = own.merge(opp, on=["fight_id", "opponent_id"], how="left")
    history = history.merge(
        own,
        on=["fight_id", "fighter_id", "opponent_id"],
        how="left",
    )

    history["_has_stats_row"] = history["_has_stats_row"].fillna(False)

    # Exclude unresolved future fights accidentally present in historical data.
    history["_result_norm"] = history["fighter_result"].map(normalize_result)

    return history


def state_before(
    history: pd.DataFrame,
    fighter_id: str,
    event_date: pd.Timestamp,
) -> dict[str, Any]:
    h = history[
        (history["fighter_id"].astype("string") == str(fighter_id))
        & (history["event_date"] < event_date)
        & (history["_result_norm"] != "")
    ].copy()

    if not h.empty:
        h = h.sort_values(
            ["event_date", "fight_order"],
            ascending=[True, False],
            na_position="last",
        )

    results = h["_result_norm"].tolist()

    wins = sum(r == "W" for r in results)
    losses = sum(r == "L" for r in results)
    draws = sum(r == "D" for r in results)
    no_contests = sum(r == "NC" for r in results)
    fights_before = len(h)

    # Exclude no contests from the win-rate denominator.
    win_rate_den = wins + losses + draws
    win_rate = safe_ratio(wins, win_rate_den) if win_rate_den else np.nan

    win_streak = 0
    loss_streak = 0
    for result in results:
        if result == "W":
            win_streak += 1
            loss_streak = 0
        elif result == "L":
            loss_streak += 1
            win_streak = 0
        else:
            # Draws and no contests break streaks.
            win_streak = 0
            loss_streak = 0

    if h.empty:
        days_since_last_fight = np.nan
    else:
        last_date = h["event_date"].max()
        days_since_last_fight = int((event_date - last_date).days)

    total_fight_time = int(round(safe_sum(
        h.get("total_fight_time_seconds", pd.Series(dtype=float))
    )))

    with_stats = h[h["_has_stats_row"].astype(bool)].copy()
    fights_with_stats = len(with_stats)
    stats_time = int(round(safe_sum(
        with_stats.get("total_fight_time_seconds", pd.Series(dtype=float))
    )))

    stats_coverage = (
        safe_ratio(fights_with_stats, fights_before)
        if fights_before > 0
        else np.nan
    )

    if "control_seconds" in with_stats.columns:
        control_mask = pd.to_numeric(
            with_stats["control_seconds"], errors="coerce"
        ).notna()
    else:
        control_mask = pd.Series(False, index=with_stats.index)

    fights_with_control = int(control_mask.sum())
    control_coverage = (
        safe_ratio(fights_with_control, fights_before)
        if fights_before > 0
        else np.nan
    )

    sig_landed = int(round(safe_sum(
        with_stats.get("sig_str_landed", pd.Series(dtype=float))
    )))
    sig_attempted = int(round(safe_sum(
        with_stats.get("sig_str_attempted", pd.Series(dtype=float))
    )))
    sig_absorbed = int(round(safe_sum(
        with_stats.get("opponent_sig_str_landed", pd.Series(dtype=float))
    )))
    sig_absorbed_attempted = int(round(safe_sum(
        with_stats.get("opponent_sig_str_attempted", pd.Series(dtype=float))
    )))

    td_landed = int(round(safe_sum(
        with_stats.get("takedowns_landed", pd.Series(dtype=float))
    )))
    td_attempted = int(round(safe_sum(
        with_stats.get("takedowns_attempted", pd.Series(dtype=float))
    )))
    td_conceded = int(round(safe_sum(
        with_stats.get("opponent_takedowns_landed", pd.Series(dtype=float))
    )))
    td_faced = int(round(safe_sum(
        with_stats.get("opponent_takedowns_attempted", pd.Series(dtype=float))
    )))

    control_seconds = int(round(safe_sum(
        with_stats.loc[control_mask, "control_seconds"]
        if "control_seconds" in with_stats.columns
        else pd.Series(dtype=float)
    )))

    knockdowns = int(round(safe_sum(
        with_stats.get("knockdowns", pd.Series(dtype=float))
    )))
    knockdowns_absorbed = int(round(safe_sum(
        with_stats.get("opponent_knockdowns", pd.Series(dtype=float))
    )))
    submission_attempts = int(round(safe_sum(
        with_stats.get("submission_attempts", pd.Series(dtype=float))
    )))
    reversals = int(round(safe_sum(
        with_stats.get("reversals", pd.Series(dtype=float))
    )))

    minutes_with_stats = stats_time / 60.0 if stats_time > 0 else 0.0

    return {
        "fights_before": fights_before,
        "wins_before": wins,
        "losses_before": losses,
        "draws_before": draws,
        "no_contests_before": no_contests,
        "win_rate_before": win_rate,
        "current_win_streak_before": win_streak,
        "current_loss_streak_before": loss_streak,
        "days_since_last_fight": days_since_last_fight,
        "total_fight_time_seconds_before": total_fight_time,
        "fights_with_stats_before": fights_with_stats,
        "stats_fight_time_seconds_before": stats_time,
        "stats_coverage_before": stats_coverage,
        "fights_with_control_stats_before": fights_with_control,
        "control_stats_coverage_before": control_coverage,
        "sig_str_landed_before": sig_landed,
        "sig_str_attempted_before": sig_attempted,
        "sig_str_absorbed_before": sig_absorbed,
        "sig_str_absorbed_attempted_before": sig_absorbed_attempted,
        "sig_str_landed_per_min_before": (
            safe_ratio(sig_landed, minutes_with_stats)
            if minutes_with_stats > 0
            else np.nan
        ),
        "sig_str_absorbed_per_min_before": (
            safe_ratio(sig_absorbed, minutes_with_stats)
            if minutes_with_stats > 0
            else np.nan
        ),
        "sig_str_accuracy_before": (
            safe_ratio(sig_landed, sig_attempted)
            if sig_attempted > 0
            else np.nan
        ),
        "sig_str_defense_before": (
            1.0 - safe_ratio(sig_absorbed, sig_absorbed_attempted)
            if sig_absorbed_attempted > 0
            else np.nan
        ),
        "takedowns_landed_before": td_landed,
        "takedowns_attempted_before": td_attempted,
        "takedowns_conceded_before": td_conceded,
        "takedowns_faced_before": td_faced,
        "takedown_accuracy_before": (
            safe_ratio(td_landed, td_attempted)
            if td_attempted > 0
            else np.nan
        ),
        "takedown_defense_before": (
            1.0 - safe_ratio(td_conceded, td_faced)
            if td_faced > 0
            else np.nan
        ),
        "control_seconds_before": control_seconds,
        "knockdowns_before": knockdowns,
        "knockdowns_absorbed_before": knockdowns_absorbed,
        "submission_attempts_before": submission_attempts,
        "reversals_before": reversals,
    }


def profile_for(
    fighter_profiles: pd.DataFrame,
    fighter_id: str,
    event_date: pd.Timestamp,
) -> dict[str, Any]:
    rows = fighter_profiles[
        fighter_profiles["fighter_id"].astype("string") == str(fighter_id)
    ]
    if rows.empty:
        return {
            "height_cm": np.nan,
            "reach_cm": np.nan,
            "age_years": np.nan,
            "stance": pd.NA,
        }

    row = rows.iloc[0]
    dob = pd.to_datetime(row.get("date_of_birth"), errors="coerce")
    age = (
        (event_date - dob).days / 365.2425
        if pd.notna(dob) and pd.notna(event_date)
        else np.nan
    )

    return {
        "height_cm": pd.to_numeric(row.get("height_cm"), errors="coerce"),
        "reach_cm": pd.to_numeric(row.get("reach_cm"), errors="coerce"),
        "age_years": age,
        "stance": row.get("stance", pd.NA),
    }


def select_upcoming_fights(
    fights: pd.DataFrame,
    event_id: str | None,
    event_name: str | None,
    event_date: str | None,
    include_past_unresolved: bool,
) -> pd.DataFrame:
    upcoming = fights[
        is_blank(fights["winner_id"])
        & is_blank(fights["fighter_1_result"])
        & is_blank(fights["fighter_2_result"])
    ].copy()

    upcoming["event_date"] = pd.to_datetime(upcoming["event_date"], errors="coerce")

    if not include_past_unresolved:
        today = pd.Timestamp.today().normalize()
        upcoming = upcoming[upcoming["event_date"] >= today]

    if event_id:
        upcoming = upcoming[
            upcoming["event_id"].astype("string") == str(event_id)
        ]

    if event_name:
        upcoming = upcoming[
            upcoming["event_name"]
            .astype("string")
            .str.contains(event_name, case=False, na=False, regex=False)
        ]

    if event_date:
        wanted = pd.Timestamp(event_date).normalize()
        upcoming = upcoming[upcoming["event_date"].dt.normalize() == wanted]

    return upcoming.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, True],
        na_position="last",
    )


def elo_snapshot_for(
    elo_snapshots: pd.DataFrame,
    fight_id: str,
    fighter_id: str,
    opponent_id: str,
) -> dict[str, Any]:
    """Fetch one pre-fight Elo snapshot generated by build_elo_snapshots.py."""

    rows = elo_snapshots[
        (elo_snapshots["fight_id"].astype("string") == str(fight_id))
        & (elo_snapshots["fighter_id"].astype("string") == str(fighter_id))
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one Elo snapshot for fight={fight_id}, "
            f"fighter={fighter_id}; found {len(rows)}. "
            "Re-run build_elo_snapshots.py after updating UFCStats data."
        )

    row = rows.iloc[0]
    if str(row["opponent_id"]) != str(opponent_id):
        raise ValueError(
            f"Elo snapshot opponent mismatch for fight={fight_id}, "
            f"fighter={fighter_id}."
        )

    forbidden_legacy = [
        column
        for column in elo_snapshots.columns
        if "effective" in column or "decay_fraction" in column
    ]
    if forbidden_legacy:
        raise ValueError(
            "Legacy Effective/decay Elo columns are present. "
            "Re-run the final production build_elo_snapshots.py: "
            f"{forbidden_legacy}"
        )

    excluded = {
        "fight_id",
        "event_id",
        "event_name",
        "event_date",
        "fight_order",
        "weight_class",
        "fighter_id",
        "fighter_name",
        "opponent_id",
        "opponent_name",
    }
    return {
        column: row[column]
        for column in elo_snapshots.columns
        if column not in excluded
    }


def build_row(
    fight: pd.Series,
    history: pd.DataFrame,
    fighters: pd.DataFrame,
    elo_snapshots: pd.DataFrame,
) -> dict[str, Any]:
    event_date = pd.Timestamp(fight["event_date"])

    f1_id = str(fight["fighter_1_id"])
    f2_id = str(fight["fighter_2_id"])

    if f1_id <= f2_id:
        a_id, a_name = f1_id, fight["fighter_1_name"]
        b_id, b_name = f2_id, fight["fighter_2_name"]
    else:
        a_id, a_name = f2_id, fight["fighter_2_name"]
        b_id, b_name = f1_id, fight["fighter_1_name"]

    a_state = state_before(history, a_id, event_date)
    b_state = state_before(history, b_id, event_date)
    a_profile = profile_for(fighters, a_id, event_date)
    b_profile = profile_for(fighters, b_id, event_date)
    a_elo = elo_snapshot_for(
        elo_snapshots,
        str(fight["fight_id"]),
        a_id,
        b_id,
    )
    b_elo = elo_snapshot_for(
        elo_snapshots,
        str(fight["fight_id"]),
        b_id,
        a_id,
    )

    row: dict[str, Any] = {
        "fight_id": fight["fight_id"],
        "event_id": fight["event_id"],
        "event_name": fight["event_name"],
        "event_date": event_date,
        "fight_order": fight["fight_order"],
        "weight_class": fight["weight_class"],
        "fighter_a_id": a_id,
        "fighter_a_name": a_name,
        "fighter_b_id": b_id,
        "fighter_b_name": b_name,
    }

    for key, value in a_state.items():
        row[f"fighter_a_{key}"] = value
    for key, value in b_state.items():
        row[f"fighter_b_{key}"] = value

    for key, value in a_profile.items():
        row[f"fighter_a_{key}"] = value
    for key, value in b_profile.items():
        row[f"fighter_b_{key}"] = value

    for key, value in a_elo.items():
        row[f"fighter_a_{key}"] = value
    for key, value in b_elo.items():
        row[f"fighter_b_{key}"] = value

    a_stance = row.get("fighter_a_stance")
    b_stance = row.get("fighter_b_stance")

    if pd.notna(a_stance) and pd.notna(b_stance):
        row["stance_matchup"] = f"{a_stance}_vs_{b_stance}"
        row["same_stance"] = bool(str(a_stance) == str(b_stance))
    else:
        row["stance_matchup"] = pd.NA
        row["same_stance"] = pd.NA

    return row


def add_differences(row: dict[str, Any], numeric_features: list[str]) -> None:
    for col in numeric_features:
        if not col.startswith("diff_"):
            continue

        base = col[len("diff_") :]
        a_col = f"fighter_a_{base}"
        b_col = f"fighter_b_{base}"

        a = row.get(a_col, np.nan)
        b = row.get(b_col, np.nan)

        if pd.isna(a) or pd.isna(b):
            row[col] = np.nan
        else:
            row[col] = float(a) - float(b)


def validate_upcoming_output(
    result: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    """Validate the final prediction dataset before writing it."""

    if result.empty:
        raise ValueError("The upcoming dataset is unexpectedly empty.")

    if result["fight_id"].duplicated().any():
        raise ValueError("The upcoming dataset contains duplicate fight_id values.")

    invalid_ab_order = (
        result["fighter_a_id"].astype("string")
        > result["fighter_b_id"].astype("string")
    )
    if invalid_ab_order.any():
        raise ValueError(
            "A/B assignment is not deterministic: fighter_a_id must be <= fighter_b_id."
        )

    forbidden_post_fight = {
        "target",
        "winner_id",
        "fighter_1_result",
        "fighter_2_result",
        "method",
        "method_details",
        "finish_round",
        "finish_time",
    }
    leaked = forbidden_post_fight & set(result.columns)
    if leaked:
        raise ValueError(
            "The upcoming dataset contains outcome/post-fight columns: "
            f"{sorted(leaked)}"
        )

    legacy_elo = [
        column
        for column in result.columns
        if "elo_" in column
        and ("effective" in column or "decay_fraction" in column)
    ]
    if legacy_elo:
        raise ValueError(
            "Legacy Effective/decay Elo columns are present in the upcoming dataset: "
            f"{legacy_elo}"
        )

    for side in ("a", "b"):
        fights_before = pd.to_numeric(
            result[f"fighter_{side}_fights_before"], errors="coerce"
        )
        debutant = pd.to_numeric(
            result[f"fighter_{side}_is_debutant"], errors="coerce"
        )
        expected_debutant = fights_before.eq(0).astype(int)
        if not debutant.eq(expected_debutant).all():
            raise ValueError(
                f"fighter_{side}_is_debutant is inconsistent with fights_before."
            )

        expected_confidence = fights_before / (
            fights_before + ELO_CONFIDENCE_PRIOR_FIGHTS
        )
        actual_confidence = pd.to_numeric(
            result[f"fighter_{side}_elo_confidence"], errors="coerce"
        )
        if not np.allclose(
            actual_confidence.to_numpy(dtype=float),
            expected_confidence.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"fighter_{side}_elo_confidence is inconsistent with fights_before."
            )

        stats_days = pd.to_numeric(
            result[f"fighter_{side}_days_since_last_fight"], errors="coerce"
        )
        elo_days = pd.to_numeric(
            result[f"fighter_{side}_elo_days_inactive_before"], errors="coerce"
        )
        same_days = stats_days.eq(elo_days) | (stats_days.isna() & elo_days.isna())
        if not same_days.all():
            raise ValueError(
                f"fighter_{side} inactivity differs between historical and Elo state."
            )

        for rating_name in [
            "elo_general_raw_before",
            "elo_striking_raw_before",
            "elo_grappling_offense_raw_before",
            "elo_grappling_defense_raw_before",
        ]:
            values = pd.to_numeric(
                result[f"fighter_{side}_{rating_name}"], errors="coerce"
            )
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"fighter_{side}_{rating_name} contains missing/non-finite values."
                )

    # Recalculate every A-B difference required by the training manifest.
    for column in numeric_features:
        if not column.startswith("diff_"):
            continue
        base = column[len("diff_"):]
        a_col = f"fighter_a_{base}"
        b_col = f"fighter_b_{base}"
        if a_col not in result.columns or b_col not in result.columns:
            raise ValueError(
                f"Cannot validate {column}: source columns are missing."
            )
        expected = (
            pd.to_numeric(result[a_col], errors="coerce")
            - pd.to_numeric(result[b_col], errors="coerce")
        )
        actual = pd.to_numeric(result[column], errors="coerce")
        match = actual.eq(expected) | (actual.isna() & expected.isna())
        if not match.all():
            raise ValueError(f"{column} does not equal fighter_a - fighter_b.")


def validate_elo_report_version(path: Path) -> None:
    """Ensure upcoming predictions use snapshots from the final Elo configuration."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Elo report: {path}. Run build_elo_snapshots.py first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    version = report.get("config_version")
    if version != EXPECTED_ELO_CONFIG_VERSION:
        raise ValueError(
            "Elo snapshots use an obsolete configuration. "
            f"Expected {EXPECTED_ELO_CONFIG_VERSION!r}, found {version!r}. "
            "Re-run build_elo_snapshots.py."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the historical dataset's pre-fight features "
            "for unresolved UFCStats events."
        )
    )
    parser.add_argument("--event-id", default=None)
    parser.add_argument(
        "--event-name",
        default=None,
        help="Case-insensitive text filter for the event name.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Exact date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--include-past-unresolved",
        action="store_true",
        help="Include unresolved fights whose event date has already passed.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Parquet path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    root = find_project_root()
    interim = root / "data" / "interim" / "ufcstats"
    processed = root / "data" / "processed" / "ufcstats"

    fights_path = interim / "fights_clean.parquet"
    details_path = interim / "fight_details_clean.parquet"
    totals_path = interim / "fight_totals_clean.parquet"
    fighters_path = interim / "fighters_clean.parquet"
    elo_snapshots_path = processed / "fighter_elo_snapshots.parquet"
    elo_report_path = processed / "fighter_elo_snapshots_report.json"
    report_path = processed / "fight_dataset_report.json"

    required_paths = [
        fights_path,
        details_path,
        totals_path,
        fighters_path,
        elo_snapshots_path,
        elo_report_path,
        report_path,
    ]
    missing_paths = [p for p in required_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(f"  - {p}" for p in missing_paths)
        )

    print(f"Reading fights: {fights_path}")
    fights = pd.read_parquet(fights_path)
    print(f"Reading fight details: {details_path}")
    details = pd.read_parquet(details_path)
    print(f"Reading statistics: {totals_path}")
    totals = pd.read_parquet(totals_path)
    print(f"Reading fighter profiles: {fighters_path}")
    fighters = pd.read_parquet(fighters_path)
    print(f"Reading Elo snapshots: {elo_snapshots_path}")
    elo_snapshots = pd.read_parquet(elo_snapshots_path)

    validate_elo_report_version(elo_report_path)

    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    manifest = report["feature_manifest"]
    metadata_columns = list(manifest["metadata_columns"])
    numeric_features = list(manifest["numeric_model_features"])
    categorical_features = list(manifest["categorical_model_features"])

    upcoming = select_upcoming_fights(
        fights=fights,
        event_id=args.event_id,
        event_name=args.event_name,
        event_date=args.date,
        include_past_unresolved=args.include_past_unresolved,
    )

    if upcoming.empty:
        unresolved = fights[
            is_blank(fights["winner_id"])
            & is_blank(fights["fighter_1_result"])
            & is_blank(fights["fighter_2_result"])
        ][["event_id", "event_name", "event_date"]].drop_duplicates()

        msg = "No upcoming fights matched the selected filter."
        if not unresolved.empty:
            msg += "\n\nUnresolved events found:\n" + unresolved.to_string(index=False)
        raise ValueError(msg)

    print(f"Upcoming fights selected: {len(upcoming)}")
    print(
        upcoming[
            ["event_date", "event_name", "fighter_1_name", "fighter_2_name"]
        ].to_string(index=False)
    )

    history = build_long_history(fights, details, totals)

    rows: list[dict[str, Any]] = []
    for _, fight in upcoming.iterrows():
        row = build_row(fight, history, fighters, elo_snapshots)
        add_differences(row, numeric_features)
        rows.append(row)

    result = pd.DataFrame(rows)

    # Metadata and model-feature lists may overlap. Preserve the first occurrence
    # of each column because PyArrow rejects duplicate Parquet field names.
    manifest_columns = (
        metadata_columns
        + numeric_features
        + categorical_features
    )

    duplicate_manifest_columns = list(
        dict.fromkeys(
            column
            for column in manifest_columns
            if manifest_columns.count(column) > 1
        )
    )

    if duplicate_manifest_columns:
        print(
            "Duplicate manifest columns retained "
            "once: "
            + ", ".join(duplicate_manifest_columns)
        )

    expected_columns = list(
        dict.fromkeys(manifest_columns)
    )

    # Fail if the training manifest changes instead of creating silent defaults.
    missing_columns = [c for c in expected_columns if c not in result.columns]
    if missing_columns:
        raise ValueError(
            "Could not build every column required by the "
            "training manifest:\n  - "
            + "\n  - ".join(missing_columns)
        )

    result = result[expected_columns].copy()

    if result.columns.duplicated().any():
        duplicates = (
            result.columns[
                result.columns.duplicated(keep=False)
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "The upcoming-fight dataset still contains duplicate columns: "
            f"{duplicates}"
        )

    # Assign Parquet- and model-compatible dtypes.
    result["event_date"] = pd.to_datetime(result["event_date"], errors="coerce")
    for col in categorical_features:
        if col == "same_stance":
            result[col] = result[col].astype("boolean")
        else:
            result[col] = result[col].astype("string")

    for col in result.columns:
        if col.endswith("_is_debutant"):
            result[col] = result[col].astype("Int64")

    validate_upcoming_output(
        result=result,
        numeric_features=numeric_features,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    result.to_csv(csv_path, index=False)

    report_out = {
        "rows_written": int(len(result)),
        "columns_written": int(len(result.columns)),
        "events": (
            result[["event_id", "event_name", "event_date"]]
            .drop_duplicates()
            .assign(event_date=lambda x: x["event_date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        ),
        "side_assignment_rule": (
            "fighter_a=min(fighter_1_id,fighter_2_id); "
            "fighter_b=max(fighter_1_id,fighter_2_id)"
        ),
        "target_included": False,
        "anti_leakage_rule": (
            "Each feature uses only fights with event_date before "
            "the upcoming fight."
        ),
    }

    report_output = output_path.with_name(output_path.stem + "_report.json")
    with report_output.open("w", encoding="utf-8") as f:
        json.dump(report_out, f, ensure_ascii=False, indent=2)

    print("\nUpcoming-fight dataset generated successfully.")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_output}")


if __name__ == "__main__":
    main()
