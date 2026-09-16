"""Run end-to-end leakage and consistency checks for the UFC ML pipeline.

This script validates the processed historical snapshots, final supervised dataset,
Elo snapshots, feature manifest, and (when present) the upcoming-fight dataset.

It is intended to be run after rebuilding the production data pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ELO_CONFIG_VERSION = "elo_v2_final_raw"
FINAL_ELO_BASE = 1500.0
FINAL_ELO_SCALE = 300.0
FINAL_CONFIDENCE_PRIOR = 3.0
FINAL_K_SCHEDULE = [
    {"prior_fights_lt": 3, "k": 30.0},
    {"prior_fights_lt": 6, "k": 28.0},
    {"prior_fights_lt": 10, "k": 24.0},
    {"prior_fights_lt": None, "k": 20.0},
]

RAW_ELO_COLUMNS = [
    "elo_general_raw_before",
    "elo_striking_raw_before",
    "elo_grappling_offense_raw_before",
    "elo_grappling_defense_raw_before",
]

POST_FIGHT_COLUMNS = {
    "winner_id",
    "fighter_1_result",
    "fighter_2_result",
    "method",
    "method_details",
    "finish_round",
    "finish_time",
}


class SanityReport:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passed.append(message)
        else:
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    def finish(self, output_path: Path) -> None:
        payload = {
            "passed": self.passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "n_passed": len(self.passed),
            "n_warnings": len(self.warnings),
            "n_failures": len(self.failures),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 72)
        print("PIPELINE SANITY CHECK")
        print("=" * 72)
        print(f"PASS: {len(self.passed)}")
        print(f"WARN: {len(self.warnings)}")
        print(f"FAIL: {len(self.failures)}")

        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.failures:
            print(f"[FAIL] {message}")

        print(f"\nReport: {output_path}")

        if self.failures:
            raise SystemExit(1)

        print("\nAll mandatory sanity checks passed.")


def find_project_root() -> Path:
    starts = [Path.cwd(), Path(__file__).resolve().parent]
    for start in starts:
        for path in [start, *start.parents]:
            if (path / "src").exists() and (path / "data").exists():
                return path
    raise FileNotFoundError("Could not locate the UFC_PREDICTOR project root.")


def blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def normalize_result(value: object) -> str | None:
    if blank(value):
        return None
    text = str(value).strip().upper()
    aliases = {
        "WIN": "W",
        "LOSS": "L",
        "DRAW": "D",
        "NO CONTEST": "NC",
        "NO_CONTEST": "NC",
        "N/C": "NC",
    }
    return aliases.get(text, text)


def is_resolved_pair(r1: str | None, r2: str | None) -> bool:
    return (r1, r2) in {
        ("W", "L"),
        ("L", "W"),
        ("D", "D"),
        ("NC", "NC"),
    }


def is_decisive_pair(r1: str | None, r2: str | None) -> bool:
    return (r1, r2) in {("W", "L"), ("L", "W")}


def same_numeric(a: pd.Series, b: pd.Series, atol: float = 1e-12) -> bool:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(aa, bb, rtol=0.0, atol=atol, equal_nan=True))


def build_expected_prefight_state(fights: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct only information that is allowed to exist before each fight."""

    ordered = fights.copy()
    ordered["event_date"] = pd.to_datetime(ordered["event_date"], errors="raise")
    ordered = ordered.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, False],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    state: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for fight in ordered.itertuples(index=False):
        fight_id = str(fight.fight_id)
        event_date = pd.Timestamp(fight.event_date)
        f1 = str(fight.fighter_1_id)
        f2 = str(fight.fighter_2_id)
        r1 = normalize_result(fight.fighter_1_result)
        r2 = normalize_result(fight.fighter_2_result)
        resolved = is_resolved_pair(r1, r2)

        for fighter_id, result in ((f1, r1), (f2, r2)):
            s = state.setdefault(
                fighter_id,
                {
                    "fights": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "nc": 0,
                    "last_date": None,
                },
            )

            last_date = s["last_date"]
            days = (
                np.nan
                if last_date is None
                else int((event_date.normalize() - last_date.normalize()).days)
            )

            records.append(
                {
                    "fight_id": fight_id,
                    "fighter_id": fighter_id,
                    "expected_fights_before": s["fights"],
                    "expected_wins_before": s["wins"],
                    "expected_losses_before": s["losses"],
                    "expected_draws_before": s["draws"],
                    "expected_no_contests_before": s["nc"],
                    "expected_days_since_last_fight": days,
                    "expected_is_debutant": int(s["fights"] == 0),
                    "expected_elo_confidence": (
                        0.0
                        if s["fights"] == 0
                        else s["fights"] / (s["fights"] + FINAL_CONFIDENCE_PRIOR)
                    ),
                    "fight_is_resolved": resolved,
                }
            )

        if resolved:
            for fighter_id, result in ((f1, r1), (f2, r2)):
                s = state[fighter_id]
                s["fights"] += 1
                if result == "W":
                    s["wins"] += 1
                elif result == "L":
                    s["losses"] += 1
                elif result == "D":
                    s["draws"] += 1
                elif result == "NC":
                    s["nc"] += 1
                s["last_date"] = event_date

    expected = pd.DataFrame(records)
    if expected.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("Expected-state reconstruction produced duplicate keys.")
    return expected


def check_elo_report(report: SanityReport, path: Path) -> None:
    if not path.exists():
        report.failures.append(f"Missing Elo report: {path}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    report.check(data.get("config_version") == EXPECTED_ELO_CONFIG_VERSION,
                 "Elo report has the final production configuration fingerprint.")
    report.check(float(data.get("base_elo", np.nan)) == FINAL_ELO_BASE,
                 "Elo report uses BASE_ELO=1500.")
    report.check(float(data.get("elo_scale", np.nan)) == FINAL_ELO_SCALE,
                 "Elo report uses ELO_SCALE=300.")
    report.check(data.get("k_schedule") == FINAL_K_SCHEDULE,
                 "Elo report uses the final 30/28/24/20 K schedule.")
    report.check(data.get("confidence_formula") == "n / (n + 3)",
                 "Elo report uses confidence n/(n+3).")

    inactivity = data.get("inactivity", {})
    report.check(inactivity.get("elo_decay_enabled") is False,
                 "Elo report explicitly disables rating decay.")

    striking = data.get("striking", {})
    expected_striking = {
        "sig_str_share_weight": 0.75,
        "knockdown_share_weight": 0.25,
        "ko_min_winner_score": 0.90,
        "ko_base_multiplier": 1.05,
        "ko_max_opponent_quality_bonus": 0.05,
        "k_multiplier": 1.00,
    }
    report.check(
        all(float(striking.get(key, np.nan)) == value for key, value in expected_striking.items()),
        "Elo report uses the final Striking configuration.",
    )

    grappling = data.get("grappling", {})
    expected_grappling = {
        "td_weight": 0.30,
        "control_weight": 0.25,
        "submission_threat_weight": 0.30,
        "recovery_proxy_weight": 0.15,
        "reversal_penalty": 0.04,
        "submission_min_winner_score": 0.85,
        "submission_base_multiplier": 1.05,
        "submission_max_opponent_quality_bonus": 0.05,
        "k_multiplier": 0.75,
    }
    report.check(
        all(float(grappling.get(key, np.nan)) == value for key, value in expected_grappling.items()),
        "Elo report uses the final Grappling configuration.",
    )

    features = data.get("feature_columns", [])
    report.check(not any("effective" in c or "decay_fraction" in c for c in features),
                 "Elo report contains Raw-Elo features only.")


def validate_fighter_snapshots(
    report: SanityReport,
    snapshots: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    report.check(
        not snapshots.duplicated(["fight_id", "fighter_id"]).any(),
        "Historical fighter snapshots have unique fight_id + fighter_id keys.",
    )

    resolved_expected = expected.loc[expected["fight_is_resolved"]].copy()
    merged = snapshots.merge(
        resolved_expected,
        on=["fight_id", "fighter_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    report.check((merged["_merge"] == "both").all(),
                 "Historical snapshots exist exactly for every resolved fighter-fight row.")
    if not (merged["_merge"] == "both").all():
        return

    checks = {
        "fights_before": "expected_fights_before",
        "wins_before": "expected_wins_before",
        "losses_before": "expected_losses_before",
        "draws_before": "expected_draws_before",
        "no_contests_before": "expected_no_contests_before",
        "days_since_last_fight": "expected_days_since_last_fight",
    }
    for actual, exp in checks.items():
        report.check(
            same_numeric(merged[actual], merged[exp]),
            f"Historical {actual} uses only information from earlier resolved fights.",
        )

    record_sum = (
        pd.to_numeric(snapshots["wins_before"], errors="coerce")
        + pd.to_numeric(snapshots["losses_before"], errors="coerce")
        + pd.to_numeric(snapshots["draws_before"], errors="coerce")
        + pd.to_numeric(snapshots["no_contests_before"], errors="coerce")
    )
    report.check(
        same_numeric(record_sum, snapshots["fights_before"]),
        "Historical W/L/D/NC counts sum to fights_before.",
    )


def validate_elo_snapshots(
    report: SanityReport,
    elo: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    report.check(
        not elo.duplicated(["fight_id", "fighter_id"]).any(),
        "Elo snapshots have unique fight_id + fighter_id keys.",
    )
    report.check(
        len(elo) == len(expected),
        "Elo snapshots contain exactly two pre-fight rows per raw fight.",
    )

    forbidden = [c for c in elo.columns if "effective" in c or "decay_fraction" in c]
    report.check(not forbidden, "Elo snapshots contain no Effective/decay features.")

    merged = elo.merge(
        expected,
        on=["fight_id", "fighter_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    report.check((merged["_merge"] == "both").all(),
                 "Every raw fighter-fight row has exactly one Elo pre-fight snapshot.")
    if not (merged["_merge"] == "both").all():
        return

    report.check(
        same_numeric(merged["is_debutant"], merged["expected_is_debutant"]),
        "Elo debutant flags match the reconstructed pre-fight history.",
    )
    report.check(
        same_numeric(merged["elo_confidence"], merged["expected_elo_confidence"]),
        "General Elo confidence matches n/(n+3) from prior resolved fights.",
    )
    report.check(
        same_numeric(
            merged["elo_days_inactive_before"],
            merged["expected_days_since_last_fight"],
        ),
        "Elo inactivity days match the last resolved fight and contain no future information.",
    )

    debutants = pd.to_numeric(merged["is_debutant"], errors="coerce").eq(1)
    for column in RAW_ELO_COLUMNS:
        values = pd.to_numeric(merged[column], errors="coerce")
        report.check(
            values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all(),
            f"{column} contains only finite values.",
        )
        report.check(
            values.loc[debutants].eq(FINAL_ELO_BASE).all(),
            f"Debutants start at {FINAL_ELO_BASE:g} in {column}.",
        )


def validate_dataset(
    report: SanityReport,
    dataset: pd.DataFrame,
    fights: pd.DataFrame,
    dataset_report: dict[str, Any],
) -> None:
    normalized = fights.copy()
    normalized["r1"] = normalized["fighter_1_result"].map(normalize_result)
    normalized["r2"] = normalized["fighter_2_result"].map(normalize_result)
    decisive = normalized[
        [is_decisive_pair(r1, r2) for r1, r2 in zip(normalized["r1"], normalized["r2"])]
    ].copy()

    report.check(len(dataset) == len(decisive),
                 "Supervised dataset has one row per decisive W/L fight.")
    report.check(not dataset["fight_id"].duplicated().any(),
                 "Supervised dataset has unique fight_id values.")
    report.check(dataset["target"].isin([0, 1]).all() and dataset["target"].notna().all(),
                 "Target is complete and binary.")

    report.check(
        (dataset["fighter_a_id"].astype("string") <= dataset["fighter_b_id"].astype("string")).all(),
        "A/B assignment is deterministic and independent of the winner.",
    )

    forbidden = POST_FIGHT_COLUMNS & set(dataset.columns)
    report.check(not forbidden, "No post-fight outcome columns are present as dataset columns.")
    legacy = [c for c in dataset.columns if "elo_" in c and ("effective" in c or "decay_fraction" in c)]
    report.check(not legacy, "Final dataset contains Raw Elo only.")

    merged = dataset[["fight_id", "fighter_a_id", "fighter_b_id", "target"]].merge(
        decisive[["fight_id", "fighter_1_id", "fighter_2_id", "winner_id"]],
        on="fight_id",
        how="left",
        validate="one_to_one",
    )
    f1 = merged["fighter_1_id"].astype("string")
    f2 = merged["fighter_2_id"].astype("string")
    expected_a = pd.Series(np.where(f1 <= f2, f1, f2), index=merged.index, dtype="string")
    expected_b = pd.Series(np.where(f1 <= f2, f2, f1), index=merged.index, dtype="string")
    report.check((merged["fighter_a_id"].astype("string") == expected_a).all(),
                 "fighter_a is the lexicographically smaller fighter_id.")
    report.check((merged["fighter_b_id"].astype("string") == expected_b).all(),
                 "fighter_b is the lexicographically larger fighter_id.")
    expected_target = (
        merged["fighter_a_id"].astype("string") == merged["winner_id"].astype("string")
    ).astype(int)
    report.check(same_numeric(merged["target"], expected_target),
                 "Target matches the recorded winner after deterministic A/B assignment.")

    sorted_dataset = dataset.sort_values(
        ["event_date", "event_id", "fight_order"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    report.check(
        dataset[["fight_id"]].reset_index(drop=True).equals(
            sorted_dataset[["fight_id"]].reset_index(drop=True)
        ),
        "Supervised dataset is in stable chronological order.",
    )

    for side in ("a", "b"):
        fights_before = pd.to_numeric(dataset[f"fighter_{side}_fights_before"], errors="coerce")
        debutant = pd.to_numeric(dataset[f"fighter_{side}_is_debutant"], errors="coerce")
        report.check(
            debutant.eq(fights_before.eq(0).astype(int)).all(),
            f"fighter_{side} debutant flags agree with historical fights_before.",
        )
        expected_conf = fights_before / (fights_before + FINAL_CONFIDENCE_PRIOR)
        report.check(
            same_numeric(dataset[f"fighter_{side}_elo_confidence"], expected_conf),
            f"fighter_{side} General Elo confidence agrees with fights_before.",
        )
        report.check(
            same_numeric(
                dataset[f"fighter_{side}_days_since_last_fight"],
                dataset[f"fighter_{side}_elo_days_inactive_before"],
            ),
            f"fighter_{side} statistical and Elo inactivity features agree.",
        )

    manifest = dataset_report.get("feature_manifest", {})
    numeric = list(manifest.get("numeric_model_features", []))
    categorical = list(manifest.get("categorical_model_features", []))
    model_features = numeric + categorical

    report.check(all(c in dataset.columns for c in model_features),
                 "Every feature in the training manifest exists in the dataset.")
    report.check("target" not in model_features,
                 "Target is excluded from the model feature manifest.")
    report.check(not any(c in model_features for c in [
        "fight_id", "event_id", "event_name", "fighter_a_id", "fighter_b_id",
        "fighter_a_name", "fighter_b_name",
    ]), "Identifiers and fighter names are excluded from model features.")
    report.check(not any("effective" in c or "decay_fraction" in c for c in model_features),
                 "Feature manifest contains no legacy Effective/decay Elo features.")

    target_rate = float(pd.to_numeric(dataset["target"], errors="coerce").mean())
    report.warn(
        0.45 <= target_rate <= 0.55,
        f"Target balance is unusual after deterministic A/B assignment: {target_rate:.3f}.",
    )


def validate_upcoming(
    report: SanityReport,
    upcoming: pd.DataFrame,
    dataset_report: dict[str, Any],
) -> None:
    report.check(not upcoming["fight_id"].duplicated().any(),
                 "Upcoming dataset has unique fight_id values.")
    report.check("target" not in upcoming.columns,
                 "Upcoming dataset contains no target column.")
    report.check(not (POST_FIGHT_COLUMNS & set(upcoming.columns)),
                 "Upcoming dataset contains no post-fight outcome columns.")
    report.check(not any("effective" in c or "decay_fraction" in c for c in upcoming.columns),
                 "Upcoming dataset contains Raw Elo only.")
    report.check(
        (upcoming["fighter_a_id"].astype("string") <= upcoming["fighter_b_id"].astype("string")).all(),
        "Upcoming A/B assignment follows the same deterministic rule as training.",
    )

    manifest = dataset_report.get("feature_manifest", {})
    expected_columns = list(dict.fromkeys(
        list(manifest.get("metadata_columns", []))
        + list(manifest.get("numeric_model_features", []))
        + list(manifest.get("categorical_model_features", []))
    ))
    report.check(set(upcoming.columns) == set(expected_columns),
                 "Upcoming dataset matches the historical model-feature schema exactly.")

    for side in ("a", "b"):
        report.check(
            same_numeric(
                upcoming[f"fighter_{side}_days_since_last_fight"],
                upcoming[f"fighter_{side}_elo_days_inactive_before"],
            ),
            f"Upcoming fighter_{side} statistical and Elo inactivity agree.",
        )
        fights_before = pd.to_numeric(upcoming[f"fighter_{side}_fights_before"], errors="coerce")
        expected_conf = fights_before / (fights_before + FINAL_CONFIDENCE_PRIOR)
        report.check(
            same_numeric(upcoming[f"fighter_{side}_elo_confidence"], expected_conf),
            f"Upcoming fighter_{side} Elo confidence agrees with fights_before.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the final UFC ML data pipeline.")
    parser.add_argument(
        "--upcoming",
        default="data/processed/ufcstats/upcoming_fight_dataset.parquet",
        help="Upcoming dataset to validate when it exists.",
    )
    parser.add_argument(
        "--require-upcoming",
        action="store_true",
        help="Fail if the upcoming dataset does not exist.",
    )
    args = parser.parse_args()

    root = find_project_root()
    interim = root / "data" / "interim" / "ufcstats"
    processed = root / "data" / "processed" / "ufcstats"

    paths = {
        "fights": interim / "fights_clean.parquet",
        "fighter_snapshots": processed / "fighter_snapshots.parquet",
        "elo_snapshots": processed / "fighter_elo_snapshots.parquet",
        "dataset": processed / "fight_dataset.parquet",
        "dataset_report": processed / "fight_dataset_report.json",
        "elo_report": processed / "fighter_elo_snapshots_report.json",
    }

    report = SanityReport()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        report.failures.extend(f"Missing required file: {path}" for path in missing)
        report.finish(processed / "pipeline_sanity_report.json")
        return

    print("Reading production pipeline outputs...")
    fights = pd.read_parquet(paths["fights"])
    snapshots = pd.read_parquet(paths["fighter_snapshots"])
    elo = pd.read_parquet(paths["elo_snapshots"])
    dataset = pd.read_parquet(paths["dataset"])
    dataset_report = json.loads(paths["dataset_report"].read_text(encoding="utf-8"))

    expected = build_expected_prefight_state(fights)

    check_elo_report(report, paths["elo_report"])
    validate_fighter_snapshots(report, snapshots, expected)
    validate_elo_snapshots(report, elo, expected)
    validate_dataset(report, dataset, fights, dataset_report)

    upcoming_path = Path(args.upcoming)
    if not upcoming_path.is_absolute():
        upcoming_path = root / upcoming_path

    if upcoming_path.exists():
        upcoming = pd.read_parquet(upcoming_path)
        validate_upcoming(report, upcoming, dataset_report)
    elif args.require_upcoming:
        report.failures.append(f"Missing required upcoming dataset: {upcoming_path}")
    else:
        report.warnings.append(
            "Upcoming dataset was not found, so upcoming-schema checks were skipped."
        )

    report.finish(processed / "pipeline_sanity_report.json")


if __name__ == "__main__":
    main()
