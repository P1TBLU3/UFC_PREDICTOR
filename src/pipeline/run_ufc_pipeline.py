"""Run the complete UFC prediction pipeline from scraping to upcoming predictions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    extra_args: tuple[str, ...] = ()


def find_project_root() -> Path:
    starts = [
        Path.cwd(),
        Path(sys.executable).resolve().parent,
    ]

    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)

    for start in starts:
        for path in [start, *start.parents]:
            if (path / "src").exists() and (path / "data").exists():
                return path

    raise FileNotFoundError(
        "Could not locate the UFC_PREDICTOR project root.\n"
        "Place the executable in the project root or run it from the project root."
    )


def find_python(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / "venv" / "Scripts" / "python.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # When running this file directly inside the activated virtual environment.
    if not getattr(sys, "frozen", False):
        return Path(sys.executable)

    raise FileNotFoundError(
        "Could not find the project virtual environment.\n"
        "Expected `.venv\\Scripts\\python.exe` or `venv\\Scripts\\python.exe`."
    )


def run_step(
    python_executable: Path,
    project_root: Path,
    step: Step,
    forwarded_args: list[str] | None = None,
) -> None:
    command = [
        str(python_executable),
        "-m",
        step.module,
        *step.extra_args,
    ]

    if forwarded_args:
        command.extend(forwarded_args)

    print()
    print("=" * 100)
    print(step.name)
    print("=" * 100)
    print("Command:", " ".join(command))
    print()

    started = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
    )

    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped at '{step.name}'. "
            f"Exit code: {completed.returncode}"
        )

    print()
    print(f"Completed in {elapsed:.1f} s.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete UFC predictor pipeline: scrape UFCStats, clean data, "
            "build frozen features/Elo, create upcoming fights and classify them."
        )
    )

    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Use the existing raw UFCStats CSV files and start from cleaning.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the production sanity-check suite.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show Chromium during scraping when supported by the scraper.",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help="Only build/predict one unresolved event ID.",
    )
    parser.add_argument(
        "--event-name",
        default=None,
        help="Only build/predict unresolved events matching this name.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Only build/predict unresolved fights on YYYY-MM-DD.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    project_root = find_project_root()
    python_executable = find_python(project_root)

    print("=" * 100)
    print("UFC FIGHT PREDICTOR — FULL PIPELINE")
    print("=" * 100)
    print("Project root:", project_root)
    print("Python:", python_executable)

    frozen_model = (
        project_root
        / "reports"
        / "final_test"
        / "final_catboost_model.cbm"
    )

    frozen_manifest = (
        project_root
        / "reports"
        / "final_model_optimization"
        / "frozen_feature_manifest.json"
    )

    frozen_calibration = (
        project_root
        / "reports"
        / "final_model_optimization"
        / "frozen_calibration_decision.json"
    )

    missing_model_artifacts = [
        path
        for path in [
            frozen_model,
            frozen_manifest,
            frozen_calibration,
        ]
        if not path.exists()
    ]

    if missing_model_artifacts:
        raise FileNotFoundError(
            "The frozen final model has not been produced yet. Missing:\n"
            + "\n".join(f"  - {path}" for path in missing_model_artifacts)
        )

    scraping_steps = [
        Step(
            "1/11 — Scrape completed and upcoming UFCStats events",
            "src.scraping.scrape_ufcstats",
        ),
        Step(
            "2/11 — Scrape/refresh fight cards",
            "src.scraping.scrape_ufcstats_fights",
        ),
        Step(
            "3/11 — Scrape fight details and statistics",
            "src.scraping.scrape_ufcstats_fight_details",
        ),
        Step(
            "4/11 — Scrape/refresh fighter profiles",
            "src.scraping.scrape_ufcstats_fighters",
        ),
    ]

    processing_steps = [
        Step(
            "5/11 — Clean and validate UFCStats data",
            "src.processing.clean_ufcstats_data",
        ),
        Step(
            "6/11 — Build historical fighter snapshots",
            "src.features.build_fighter_snapshots",
        ),
        Step(
            "7/11 — Build final Raw-Elo snapshots",
            "src.features.build_elo_snapshots",
        ),
        Step(
            "8/11 — Build historical fight dataset and feature manifest",
            "src.features.build_fight_dataset",
        ),
    ]

    prediction_steps = [
        Step(
            "9/11 — Build upcoming fight dataset",
            "src.features.build_upcoming_fight_dataset",
        ),
        Step(
            "10/11 — Run production sanity checks",
            "src.features.sanity_check_pipeline",
        ),
        Step(
            "11/11 — Classify upcoming fights with the frozen final model",
            "src.modeling.predict_upcoming_fights",
        ),
    ]

    if not args.skip_scraping:
        for step in scraping_steps:
            forwarded = (
                ["--show-browser"]
                if args.show_browser
                else None
            )
            run_step(
                python_executable,
                project_root,
                step,
                forwarded_args=forwarded,
            )
    else:
        print()
        print("Scraping skipped by request.")

    for step in processing_steps:
        run_step(
            python_executable,
            project_root,
            step,
        )

    upcoming_args = []
    if args.event_id:
        upcoming_args.extend(["--event-id", args.event_id])
    if args.event_name:
        upcoming_args.extend(["--event-name", args.event_name])
    if args.date:
        upcoming_args.extend(["--date", args.date])

    run_step(
        python_executable,
        project_root,
        prediction_steps[0],
        forwarded_args=upcoming_args or None,
    )

    if not args.skip_sanity:
        run_step(
            python_executable,
            project_root,
            prediction_steps[1],
        )
    else:
        print()
        print("Sanity checks skipped by request.")

    run_step(
        python_executable,
        project_root,
        prediction_steps[2],
    )

    print()
    print("=" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(
        "Predictions are available in:\n"
        "  reports/upcoming_predictions/upcoming_predictions.csv"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline cancelled by user.")
        raise SystemExit(130)
    except Exception as error:
        print()
        print("=" * 100)
        print("PIPELINE FAILED")
        print("=" * 100)
        print(error)
        raise SystemExit(1)
