# UFC Predictor — One-click production pipeline

This package adds a single production entry point that runs the UFC predictor from UFCStats scraping to upcoming-fight classification.

## Files

Copy:

- `run_ufc_pipeline.py` → `src/pipeline/run_ufc_pipeline.py`
- `predict_upcoming_fights.py` → `src/modeling/predict_upcoming_fights.py`
- `run_ufc_pipeline.bat` → project root
- `build_ufc_pipeline_exe.bat` → project root

Create `src/pipeline/__init__.py` if the folder does not already exist.

## Full production order

1. `src.scraping.scrape_ufcstats`
2. `src.scraping.scrape_ufcstats_fights`
3. `src.scraping.scrape_ufcstats_fight_details`
4. `src.scraping.scrape_ufcstats_fighters`
5. `src.processing.clean_ufcstats_data`
6. `src.features.build_fighter_snapshots`
7. `src.features.build_elo_snapshots`
8. `src.features.build_fight_dataset`
9. `src.features.build_upcoming_fight_dataset`
10. `src.features.sanity_check_pipeline`
11. `src.modeling.predict_upcoming_fights`

The scraper order follows the project dependencies: the fight-card scraper consumes `events.csv`; fight details and fighter profiles consume `fights.csv`; cleaning consumes the raw CSV datasets.

## Required frozen model artifacts

The production launcher expects:

- `reports/final_test/final_catboost_model.cbm`
- `reports/final_model_optimization/frozen_feature_manifest.json`
- `reports/final_model_optimization/frozen_calibration_decision.json`
- `reports/final_test/final_calibrator.pkl` only when the frozen calibration method is Platt or isotonic.

These files are produced by the final model-development/evaluation notebooks.

## Use immediately without building an EXE

From the project root:

```powershell
run_ufc_pipeline.bat
```

or:

```powershell
python -m src.pipeline.run_ufc_pipeline
```

## Useful options

Skip scraping and rebuild from current raw data:

```powershell
python -m src.pipeline.run_ufc_pipeline --skip-scraping
```

Show Chromium:

```powershell
python -m src.pipeline.run_ufc_pipeline --show-browser
```

Only predict one date:

```powershell
python -m src.pipeline.run_ufc_pipeline --date 2026-09-19
```

Only predict one event ID:

```powershell
python -m src.pipeline.run_ufc_pipeline --event-id <EVENT_ID>
```

## Build the Windows EXE

Run:

```powershell
build_ufc_pipeline_exe.bat
```

The result will be:

```text
dist\UFC_Predictor_Pipeline.exe
```

Copy that EXE to the project root.

The EXE is intentionally a lightweight project launcher. It uses the project's `.venv\Scripts\python.exe` to execute the real modules. CatBoost, Playwright and Chromium therefore remain in the virtual environment instead of being duplicated inside a huge standalone executable.

## Prediction outputs

The final inference step writes:

- `reports/upcoming_predictions/upcoming_predictions.csv`
- `reports/upcoming_predictions/upcoming_predictions.parquet`
- timestamped CSV copies for traceability.

The CSV contains, among other fields:

- event
- fighter A
- fighter B
- P(fighter A)
- P(fighter B)
- predicted winner
- prediction confidence
