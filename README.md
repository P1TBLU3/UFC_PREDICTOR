# UFC Fight Predictor 🥊

A small machine learning project built to predict UFC fight outcomes using historical UFC data.

The main idea is simple:

> Collect UFC fight data, build information that would have been available **before each fight**, train a model on historical results, and use it to estimate the probability of each fighter winning an upcoming bout.

The project started as a way to learn and experiment with machine learning, but it gradually became a complete end-to-end pipeline including scraping, data cleaning, feature engineering, Elo ratings, model optimization, probability calibration and prediction of upcoming events.

---

## Project overview

The pipeline roughly looks like this:

```text
UFCStats
   ↓
Scraping
   ↓
Data cleaning
   ↓
Pre-fight fighter history
   ↓
Elo ratings
   ↓
Fight-level dataset
   ↓
CatBoost model
   ↓
Probability calibration
   ↓
Upcoming fight predictions
```

One of the main goals throughout the project was avoiding **future information leakage**.

For every historical fight, the model only sees information that would have been available before that fight took place.

---

# Project structure

The exact repository may evolve slightly, but the main structure is:

```text
UFC_PREDICTOR/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── scraping/
│   ├── processing/
│   ├── features/
│   ├── modeling/
│   └── pipeline/
│
├── run_ufc_pipeline.bat
├── build_ufc_pipeline_exe.bat
└── README.md
```

---

## `data/`

Contains the datasets generated at the different stages of the pipeline.

### `data/raw/`

Raw data downloaded from UFCStats.

This includes information such as:

* events;
* fights;
* detailed fight results;
* fight statistics;
* round statistics;
* fighter profiles.

These files are kept as close as possible to the original scraped data.

### `data/interim/`

Cleaned and validated versions of the raw data.

This stage converts fields to the correct data types, normalizes values and checks that the different tables are consistent with each other.

### `data/processed/`

Datasets ready to be used by the machine learning pipeline.

This includes things such as:

* fighter historical snapshots;
* Elo snapshots;
* the final fight-level training dataset;
* the upcoming fight dataset.

---

# `src/`

This folder contains the actual pipeline.

## `src/scraping/`

Scripts responsible for collecting data from UFCStats.

### `scrape_ufcstats.py`

Downloads the list of UFC events, including completed and upcoming events.

### `scrape_ufcstats_fights.py`

Visits each event and collects the fight card.

This gives us the fighters, fight order, results and basic fight information.

### `scrape_ufcstats_fight_details.py`

Collects the more detailed information for each fight.

Among other things, it generates:

* fight details;
* total fight statistics;
* round-level statistics.

### `scrape_ufcstats_fighters.py`

Collects fighter profile information such as:

* height;
* reach;
* stance;
* date of birth.

---

# `src/processing/`

## `clean_ufcstats_data.py`

Takes the raw CSV files and turns them into clean Parquet datasets.

It also performs several consistency checks before allowing the pipeline to continue.

In short:

```text
raw UFCStats data
        ↓
clean + validate
        ↓
interim Parquet files
```

---

# `src/features/`

This is where most of the feature engineering happens.

## `build_fighter_snapshots.py`

Recreates the historical state of every fighter immediately before each fight.

For example:

* previous fights;
* wins and losses;
* win streak;
* previous striking statistics;
* takedown statistics;
* control time;
* submissions;
* inactivity.

The important part is that the current fight is **not included** in its own features.

---

## `build_elo_snapshots.py`

Builds the custom Elo rating system used in the project.

Instead of using only one rating, the project keeps separate information for:

* overall performance;
* striking;
* grappling offense;
* grappling defense.

The final Elo configuration was selected through several temporal validation experiments.

---

## `build_fight_dataset.py`

Combines everything into one row per historical fight.

For each fight it joins:

* fighter A historical features;
* fighter B historical features;
* Elo ratings;
* physical characteristics;
* difference features between both fighters.

This produces the final dataset used to train the model.

The A/B assignment is deterministic and independent of who actually won the fight.

---

## `build_upcoming_fight_dataset.py`

Creates the same features for fights that have not happened yet.

The important part is that it uses the **same feature structure as the training dataset**, so the final model can directly classify upcoming fights.

---

## `sanity_check_pipeline.py`

Runs a final set of checks over the production datasets.

It looks for problems such as:

* duplicated fights;
* inconsistent fighter histories;
* accidental future information;
* mismatched Elo values;
* training/upcoming schema differences.

If this script passes, the data pipeline should be in a good state.

---

# `src/modeling/`

Contains the final prediction logic.

The main model used in the project is **CatBoost**.

Random Forest was also tested during development, but CatBoost became the main model used for the final pipeline.

## `predict_upcoming_fights.py`

Loads the frozen final model and predicts upcoming UFC fights.

It outputs probabilities such as:

```text
Fighter A: 0.64
Fighter B: 0.36

Predicted winner: Fighter A
```

It also applies the final probability calibration method if calibration was selected during model development.

---

# `notebooks/`

The notebooks are mainly the **experimental part of the project**.

They are not required every time predictions are generated.

They were used to understand the data, test ideas and gradually decide how the final system should work.

Examples include:

* exploratory data analysis;
* feature analysis;
* Elo testing;
* Elo optimization;
* CatBoost hyperparameter optimization;
* probability calibration;
* final untouched test evaluation.

A simplified view is:

```text
Notebooks
   ↓
experiments
   ↓
compare approaches
   ↓
freeze the best decisions
   ↓
move them into src/
```

So the notebooks document the reasoning and experimentation behind the final production pipeline.

---

# Final model

The final model is based on **CatBoost** and uses historical pre-fight information from both fighters.

The main evaluation metric used during development was **Log Loss**, because the objective is not only to guess the winner but also to produce useful probabilities.

Other metrics used include:

* Brier Score;
* ROC AUC;
* Accuracy;
* reliability curves;
* Expected Calibration Error.

The final test set is kept separate from model development and is only used once the model configuration has been frozen.

---

# Manual pipeline

If running everything manually, the scripts should be executed in this order.

```text
1. scrape_ufcstats.py

2. scrape_ufcstats_fights.py

3. scrape_ufcstats_fight_details.py

4. scrape_ufcstats_fighters.py

5. clean_ufcstats_data.py

6. build_fighter_snapshots.py

7. build_elo_snapshots.py

8. build_fight_dataset.py

9. build_upcoming_fight_dataset.py

10. sanity_check_pipeline.py

11. predict_upcoming_fights.py
```

Using Python modules from the project root, this would look roughly like:

```powershell
python -m src.scraping.scrape_ufcstats
python -m src.scraping.scrape_ufcstats_fights
python -m src.scraping.scrape_ufcstats_fight_details
python -m src.scraping.scrape_ufcstats_fighters

python -m src.processing.clean_ufcstats_data

python -m src.features.build_fighter_snapshots
python -m src.features.build_elo_snapshots
python -m src.features.build_fight_dataset
python -m src.features.build_upcoming_fight_dataset
python -m src.features.sanity_check_pipeline

python -m src.modeling.predict_upcoming_fights
```

---

# Automatic pipeline

The whole production process can also be run automatically.

Instead of executing every script manually, the project includes a pipeline launcher that runs everything in the correct order and stops if one of the stages fails.

You can run:

```powershell
python -m src.pipeline.run_ufc_pipeline
```

or simply:

```text
run_ufc_pipeline.bat
```

There is also an option to build a small Windows executable:

```text
build_ufc_pipeline_exe.bat
```

which generates:

```text
UFC_Predictor_Pipeline.exe
```

More information about the automatic pipeline, available options and the Windows executable can be found in:

```text
README_BINARY.md
```

---

# Output

Once the full pipeline finishes, the upcoming predictions are stored in:

```text
reports/upcoming_predictions/
```

The main output is:

```text
upcoming_predictions.csv
```

with information such as:

```text
Event
Fighter A
Fighter B
P(Fighter A)
P(Fighter B)
Predicted winner
Confidence
```

---

# Final note

This project is mainly an experiment in applying machine learning to real-world sports data.

The predictions are not meant to be treated as guaranteed fight outcomes. MMA is highly unpredictable, and no dataset can fully represent everything that happens inside the cage.

Still, that is part of what makes the problem interesting.
