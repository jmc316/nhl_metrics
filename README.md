# nhl_metrics

nhl_metrics is a Python project that forecasts NHL regular-season game results, projects final standings, and simulates playoff outcomes.

The project pulls schedule and standings data from the NHL API, engineers team performance and travel features, trains and applies a scikit-learn random forest model, writes daily prediction artifacts to CSV, and generates visual outputs (including per-game SHAP explanations) to summarize predictions.

## Overview

- Builds and updates a multi-season NHL game dataset in output/season_schedules/ from live API data.
- Engineers a broad set of schedule, team, player, and goalie features from historical results (see [Model](#model) below for the subset actually used to train the model), including:
  - team Elo ratings
  - points percentage, goal differential, corsi/fenwick, power play/penalty kill percentage (season-to-date and rolling windows)
  - starting goalie save percentage and goals-against average (overall and by strength state)
  - lineup strength, based on a weighted value of each roster player's recent production
  - rest days, games played, travel distance, and time zones crossed over recent windows
  - rivalry, market intensity, and other schedule/venue context (outdoor venues, altitude, home openers, etc.)
  - playoff series state features
- Predicts future game scores day by day for the active schedule.
- Converts predicted game outcomes into projected standings with NHL tiebreak logic and playoff seeding.
- Simulates playoff brackets (Rounds 1-4) and predicts series and game progression.
- Runs Monte Carlo-style simulations (n iterations) to estimate probabilities for:
  - playoff qualification
  - seed outcomes
  - reaching later rounds
  - winning the Stanley Cup
- Provides a terminal UI for:
  - updating current or historical as-of-date predictions
  - running playoff probability simulations
  - reviewing model accuracy summaries

![Playoff Probabilities](images/sample_playoff_tree.png)

## Model

Game outcomes (home team win/loss) are predicted with a scikit-learn `RandomForestClassifier` (300 trees, `max_depth=10`, `min_samples_leaf=45`, `max_features='sqrt'`), defined in [utils/skl_utils.py](utils/skl_utils.py). The model is validated with a walk-forward, season-by-season split: for each season N (after the first), the model trains on all prior seasons and validates on season N, so validation always predicts strictly future data. A final model is then re-fit on all completed games for live predictions.

### Features

The model is trained on the following relative (home-minus-away, unless noted) and standalone features, selected in `preprocess_feature_data()` from the full engineered feature set:

- Relative team Elo rating
- Home team's regular season game-number percentage (a proxy for how far into the season the game falls)
- Relative lineup strength (weighted value of each team's active roster)
- Relative time zones crossed over the last 7 and 4 days
- Relative games played over the last 7 days
- Relative penalty kill percentage over the last 5 and 20 games
- Relative corsi percentage over the last 5 and 20 games
- Relative even-strength and power-play starting goalie save percentage over the last 3 games
- Whether the home team is returning home after a 3+ game road trip (boolean)
- Rivalry level of the matchup (division/conference/neither)
- Venue (label encoded) and venue timezone (target encoded)

### Validation Metrics

Metrics below are averaged across all walk-forward validation folds:

| Metric | Value |
| --- | --- |
| Baseline Accuracy (always pick home team) | 0.537 |
| Model Validation Accuracy | 0.5869 |
| Target AUC | 0.68 |
| Model Validation AUC | 0.6177 |
| Model Validation Log Loss | 0.6704 |
| Model Validation Brier Score | 0.2389 |

Permutation importance and pairwise feature correlations (>|0.7|) are also printed to the terminal at the end of each training run for further feature diagnostics.

### Prediction Explainability (SHAP)

For each predicted game, a SHAP waterfall chart is generated via `explain_predictions()` in [utils/skl_utils.py](utils/skl_utils.py), showing how each feature pushed the predicted home win probability up or down from the model's baseline. These are saved alongside the day's other prediction artifacts as `shap_{home}_{away}_{date}.png` under output/season_predictions/{date}/.

## Requirements

- Python 3.10+ recommended
- pip
- Internet access (for live NHL API pulls)

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies from project root:

```bash
pip install -r requirements.txt
```

## Running The App

Start from main.py, which launches the terminal UI:

```bash
python main.py
```

Then follow terminal prompts:
  - Update Predictions (to-date or historical as-of date)
  - Playoff Spot Probability (simulation count n)

## Main Outputs

Generated files are saved under dated folders in output/season_predictions/{date}/, including:

- regularseason_predictions_{date}.csv
- regularseason_standings_{date}.csv
- playoff_tree_predictions_{date}.csv
- skl_rf_model_features.txt
- shap_{home}_{away}_{date}.png (per-game SHAP explanation charts)
- simulation probability outputs (when requested)

## Data And API

- NHL API client: nhl-api-py
- API reference: https://pypi.org/project/nhl-api-py/#description

## Project Structure (Key Files)

- main.py: terminal entry point
- predict.py: direct prediction pipeline execution
- playoffs.py and playoff_probability.py: postseason simulation logic
- features/features.py: feature engineering orchestration (see also features/feat_*.py)
- utils/skl_utils.py: model training, inference, and SHAP explanation
- nhl_client.py: NHL API access

## Notes

- First runs can take longer because schedule data, features, and models are generated and cached to output/.
- Prediction artifacts are written to dated folders under output/season_predictions/.
- To run the prediction pipeline directly (without UI), execute predict.py.
