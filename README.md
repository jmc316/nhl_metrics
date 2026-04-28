# nhl_metrics

nhl_metrics is a Python project that forecasts NHL regular-season game results, projects final standings, and simulates playoff outcomes.

The project pulls schedule and standings data from the NHL API, engineers team performance and travel features, trains and applies a scikit-learn random forest model, writes daily prediction artifacts to CSV, and generates visual outputs to summarize predictions.

## Overview

- Builds and updates a multi-season NHL game dataset in output/season_schedules/ from live API data.
- Engineers model features from historical results, including:
  - points percentage (season-to-date and rolling windows)
  - recent goals for and against trends
  - rest days between games
  - team travel distance over recent days (venue geolocation + haversine distance)
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
- simulation probability outputs (when requested)

## Data And API

- NHL API client: nhl-api-py
- API reference: https://pypi.org/project/nhl-api-py/#description

## Project Structure (Key Files)

- main.py: terminal entry point
- predict.py: direct prediction pipeline execution
- playoffs.py and playoff_probability.py: postseason simulation logic
- features.py: feature engineering
- nhl_client.py: NHL API access

## Notes

- First runs can take longer because schedule data, features, and models are generated and cached to output/.
- Prediction artifacts are written to dated folders under output/season_predictions/.
- To run the prediction pipeline directly (without UI), execute predict.py.