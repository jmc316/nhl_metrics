# nhl_metrics

`nhl_metrics` is a Python project that forecasts NHL regular-season game results, projects final standings, and simulates playoff outcomes.

It pulls schedule and standings data from the NHL API, engineers team performance and travel-based features, trains/uses a scikit-learn random forest model, and writes daily prediction artifacts to CSV.

## What This Codebase Does

- Builds and updates a multi-season NHL game dataset (`output/season_schedules/`) from live API data.
- Engineers model features from historical results, including:
	- points percentage (season-to-date and rolling windows)
	- recent goals for/against trends
	- rest days between games
	- team travel distance over recent days (via venue geolocation + haversine distance)
	- playoff series state features
- Predicts future game scores day-by-day for the active schedule.
- Converts predicted game outcomes into projected standings with NHL tiebreak logic and playoff seeds.
- Simulates playoff brackets (Rounds 1-4) and predicts series/game progression.
- Runs Monte Carlo-style simulations (`n` iterations) to estimate probabilities for:
	- playoff qualification
	- seed outcomes
	- reaching later rounds / winning the Stanley Cup
- Provides a terminal UI with user/admin modes for:
	- viewing standings/team stats
	- updating current or historical-as-of-date predictions
	- running playoff probability simulations
	- reviewing model accuracy summaries

## Main Outputs

Generated files are saved under dated folders in `output/season_predictions/{date}/`, including:

- `regularseason_predictions_{date}.csv`
- `regularseason_standings_{date}.csv`
- `playoff_tree_predictions_{date}.csv`
- `skl_rf_model_features.txt`
- simulation probability outputs (when requested)

## Data + API

- NHL API client: `nhl-api-py`
- API reference: https://pypi.org/project/nhl-api-py/#description

## How to Run

Start from `main.py`, which launches the terminal UI.

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app from the project root:

```bash
python main.py
```

4. Choose mode in `main.py`:
	 - `role = 'admin'` (default): update predictions and playoff probabilities.
	 - `role = 'user'`: view team stats and model accuracy screens.

5. Follow the terminal menu prompts:
	 - Admin flow:
		 - Update Predictions (to-date or historical-as-of date)
		 - Playoff Spot Probability (simulation count `n`)
	 - User flow:
		 - NHL Team Stats
		 - Model Accuracy

## Notes

- First runs can take longer because schedule data/features/models are generated and cached to `output/`.
- Prediction artifacts are written to dated folders under `output/season_predictions/`.
- If you want to run the prediction pipeline directly (without UI), you can also execute `predict.py`.