import playoffs

import pandas as pd
import constants as cons
import utils.nhl_utils as nhlu
import utils.skl_utils as sklu

from utils.file_utils import csvLoad
from datetime import datetime as dt
from features.features import feature_data_load
from analyze import prediction_analysis
from pred_returns import daily_probability
from predict import predict_season, playoff_spot_predictions


def get_asofdate():

    if dt.now().month < int(cons.season_stdt[:2]):
        cur_season_stdt = pd.to_datetime(f'{dt.now().year - 1}-{cons.season_stdt}').date()
    else:
        cur_season_stdt = pd.to_datetime(f'{dt.now().year}-{cons.season_stdt}').date()

    max_dt = dt.now().date()

    # get user input for date
    print('Select a date to run the simulations from:')
    today_dt = input('> ')
    print()

    return today_dt


def todate_predict():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    print('Updating predictions for current season...\n')
    feature_df = predict_season(to_csv=True, set_model_state=True, today_dt=today_dt)
    feature_df_game_points = nhlu.assign_game_points(feature_df.loc[(feature_df[cons.game_type_col]==2) & (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col]))])
    season_results_df = nhlu.generate_final_standings(feature_df_game_points, today_dt, to_csv=True)
    if not feature_df.loc[(feature_df[cons.game_type_col]==2) &
                      (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col])) &
                      feature_df[cons.last_period_col].isna()].empty:
        nhlu.nhl_team_standings(season_results_df)
    playoff_df, _, _, _ = playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt)

    # set the first comparison date to the first date of the current season
    prob_date_since = playoff_df.loc[playoff_df[cons.season_name_col]==playoff_df[cons.season_name_col].max(),
                                     cons.starttime_est_col].dt.date.min().strftime(cons.date_format_yyyy_mm_dd)
    daily_probability(today_dt, season=playoff_df[cons.season_name_col].max(), date_since=prob_date_since)


def historic_predict(today_dt=None):

    if today_dt is None:
        today_dt = get_asofdate()

    print(f'Updating predictions for current season as of {today_dt}...\n')
    feature_df = predict_season(to_csv=True, set_model_state=True, today_dt=today_dt)
    feature_df_game_points = nhlu.assign_game_points(feature_df.loc[(feature_df[cons.game_type_col]==2) & (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col]))])
    season_results_df = nhlu.generate_final_standings(feature_df_game_points, today_dt, to_csv=True)
    if not feature_df.loc[(feature_df[cons.game_type_col]==2) &
                      (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col])) &
                      feature_df[cons.last_period_col].isna()].empty:
        nhlu.nhl_team_standings(season_results_df)
    playoff_df, _, _, _ = playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt)

    # set the first comparison date to the first date of the current season
    prob_date_since = playoff_df.loc[playoff_df[cons.season_name_col]==playoff_df[cons.season_name_col].max(),
                                        cons.starttime_est_col].dt.date.min().strftime(cons.date_format_yyyy_mm_dd)
    daily_probability(today_dt, date_since=prob_date_since, season=playoff_df[cons.season_name_col].max())


def historic_range_predict():

    start_dt = get_asofdate()

    print('Select date to generate predictions through:')
    end_date = input('> ')
    print()

    for single_date in pd.date_range(start=start_dt, end=end_date):
        historic_predict(today_dt=single_date.strftime(cons.date_format_yyyy_mm_dd))


def todate_playoff_spot_predict():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    # get user input for n
    print('Select the number of iterations to run the simulation for:')
    n_in = input('> ')
    print()

    playoff_spot_predictions(today_dt, n=n_in)


def historic_playoff_spot_predict():

    today_dt = get_asofdate()

    # get user input for n
    print('Select the number of iterations to run the simulation for:')
    n_in = input('> ')
    print()

    playoff_spot_predictions(today_dt, n=n_in)


def run_inference():

    print('1. Update to-date Predictions')
    print('2. Update historic Predictions')
    print('3. Update range of historic predictions')
    user_response = input('> ')
    print()

    match user_response:
        case '1': # 'Update to-date Predictions'
            todate_predict()
        case '2': # 'Update historic Predictions'
            historic_predict()
        case '3': # 'Update range of historic predictions'
            historic_range_predict()


def update_playoff_spot_probabilities():

    print('1. Update to-date Playoff Spot Probabilities')
    print('2. Update historic Playoff Spot Probabilities')
    user_response = input('> ')
    print()

    match user_response:
        case '1': # 'Update to-date Playoff Spot Probabilities'
            todate_playoff_spot_predict()
        case '2': # 'Update historic Playoff Spot Probabilities'
            historic_playoff_spot_predict()


def model_accuracy():

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    since_dt = get_asofdate().strftime(cons.date_format_yyyy_mm_dd)

    season_prediction_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))

    prediction_analysis(season_prediction_df, since_dt, today_dt)


def train_model():
    print('Training model...')

    # load all of the feature data with all actuals
    feature_df = feature_data_load()

    # cons.last_actual_game_date = feature_df.loc[feature_df[cons.last_period_col].notna(), cons.starttime_est_col].dt.date.max()

    # pre-process the feature data
    processed_df, feature_list = sklu.preprocess_feature_data(feature_df)

    # train the model on the training set and save the model to a file for future use
    _ = sklu.model_train(processed_df, feature_list, set_model_state=True, today_dt=dt.now().date().strftime(cons.date_format_yyyy_mm_dd))

    print('Model training execution complete.\n')