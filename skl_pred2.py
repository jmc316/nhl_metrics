import os
import numpy as np
import constants as cons
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from skl_utils import prev10_result, prev10_result, prevN_gfpg, season_gfpg, season_result


def create_season_df(season_name, from_csv=True, to_csv=False, debug=False):

    print(f'\tRetrieving {season_name[:4]}-{season_name[4:]} NHL season schedule...')

    season_filename = season_name + '_' + cons.season_sched_filename

    # if the season schedule CSV file already exists in the output folder, load it instead of fetching the data from the API again
    if from_csv and season_filename in os.listdir(cons.season_sched_folder):
        if debug: print('\tSeason schedule CSV file already exists. Loading from file...')
        return pd.read_csv(cons.season_sched_folder + season_filename)
    if debug: print('\tSeason schedule CSV file does not exist. Fetching from API...')

    # get the first day of the season
    # this needs to be variable later on, but for now we can hardcode it
    first_day = f'{season_name[:4]}-{cons.season_stdt}'
    last_day = f'{season_name[4:]}-{cons.season_enddt}'

    # initialize an empty dataframe to store the season schedule data
    season_sched = pd.DataFrame()

    # loop through each week of the season and fetch the schedule data for that week, then concatenate it to the season schedule dataframe
    for week in pd.date_range(start=first_day, end=last_day, freq='W'):
        print(f'\t... {week.strftime("%Y-%m-%d")} ...')
        for dow in range(0, 7):
            while True:
                try:
                    # fetch the schedule data for this week and day of week from the NHL API
                    weekly_sched_raw = pd.DataFrame(cons.nhl_client.schedule.weekly_schedule(date=week.strftime("%Y-%m-%d"))['gameWeek'][dow]['games'])
                except Exception as ex:
                    # re-try this week's schedule if there was a timeout error
                    print(f'\t\t... {ex} ...')
                    continue

                # if there are no games in this week, skip to the next week
                if weekly_sched_raw.empty:
                    break

                # initialize columns that are the same as the raw data
                weekly_sched = weekly_sched_raw[['id', cons.season_col, cons.game_type_col, cons.starttime_utc_col, cons.venue_timezone_col]]

                weekly_sched.rename(columns={'id': cons.game_id_col, 'season': cons.season_name_col}, inplace=True)

                # only include NHL games (gameType 2 = regular season, 3 = playoffs)
                weekly_sched = weekly_sched.loc[weekly_sched[cons.game_type_col].isin([2, 3])]
                weekly_sched_raw = weekly_sched_raw.loc[weekly_sched_raw[cons.game_type_col].isin([2, 3])]

                # if there are no valid NHL games in this week, skip to the next week
                if weekly_sched.empty:
                    break

                # create columns that are derived from the raw data
                weekly_sched[cons.venue_col] = [item['default'] for item in weekly_sched_raw['venue']]
                weekly_sched[cons.away_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw[cons.away_team_col]]
                weekly_sched[cons.home_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw[cons.home_team_col]]

                # if the game has already been played, extract the scores and last period type from the raw data; otherwise, set these columns to None for now and they will be filled in with predictions later
                if cons.game_outcome_col in weekly_sched_raw.columns:
                    weekly_sched[cons.away_team_score_col] = [item['score'] for item in weekly_sched_raw[cons.away_team_col]]
                    weekly_sched[cons.home_team_score_col] = [item['score'] for item in weekly_sched_raw[cons.home_team_col]]
                    weekly_sched[cons.last_period_col] = [item['lastPeriodType'] for item in weekly_sched_raw[cons.game_outcome_col]]
                else:
                    weekly_sched[cons.away_team_score_col] = None
                    weekly_sched[cons.home_team_score_col] = None
                    weekly_sched[cons.last_period_col] = None

                weekly_sched[cons.season_name_col] = weekly_sched[cons.season_name_col].astype(str)

                # concatenate this week's schedule to the season schedule dataframe
                season_sched = pd.concat([season_sched, weekly_sched], ignore_index=True)

                break

    # save the season schedule to a CSV file for future use
    if to_csv:
        if debug: print('\tSaving season schedule to CSV file...')
        season_sched.to_csv(cons.season_sched_folder + season_name + '_' + cons.season_sched_filename, index=False)

    return season_sched


def create_feature_df():

    feature_df = create_feature_set()

    feature_df = feature_type_prep(feature_df)

    ### CREATE FEATURES INDEPENDENT OF PREDICT SET RESULTS###
    # these features are not dependent on the outcome of games
    # can be created for all completed and future games at the start of the season
    feature_df = sched_feature_add(feature_df)

    ### CREATE FEATURES MAPPED FROM OTHER COLUMNS ###
    # these features are not dependent on the outcome of games, but they are derived from other columns
    feature_df = sched_feature_map(feature_df)

    ### CREATE FEATURES DEPENDENT ON PREDICT SET RESULTS ###
    # these features are dependent on the outcome of games
    # can only be created one day at a time after predicting
    feature_df = predict_season(feature_df)


def predict_season(feature_df):

    # filter dataframe on all completed games and first scheduled game day after
    feature_df[cons.game_date_col] = feature_df[cons.starttime_utc_col].dt.date
    next_game_date = feature_df.loc[feature_df[cons.away_team_score_col].isna(), cons.game_date_col].min()
    feature_df_filt = feature_df[feature_df[cons.game_date_col] <= next_game_date]

    # add dependent features to filtered dataframe
    feature_df_filt = dependent_feature_add(feature_df_filt, backfill=True)

    # make predictions for first scheduled game day after completed games and add to filtered dataframe
    feature_df_filt = make_predictions(feature_df_filt)

    # re-create feature dataframe with added predictions and features
    feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    # loop through remaining scheduled game days and repeat following process:
    #   1. add dependent features to scheduled rows
    #   2. make predictions for scheduled rows 
    #   3. re-create feature dataframe with added predictions and features
    for next_game_date in feature_df.loc[feature_df[cons.away_team_score_col].isna(), cons.game_date_col].unique():
        print(f'\tPredicting games for {next_game_date.strftime("%Y-%m-%d")}...')
        feature_df_filt = feature_df[feature_df[cons.game_date_col] <= next_game_date]
        feature_df_filt = dependent_feature_add(feature_df_filt, backfill=False, debug=False)
        feature_df_filt = make_predictions(feature_df_filt)
        feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    print()


def make_predictions(feature_df):

    print('\tTraining model and generating predictions...')

    # handle missing values by dropping rows with missing target values and training the model on the remaining data
    feature_df = feature_df.replace({None: np.nan}, inplace=True) 
    train_df = feature_df.dropna()

    # train a random forest regressor on the training data
    X_train = train_df[cons.feature_cols]
    y_train = train_df[cons.predict_cols]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Identify rows with missing 'target' values and predict
    to_predict = feature_df[feature_df[cons.last_period_col].isna()]
    X_to_predict = to_predict[cons.feature_cols]
    predictions = model.predict(X_to_predict)

    # Add predictions back into the original DataFrame as ints
    to_predict[cons.predict_cols] = predictions.astype(int)
    feature_df.update(to_predict)

    return feature_df


def dependent_feature_add(feature_df, backfill=True, debug=True):

    # calculate the number of wins for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home wins ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_wins_col, cons.home_team_name_col)

    # calculate the number of losses for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home losses ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_losses_col, cons.home_team_name_col)

    # calculate the number of OTLs for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home OTLs ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_otls_col, cons.home_team_name_col)

    # calculate the number of wins for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away wins ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_wins_col, cons.away_team_name_col)

    # calculate the number of losses for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away losses ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_losses_col, cons.away_team_name_col)

    # calculate the number of OTLs for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away OTLs ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_otls_col, cons.away_team_name_col)

    # calculate the number of wins in the previous 10 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home prev 10 wins ...')
    feature_df = prev10_result(feature_df, backfill, cons.home_team_prev_10_wins_col, cons.home_team_name_col)

    # calculate the number of losses in the previous 10 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home prev 10 losses ...')
    feature_df = prev10_result(feature_df, backfill, cons.home_team_prev_10_losses_col, cons.home_team_name_col)

    # calculate the number of OTLs in the previous 10 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home prev 10 OTLs ...')
    feature_df = prev10_result(feature_df, backfill, cons.home_team_prev_10_otl_col, cons.home_team_name_col)

    # calculate the number of wins in the previous 10 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away prev 10 wins ...')
    feature_df = prev10_result(feature_df, backfill, cons.away_team_prev_10_wins_col, cons.away_team_name_col)

    # calculate the number of losses in the previous 10 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away prev 10 losses ...')
    feature_df = prev10_result(feature_df, backfill, cons.away_team_prev_10_losses_col, cons.away_team_name_col)

    # calculate the number of OTLs in the previous 10 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away prev 10 OTLs ...')
    feature_df = prev10_result(feature_df, backfill, cons.away_team_prev_10_otl_col, cons.away_team_name_col)

    # calculate goals for in the previous 10 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home prev 10 goals for ...')
    feature_df = prevN_gfpg(10, feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col)

    # calculate goals for in the previous 10 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away prev 10 goals for ...')
    feature_df = prevN_gfpg(10, feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col)

    # calculate goals for for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home goals for ...')
    feature_df = season_gfpg(feature_df, backfill, cons.home_team_goals_for_col, cons.home_team_name_col)

    # calculate goals for for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away goals for ...')
    feature_df = season_gfpg(feature_df, backfill, cons.away_team_goals_for_col, cons.away_team_name_col)

    return feature_df


def sched_feature_add(feature_df):

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    feature_df[cons.starttime_utc_col] = pd.to_datetime(feature_df[cons.starttime_utc_col])
    # feature_df[cons.game_year_col] = feature_df[cons.starttime_utc_col].dt.year
    feature_df[cons.game_month_col] = feature_df[cons.starttime_utc_col].dt.month
    # feature_df[cons.game_day_col] = feature_df[cons.starttime_utc_col].dt.day
    feature_df[cons.game_time_col] = feature_df[cons.starttime_utc_col].dt.hour * 60 + feature_df[cons.starttime_utc_col].dt.minute

    return feature_df


def sched_feature_map(feature_df):

    # add feature columns for team ids
    feature_df[cons.home_team_id_col] = feature_df[cons.home_team_name_col].map(cons.team_id_map)
    feature_df[cons.away_team_id_col] = feature_df[cons.away_team_name_col].map(cons.team_id_map)

    # encode the categorical features using ordinal encoding
    encoder = OrdinalEncoder()
    feature_df[cons.venue_timezone_col] = encoder.fit_transform(feature_df[[cons.venue_timezone_col]]).astype(int)
    feature_df[cons.venue_col] = encoder.fit_transform(feature_df[[cons.venue_col]]).astype(int)

    # encode the target variable 'lastPeriod' using a mapping of the period types to integers
    feature_df[cons.last_period_col] = feature_df.loc[
        feature_df[cons.last_period_col].notna(), cons.last_period_col].map(cons.last_period_map).astype(int)

    return feature_df


def feature_type_prep(feature_df):

    feature_df[cons.season_name_col] = feature_df[cons.season_name_col].astype(str)

    return feature_df


def create_feature_set():
    # a list of schedule files that have already been generated
    season_sched_list = [file for file in os.listdir(cons.season_sched_folder) if file.endswith(cons.season_sched_filename)]

    data_df = pd.DataFrame()

    for filename in season_sched_list:
        if data_df.empty:
            data_df = create_season_df(filename[:8], from_csv=True, to_csv=False)
        else:
            data_df = pd.concat([data_df, create_season_df(filename[:8], from_csv=True, to_csv=False)], ignore_index=True)

    return data_df


if __name__ == "__main__":

    ######################
    # create season schedule dataframe for inputted season
    # create_season_df('20252026', from_csv=False, to_csv=True, debug=True)
    feature_df = create_feature_df()
    # season_results_points = assign_game_points(season_results)
    # final_standings = generate_final_standings(season_results_points)

    ######################
    # create a single season prediction wiuth saving the results as csv files
    # season_results = season_predictions(to_csv=True)
    # season_results = assign_game_points(season_results, to_csv=True)
    # final_standings = generate_final_standings(pd.DataFrame(), to_csv=True, load_csv=True)

    ######################
    # create playoff spot predictions for current season based on n simulations
    # playoff_spot_predictions(n=20)