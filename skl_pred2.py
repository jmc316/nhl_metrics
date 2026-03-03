import os
import numpy as np
import constants as cons
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from skl_utils import prevN_result, prevN_gfpg, season_gfpg, season_result
import skl_utils as sku


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


def predict_season(feature_df):

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

    feature_df.to_csv('results.csv', index=False)

    print('Season predictions complete.')


def make_predictions(data_df):

    # data_df.drop(columns=[cons.game_id_col, cons.game_time_col, cons.venue_timezone_col, cons.game_type_col, cons.season_name_col], inplace=True)
    # data_df.drop(columns=[cons.game_date_col], inplace=True)

    label_encoder = LabelEncoder()
    categorical_df = data_df.select_dtypes(include=['object', 'str']).apply(label_encoder.fit_transform)
    numerical_df = data_df.select_dtypes(exclude=['object', 'str'])
    encoded_df = pd.concat([numerical_df, categorical_df], axis=1)
    encoded_df.replace({None: np.nan}, inplace=True) 

    x_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), encoded_df.columns.difference(cons.predict_cols)]
    y_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), cons.predict_cols]
    x_predict_df = encoded_df.loc[encoded_df[cons.home_team_score_col].isna(), encoded_df.columns.difference(cons.predict_cols)]

    model = RandomForestRegressor(n_estimators=1000, random_state=0, oob_score=True)

    model.fit(x_train_df.values, y_train_df.values)

    oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    trainset_predictions = model.predict(x_train_df.values)

    mse = mean_squared_error(y_train_df.values, trainset_predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y_train_df.values, trainset_predictions)
    print(f'R-squared: {r2}')

    predictset_predictions = model.predict(x_predict_df.values)

    predict_df = data_df[data_df[cons.last_period_col].isna()]
    predict_df[cons.predict_cols] = predictset_predictions
    predict_df['awayTeamScore_int'] = predict_df[cons.away_team_score_col].round().astype(int)
    predict_df['homeTeamScore_int'] = predict_df[cons.home_team_score_col].round().astype(int)
    predict_df['lastPeriod'] = np.where(predict_df['homeTeamScore_int'] == predict_df['awayTeamScore_int'], 'OT', 'REG')

    data_df.update(predict_df[cons.predict_cols])

    # importances = model.feature_importances_
    # importance_df = pd.DataFrame({'feature': x_train_df.columns, 'importance': importances})
    # print(importance_df.sort_values(by='importance', ascending=False))

    return data_df


def dependent_feature_add(feature_df, backfill=True, debug=True):

    # calculate the number of wins for the home team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] home team wins ...')
    # feature_df = season_result(feature_df, backfill, cons.home_team_wins_col, cons.home_team_name_col)

    # # calculate the number of losses for the home team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] home team losses ...')
    # feature_df = season_result(feature_df, backfill, cons.home_team_losses_col, cons.home_team_name_col)

    # # calculate the number of OTLs for the home team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] home team OTLs ...')
    # feature_df = season_result(feature_df, backfill, cons.home_team_otls_col, cons.home_team_name_col)

    # # calculate the points percentage of the home team in all matchups
    # if debug: print('\t\t... [feature_creation] home team points percentage ...')
    # if 'home_points_percentage' not in feature_df.columns:
    #     feature_df['home_points_percentage'] = (feature_df[cons.home_team_wins_col] * 2 + feature_df[cons.home_team_otls_col]) / ((feature_df[cons.home_team_wins_col] + feature_df[cons.home_team_otls_col] + feature_df[cons.home_team_losses_col]) * 2)
    # else:
    #     feature_df_new = feature_df.loc[feature_df['home_points_percentage'].isna()]
    #     feature_df_new['home_points_percentage'] = (feature_df_new[cons.home_team_wins_col] * 2 + feature_df_new[cons.home_team_otls_col]) / ((feature_df_new[cons.home_team_wins_col] + feature_df_new[cons.home_team_otls_col] + feature_df_new[cons.home_team_losses_col]) * 2)
    #     feature_df.update(feature_df_new['home_points_percentage'])
    # feature_df.drop(columns=[cons.home_team_wins_col, cons.home_team_otls_col, cons.home_team_losses_col], inplace=True)

    # # calculate the number of wins for the away team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] away team wins ...')
    # feature_df = season_result(feature_df, backfill, cons.away_team_wins_col, cons.away_team_name_col)

    # # calculate the number of losses for the away team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] away team losses ...')
    # feature_df = season_result(feature_df, backfill, cons.away_team_losses_col, cons.away_team_name_col)

    # # calculate the number of OTLs for the away team in all matchups
    # if debug: print('\t\t... [sub_feature_creation] away team OTLs ...')
    # feature_df = season_result(feature_df, backfill, cons.away_team_otls_col, cons.away_team_name_col)

    # # calculate the points percentage of the away team in all matchups
    # if debug: print('\t\t... [feature_creation] away team points percentage ...')
    # if 'away_points_percentage' not in feature_df.columns:
    #     feature_df['away_points_percentage'] = (feature_df[cons.away_team_wins_col] * 2 + feature_df[cons.away_team_otls_col]) / ((feature_df[cons.away_team_wins_col] + feature_df[cons.away_team_otls_col] + feature_df[cons.away_team_losses_col]) * 2)
    # else:
    #     feature_df_new = feature_df.loc[feature_df['away_points_percentage'].isna()]
    #     feature_df_new['away_points_percentage'] = (feature_df_new[cons.away_team_wins_col] * 2 + feature_df_new[cons.away_team_otls_col]) / ((feature_df_new[cons.away_team_wins_col] + feature_df_new[cons.away_team_otls_col] + feature_df_new[cons.away_team_losses_col]) * 2)
    #     feature_df.update(feature_df_new['away_points_percentage'])
    # feature_df.drop(columns=[cons.away_team_wins_col, cons.away_team_otls_col, cons.away_team_losses_col], inplace=True)

    # calculate the number of wins in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 wins ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev7Wins', cons.home_team_name_col, 7)

    # calculate the number of losses in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 losses ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev7Losses', cons.home_team_name_col, 7)

    # calculate the number of OTLs in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 OTLs ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev7OTLs', cons.home_team_name_col, 7)

    # calculate the points percentage in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 7 points percentage ...')
    if 'home_team_prev_7_points_percentage' not in feature_df.columns:
        feature_df['home_team_prev_7_points_percentage'] = (feature_df['homeTeamPrev7Wins'] * 2 + feature_df['homeTeamPrev7OTLs']) / ((feature_df['homeTeamPrev7Wins'] + feature_df['homeTeamPrev7OTLs'] + feature_df['homeTeamPrev7Losses']) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df['home_team_prev_7_points_percentage'].isna()]
        feature_df_new['home_team_prev_7_points_percentage'] = (feature_df_new['homeTeamPrev7Wins'] * 2 + feature_df_new['homeTeamPrev7OTLs']) / ((feature_df_new['homeTeamPrev7Wins'] + feature_df_new['homeTeamPrev7OTLs'] + feature_df_new['homeTeamPrev7Losses']) * 2)
        feature_df.update(feature_df_new['home_team_prev_7_points_percentage'])
    feature_df.drop(columns=['homeTeamPrev7Wins', 'homeTeamPrev7OTLs', 'homeTeamPrev7Losses'], inplace=True)

    # calculate the number of wins in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 wins ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev3Wins', cons.home_team_name_col, 3)

    # calculate the number of losses in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 losses ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev3Losses', cons.home_team_name_col, 3)

    # calculate the number of OTLs in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 OTLs ...')
    feature_df = prevN_result(feature_df, backfill, 'homeTeamPrev3OTLs', cons.home_team_name_col, 3)

    # calculate the points percentage in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 3 points percentage ...')
    if 'home_team_prev_3_points_percentage' not in feature_df.columns:
        feature_df['home_team_prev_3_points_percentage'] = (feature_df['homeTeamPrev3Wins'] * 2 + feature_df['homeTeamPrev3OTLs']) / ((feature_df['homeTeamPrev3Wins'] + feature_df['homeTeamPrev3OTLs'] + feature_df['homeTeamPrev3Losses']) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df['home_team_prev_3_points_percentage'].isna()]
        feature_df_new['home_team_prev_3_points_percentage'] = (feature_df_new['homeTeamPrev3Wins'] * 2 + feature_df_new['homeTeamPrev3OTLs']) / ((feature_df_new['homeTeamPrev3Wins'] + feature_df_new['homeTeamPrev3OTLs'] + feature_df_new['homeTeamPrev3Losses']) * 2)
        feature_df.update(feature_df_new['home_team_prev_3_points_percentage'])
    feature_df.drop(columns=['homeTeamPrev3Wins', 'homeTeamPrev3OTLs', 'homeTeamPrev3Losses'], inplace=True)

    # calculate goals for in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 3 goals for ...')
    feature_df = prevN_gfpg(3, feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col)

    # calculate goals for in the previous 3 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 3 goals for ...')
    feature_df = prevN_gfpg(3, feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col)

        # calculate goals for in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 7 goals for ...')
    feature_df = prevN_gfpg(7, feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col)

    # calculate goals for in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 7 goals for ...')
    feature_df = prevN_gfpg(7, feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col)

    # # calculate goals for for the home team in all matchups
    # if debug: print('\t\t... [feature_creation] home goals for ...')
    # feature_df = season_gfpg(feature_df, backfill, cons.home_team_goals_for_col, cons.home_team_name_col)

    # # calculate goals for for the away team in all matchups
    # if debug: print('\t\t... [feature_creation] away goals for ...')
    # feature_df = season_gfpg(feature_df, backfill, cons.away_team_goals_for_col, cons.away_team_name_col)

    return feature_df


def sched_feature_add(feature_df):

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    feature_df[cons.starttime_utc_col] = pd.to_datetime(feature_df[cons.starttime_utc_col]).dt.tz_convert('US/Eastern')
    # feature_df[cons.game_year_col] = feature_df[cons.starttime_utc_col].dt.year
    # feature_df[cons.game_month_col] = feature_df[cons.starttime_utc_col].dt.month
    feature_df[cons.game_date_col] = feature_df[cons.starttime_utc_col].dt.date
    feature_df[cons.game_time_col] = feature_df[cons.starttime_utc_col].dt.hour * 60 + feature_df[cons.starttime_utc_col].dt.minute
    feature_df.drop(columns=[cons.starttime_utc_col], inplace=True)

    return feature_df


def sched_feature_map(feature_df):

    # add feature columns for team ids
    # feature_df[cons.home_team_id_col] = feature_df[cons.home_team_name_col].map(cons.team_id_map)
    # feature_df[cons.away_team_id_col] = feature_df[cons.away_team_name_col].map(cons.team_id_map)

    # encode the categorical features using ordinal encoding
    # encoder = OrdinalEncoder()
    # feature_df[cons.venue_timezone_col] = encoder.fit_transform(feature_df[[cons.venue_timezone_col]]).astype(int)
    # feature_df[cons.venue_col] = encoder.fit_transform(feature_df[[cons.venue_col]]).astype(int)

    # encode the target variable 'lastPeriod' using a mapping of the period types to integers
    # feature_df[cons.last_period_col] = feature_df.loc[
    #     feature_df[cons.last_period_col].notna(), cons.last_period_col].map(cons.last_period_map).astype(int)

    return feature_df


def feature_type_prep(feature_df):

    feature_df[cons.season_name_col] = feature_df[cons.season_name_col].astype(str)

    return feature_df


if __name__ == "__main__":

    feature_df = create_feature_df()

    ######################
    # create season schedule dataframe for inputted season
    # create_season_df('20252026', from_csv=False, to_csv=True, debug=True)
    # data_df = pd.read_csv('data_df.csv')
    # make_predictions(data_df)
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