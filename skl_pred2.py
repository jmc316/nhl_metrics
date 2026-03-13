import os
import random
import numpy as np
from tabulate import tabulate
import constants as cons
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from skl_utils import days_since_last_played, prevN_result, prevN_gfpg, season_gfpg, season_result
import skl_utils as sku
import utils as ut


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

    oob_list, mse_list, rsq_list = [], [], []

    # make predictions for first scheduled game day after completed games and add to filtered dataframe
    feature_df_filt = make_predictions(feature_df_filt, oob_list, mse_list, rsq_list)

    # re-create feature dataframe with added predictions and features
    feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    # loop through remaining scheduled game days and repeat following process:
    #   1. add dependent features to scheduled rows
    #   2. make predictions for scheduled rows 
    #   3. re-create feature dataframe with added predictions and features
    for next_game_date in feature_df.loc[feature_df[cons.away_team_score_col].isna(), cons.game_date_col].unique():
        # print(f'\tPredicting games for {next_game_date.strftime("%Y-%m-%d")}...')
        feature_df_filt = feature_df[feature_df[cons.game_date_col] <= next_game_date]
        feature_df_filt = dependent_feature_add(feature_df_filt, backfill=False, debug=False)
        feature_df_filt = make_predictions(feature_df_filt, oob_list, mse_list, rsq_list)
        feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    print()
    print(f'Average Out-of-Bag Score: {np.mean(oob_list)}')
    print(f'Average Mean Squared Error: {np.mean(mse_list)}')
    print(f'Average R-squared: {np.mean(rsq_list)}')

    # feature_df.to_csv('results.csv', index=False)

    print('Season predictions complete.')

    return feature_df


def make_predictions(data_df, oob_list, mse_list, rsq_list):

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


    model = ut.init_model()

    model.fit(x_train_df.values, y_train_df.values)

    oob_score = model.oob_score_
    # print(f'Out-of-Bag Score: {oob_score}')
    oob_list.append(oob_score)

    trainset_predictions = model.predict(x_train_df.values)

    mse = mean_squared_error(y_train_df.values, trainset_predictions)
    # print(f'Mean Squared Error: {mse}')
    mse_list.append(mse)

    r2 = r2_score(y_train_df.values, trainset_predictions)
    # print(f'R-squared: {r2}')
    rsq_list.append(r2)

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

    # calculate days since last played game for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.home_team_days_since_last_game_col, cons.home_team_name_col)

    # calculate days since last played game for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.away_team_days_since_last_game_col, cons.away_team_name_col)

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


def assign_game_points(season_results, to_csv=False):

    print('\tAssigning game points...')

    # assign points to each team based on the predicted scores and last period type
    # (2 points for a win in regulation, 1 point for an OT/SO loss, 0 points for a regulation loss)
    season_results[cons.home_team_points_col] = np.where(
        season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col], 2, np.where(
            season_results[cons.last_period_col] != 'REG', 1, np.where(
                season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col], 0, 0)))
    season_results[cons.away_team_points_col] = np.where(
        season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col], 2, np.where(
            season_results[cons.last_period_col] != 'REG', 1, np.where(
                season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col], 0, 0)))

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        col_order = [cons.game_id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col,
                     cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col,
                     cons.last_period_col, cons.home_team_points_col, cons.away_team_points_col]
        season_results[col_order].to_csv(cons.output_folder + cons.season_sched_pred_filename, index=False)

    return season_results


def home_away_accumulation(home_df, away_df, stat_col, keep_segregated_cols=False, debug=False):

    # fill in missing teams in the homeTeam dataframe
    for team in cons.team_colors.keys():
        if team not in home_df[cons.home_team_name_col].values:
            if debug: print(f'\t... Adding {team} to homeTeam{stat_col} dataframe with 0 {stat_col} ...')
            home_df = pd.concat([home_df, pd.DataFrame({cons.home_team_name_col: [team], f'homeTeam{stat_col}': [0]})], ignore_index=True)

    # fill in missing teams in the awayTeam dataframe
    for team in cons.team_colors.keys():
        if team not in away_df[cons.away_team_name_col].values:
            if debug: print(f'\t... Adding {team} to awayTeam{stat_col} dataframe with 0 {stat_col} ...')
            away_df = pd.concat([away_df, pd.DataFrame({cons.away_team_name_col: [team], f'awayTeam{stat_col}': [0]})], ignore_index=True)

    # merge the home and away dataframes on the team name column, then sum the home and away stats to get the total stat for each team
    merged_df = pd.merge(home_df, away_df, left_on=cons.home_team_name_col, right_on=cons.away_team_name_col)
    merged_df[f'total{stat_col}'] = merged_df[f'homeTeam{stat_col}'] + merged_df[f'awayTeam{stat_col}']
    if not keep_segregated_cols:
        merged_df.drop(columns=[f'homeTeam{stat_col}', f'awayTeam{stat_col}'], inplace=True)
    else:
        merged_df.rename(columns={f'homeTeam{stat_col}': f'totalHome{stat_col}',
                                  f'awayTeam{stat_col}': f'totalAway{stat_col}'}, inplace=True)
    merged_df.drop(columns=[cons.away_team_name_col], inplace=True)
    merged_df.rename(columns={cons.home_team_name_col: cons.team_name_col}, inplace=True)
    merged_df.sort_values(by=f'total{stat_col}', ascending=False, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def generate_final_standings(season_results, to_csv=False, load_csv=False):

    # if the final standings CSV file already exists in the output folder, load it instead of re-generating the standings from the season results dataframe
    if load_csv:
        season_results = pd.read_csv(cons.output_folder + cons.season_sched_pred_filename)

    print('\tGenerating final standings...')

    # filter the final schedule on the current season
    season_results[cons.season_name_col] = season_results[cons.season_name_col].astype(str)
    current_season = str(int(max(season_results[cons.season_name_col].str[4:]))-1) + max(season_results[cons.season_name_col].str[4:])
    season_results = season_results.loc[season_results[cons.season_name_col] == current_season]

    # calculate games played for each team
    home_games = season_results.groupby(cons.home_team_name_col)[cons.home_team_score_col].count().reset_index(name=cons.home_team_games_col)
    away_games = season_results.groupby(cons.away_team_name_col)[cons.away_team_score_col].count().reset_index(name=cons.away_team_games_col)
    games_played_df = home_away_accumulation(home_games, away_games, 'Games')

    # calculate total points for each team
    home_points = season_results.groupby(cons.home_team_name_col)[cons.home_team_points_col].sum().reset_index()
    away_points = season_results.groupby(cons.away_team_name_col)[cons.away_team_points_col].sum().reset_index()
    points_df = home_away_accumulation(home_points, away_points, 'Points')

    # calculate total wins for each team
    home_wins = season_results.loc[season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col]].groupby(
        cons.home_team_name_col).size().reset_index(name=cons.home_team_wins_col)
    away_wins = season_results.loc[season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col]].groupby(
        cons.away_team_name_col).size().reset_index(name=cons.away_team_wins_col)
    wins_df = home_away_accumulation(home_wins, away_wins, 'Wins', keep_segregated_cols=True)

    # calculate total losses for each team
    home_losses = season_results.loc[(season_results[cons.last_period_col]==0) &
                                     (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                         cons.home_team_name_col).size().reset_index(name=cons.home_team_losses_col)
    away_losses = season_results.loc[(season_results[cons.last_period_col]==0) &
                                     (season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col])].groupby(
                                         cons.away_team_name_col).size().reset_index(name=cons.away_team_losses_col)
    losses_df = home_away_accumulation(home_losses, away_losses, 'Losses', keep_segregated_cols=True)

    # calculate total OT/SO losses for each team
    home_otls = season_results.loc[(season_results[cons.last_period_col]=='OT') &
                                   (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                       cons.home_team_name_col).size().reset_index(name=cons.home_team_otls_col)
    away_otls = season_results.loc[(season_results[cons.last_period_col]=='OT') &
                                   (season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col])].groupby(
                                       cons.away_team_name_col).size().reset_index(name=cons.away_team_otls_col)
    otls_df = home_away_accumulation(home_otls, away_otls, 'OTLs', keep_segregated_cols=True)

    # calculate total regulation wins for each team (used for tiebreakers in the standings)
    home_reg_wins = season_results.loc[(season_results[cons.last_period_col]=='REG') &
                                       (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                           cons.home_team_name_col).size().reset_index(name=cons.home_team_reg_wins_col)
    away_reg_wins = season_results.loc[(season_results[cons.last_period_col]=='REG') &
                                       (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                           cons.away_team_name_col).size().reset_index(name=cons.away_team_reg_wins_col)
    reg_wins_df = home_away_accumulation(home_reg_wins, away_reg_wins, 'RegWins')

    # calculate total regulation/OT wins for each team (used for tiebreakers in the standings)
    home_reg_ot_wins = season_results.loc[(season_results[cons.last_period_col]!='SO') &
                                          (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                              cons.home_team_name_col).size().reset_index(name=cons.home_team_reg_ot_wins_col)
    away_reg_ot_wins = season_results.loc[(season_results[cons.last_period_col]!='SO') &
                                          (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                              cons.away_team_name_col).size().reset_index(name=cons.away_team_reg_ot_wins_col)
    reg_ot_wins_df = home_away_accumulation(home_reg_ot_wins, away_reg_ot_wins, 'RegOTWins')

    # calculate total shootout wins for each team
    home_so_wins = season_results.loc[(season_results[cons.last_period_col]=='SO') &
                                      (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                          cons.home_team_name_col).size().reset_index(name=cons.home_team_so_wins_col)
    away_so_wins = season_results.loc[(season_results[cons.last_period_col]=='SO') &
                                      (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                          cons.away_team_name_col).size().reset_index(name=cons.away_team_so_wins_col)
    so_wins_df = home_away_accumulation(home_so_wins, away_so_wins, 'SOWins')

    # calculate total shootout losses for each team
    home_so_losses = season_results.loc[(season_results[cons.last_period_col]=='SO') &
                                        (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                            cons.home_team_name_col).size().reset_index(name=cons.home_team_so_losses_col)
    away_so_losses = season_results.loc[(season_results[cons.last_period_col]=='SO') &
                                        (season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col])].groupby(
                                            cons.away_team_name_col).size().reset_index(name=cons.away_team_so_losses_col)
    so_losses_df = home_away_accumulation(home_so_losses, away_so_losses, 'SOLosses')

    # calculate total Goals For and Goals Against for each team (used for tiebreakers in the standings)
    home_goals_for = season_results.groupby(
        cons.home_team_name_col)[cons.home_team_score_col].sum().reset_index(name=cons.home_team_goals_for_col)
    away_goals_for = season_results.groupby(
        cons.away_team_name_col)[cons.away_team_score_col].sum().reset_index(name=cons.away_team_goals_for_col)
    goals_for_df = home_away_accumulation(home_goals_for, away_goals_for, 'GoalsFor')
    home_goals_against = season_results.groupby(
        cons.home_team_name_col)[cons.away_team_score_col].sum().reset_index(name=cons.home_team_goals_against_col)
    away_goals_against = season_results.groupby(
        cons.away_team_name_col)[cons.home_team_score_col].sum().reset_index(name=cons.away_team_goals_against_col)
    goals_against_df = home_away_accumulation(home_goals_against, away_goals_against, 'GoalsAgainst')

    # merge the points, wins, losses, and OT/SO losses dataframes together to create the final standings dataframe
    final_standings = pd.merge(points_df, wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, losses_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, otls_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, reg_wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, reg_ot_wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, goals_for_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, goals_against_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, so_wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, so_losses_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, games_played_df, on=cons.team_name_col)

    # free memory by deleting the intermediate dataframes that are no longer needed
    del points_df, wins_df, losses_df, otls_df, reg_wins_df, reg_ot_wins_df, goals_for_df, goals_against_df, so_wins_df, so_losses_df, games_played_df

    # calculate functional columns
    final_standings[cons.goal_diff_col] = final_standings[cons.total_goals_for_col] - final_standings[cons.total_goals_against_col]
    final_standings[cons.points_percentage_col] = final_standings[cons.total_points_col] / (final_standings[cons.total_games_col] * 2)

    # load individual team info to merge with final standings for wildcard setup
    team_info_df = ut.team_info()
    final_standings = pd.merge(final_standings, team_info_df[[cons.team_name_col, cons.division_name_col, cons.conference_name_col]], on=cons.team_name_col)

    # assign divisionSeed, conferenceSeedbased on total points and tiebreakers within each division, conference
    final_standings.sort_values(by=[cons.division_name_col] + cons.tiebreaker_cols, ascending=[True] + [False]*len(cons.tiebreaker_cols), inplace=True)
    final_standings[cons.division_seed_col] = [val%8+1 for val in list(final_standings.reset_index(drop=True).reset_index().index)]
    final_standings.sort_values(by=[cons.conference_name_col] + cons.tiebreaker_cols, ascending=[True] + [False]*len(cons.tiebreaker_cols), inplace=True)
    final_standings[cons.conference_seed_col] = [val%16+1 for val in list(final_standings.reset_index(drop=True).reset_index().index)]
    final_standings.sort_values(by=cons.tiebreaker_cols, ascending=[False]*len(cons.tiebreaker_cols), inplace=True)

    # calculate playoff seed by including the top three teams from every division and then the top two remaining teams from each conference
    # division playoff spots are labelled as 'div_x', where x=[1, 2, 3] represents the division seed
    final_standings[cons.playoff_seed_col] = np.where(final_standings[cons.division_seed_col] <= 3, 'div_' + final_standings[cons.division_seed_col].astype(str), cons.missed_val)
    # wildcard playoff spots are labelled as 'wc_x', where x=[1, 2] represents the wildcard seed; non-playoff teams are labelled as 'Missed'
    final_standings.loc[final_standings[cons.playoff_seed_col] == cons.missed_val, cons.playoff_seed_col] = np.where(
        final_standings.loc[final_standings[cons.playoff_seed_col] == cons.missed_val].groupby(
            cons.conference_name_col)[cons.total_points_col].rank(method='min', ascending=False) <= 2, 
        'wc_' + final_standings.loc[final_standings[cons.playoff_seed_col] == cons.missed_val].groupby(
            cons.conference_name_col)[cons.total_points_col].rank(method='min', ascending=False).astype(int).astype(str), cons.missed_val)

    # reorder the columns and sort the final standings by the tiebreaker columns in descending order
    final_standings = final_standings[cons.final_standings_col_order]
    final_standings.sort_values(by=cons.tiebreaker_cols, ascending=[False]*len(cons.tiebreaker_cols), inplace=True)

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        final_standings.to_csv(cons.output_folder + cons.final_standings_filename, index=False)

    return final_standings


def playoff_spot_predictions(n=100):

    count_df = pd.DataFrame(columns=[cons.team_name_col, cons.div_1_val, cons.div_2_val, cons.div_3_val, cons.wc_1_val, cons.wc_2_val, cons.missed_val])
    count_df[cons.team_name_col] = list(cons.team_colors.keys())
    count_df.fillna(0, inplace=True)

    for i in range(n):
        print(f'\nSimulation {i+1} of {n}...')
        season_results = create_feature_df()
        season_results_points = assign_game_points(season_results)
        final_standings = generate_final_standings(season_results_points)

        # count the number of times each team finishes in each playoff seed across all simulations
        for _, row in final_standings.iterrows():
            count_df.loc[count_df[cons.team_name_col] == row[cons.team_name_col], row[cons.playoff_seed_col]] += 1

    count_df[f'{cons.div_1_val}_%'] = count_df[cons.div_1_val] / n * 100
    count_df[f'{cons.div_2_val}_%'] = count_df[cons.div_2_val] / n * 100
    count_df[f'{cons.div_3_val}_%'] = count_df[cons.div_3_val] / n * 100
    count_df[f'{cons.wc_1_val}_%'] = count_df[cons.wc_1_val] / n * 100
    count_df[f'{cons.wc_2_val}_%'] = count_df[cons.wc_2_val] / n * 100
    count_df[f'{cons.missed_val}_%'] = count_df[cons.missed_val] / n * 100
    count_df[f'{cons.playoff_per_col}'] = (n - count_df[cons.missed_val]) / n * 100

    playoff_probabilities_printer(count_df)
    return count_df


def playoff_probabilities_printer(count_df):

    percent_cols = [cons.team_name_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%', f'{cons.missed_val}_%', f'{cons.playoff_per_col}']

    count_df = count_df.sort_values(by=[cons.playoff_per_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%'], ascending=False).reset_index(drop=True)[percent_cols]
    print('\nPlayoff Spot Probabilities:')

    print(tabulate(count_df, headers='keys', tablefmt='grid'))


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

    pass