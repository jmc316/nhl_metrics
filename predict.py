import os
import playoffs

import numpy as np
import pandas as pd
import features as ft
import constants as cons
import nhl_utils as nhlu
import skl_utils as sklu
import nhl_client as nhlc

from datetime import datetime as dt
from file_utils import csvLoad, csvSave
from analyze import game_result_comparison


def predict_season(to_csv, set_model_random_state, today_dt):

    # create the schedule dataframe from all seasons and games
    sched_df = create_df_set()

    # if there is a manual today_dt, Nullify all scores for games on or after the date
    if today_dt != dt.now().date().strftime(cons.date_format_yyyy_mm_dd):
        today_date_utc = pd.to_datetime(today_dt).tz_localize('EST').tz_convert('UTC')
        sched_df.loc[pd.to_datetime(sched_df[cons.starttime_utc_col]) >= today_date_utc, cons.predict_cols] = np.nan

    # add features that are not dependent on the prediction set results
    feature_df = ft.datetime_feature_add(sched_df)

    cons.last_actual_game_date = sched_df.loc[sched_df[cons.away_team_score_col].notna(), cons.game_date_col].max()

    # initialize empty metrics lists to store the out-of-bag score, mean squared error, and R-squared for each prediction iteration
    oob_list, mse_list, rsq_list = [], [], []

    # next game date is first date to predict games for, create df with games that don't need predicting
    next_game_date = feature_df.loc[feature_df[cons.away_team_score_col].isna(), cons.game_date_col].min()
    feature_df_filt = feature_df[feature_df[cons.game_date_col] <= next_game_date]

    # check to see if historical feature data for seasons in the input data already exists
    # if complete historical data exists, change next_game_date to first date with missing features
    feature_df_filt_load = pd.DataFrame()
    backfill_bool = False
    seasons_to_save = []
    for season in feature_df_filt[cons.season_name_col].unique():
        if feature_df_filt.loc[feature_df_filt[cons.season_name_col] == season, cons.last_period_col].notna().all():
            if os.path.exists(f'{cons.season_feature_sets_folder}{cons.feature_data_filename.format(season=season)}'):
                print(f'Historical feature data for {season[:4]}-{season[4:]} season already exists. Loading from file...')
                season_feature_df = csvLoad(cons.season_feature_sets_folder, f'{cons.feature_data_filename.format(season=season)}')
                feature_df_filt_load = pd.concat([feature_df_filt_load, season_feature_df], ignore_index=True)
                continue

        print(f'Historical feature data for {season[:4]}-{season[4:]} season does not exist, creating features... ')
        feature_df_filt_load = pd.concat([feature_df_filt_load, feature_df_filt[feature_df_filt[cons.season_name_col] == season]], ignore_index=True)
        backfill_bool = True
        seasons_to_save.append(season)

    # add dependent features to filtered dataframe
    feature_df_filt = ft.dependent_feature_add(feature_df_filt_load, backfill=backfill_bool, debug=False)

    # save the feature dataframe with added features for the first game day to predict to a CSV file for future use; this will allow us to skip the feature engineering process for this game day in future runs and go straight to making predictions
    if to_csv:
        for season in seasons_to_save:
            if feature_df_filt.loc[feature_df_filt[cons.season_name_col] == season, cons.last_period_col].notna().all():
                print(f'Saving feature data for {season[:4]}-{season[4:]} season to CSV file...')
                csvSave(feature_df_filt.loc[feature_df_filt[cons.season_name_col] == season], cons.season_feature_sets_folder, f'{cons.feature_data_filename.format(season=season)}')

    # make predictions for first scheduled game day after completed games and add to filtered dataframe
    print(f'\tPredicting games for {next_game_date.strftime("%Y-%m-%d")}...')
    feature_df_filt = sklu.make_predictions(feature_df_filt, oob_list, mse_list, rsq_list, set_model_random_state, today_dt, load_model=False, save_model=True)

    # re-create feature dataframe with added predictions and features
    feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    # loop through remaining scheduled game days and repeat following process:
    #   1. add dependent features to scheduled rows
    #   2. make predictions for scheduled rows 
    #   3. re-create feature dataframe with added predictions and features
    for next_game_date in feature_df.loc[feature_df[cons.away_team_score_col].isna(), cons.game_date_col].unique():
        print(f'\tPredicting games for {next_game_date.strftime("%Y-%m-%d")}...')
        feature_df_filt = feature_df[feature_df[cons.game_date_col] <= next_game_date]
        feature_df_filt = ft.dependent_feature_add(feature_df_filt, backfill=False, debug=False)
        feature_df_filt = sklu.make_predictions(feature_df_filt, oob_list, mse_list, rsq_list, set_model_random_state, today_dt, load_model=True, save_model=False)
        feature_df = pd.concat([feature_df_filt, feature_df.loc[feature_df[cons.game_date_col] > next_game_date]], ignore_index=True)

    print()
    print(f'Average Out-of-Bag Score: {np.mean(oob_list)}')
    print(f'Average Mean Squared Error: {np.mean(mse_list)}')
    print(f'Average R-squared: {np.mean(rsq_list)}')
    print('Season predictions complete.\n')

    if to_csv:
        print('Saving season predictions to CSV file...')
        csvSave(feature_df, cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))

    return feature_df


def create_df_set():

    # a list of schedule files that have already been generated
    season_sched_list = [file for file in os.listdir(cons.season_sched_folder) if file.endswith(cons.season_sched_filename.format(season=''))]

    # initialize empty dataframe to store the season schedule data
    sched_df = pd.DataFrame()

    # loop through each schedule file and concatenate it to the season schedule dataframe;
    # if there are no schedule files, create the season schedule dataframe by fetching the data from the API
    for filename in season_sched_list:
        if sched_df.empty:
            sched_df = create_season_df(filename[:8], from_csv=True, to_csv=False)
        else:
            sched_df = pd.concat([sched_df, create_season_df(filename[:8], from_csv=True, to_csv=False)], ignore_index=True)

    # check if the current season schedule needs to be updated
    sched_df = schedule_update(sched_df)

    return sched_df


def schedule_update(sched_df):

    # today's date and the date of the first unplayed game in the schedule dataframe
    today_dt = dt.now().date()
    stored_dt = pd.to_datetime(min(sched_df.loc[sched_df[cons.last_period_col].isna(), cons.starttime_utc_col])).date()

    # if the date of the first unplayed game is in the past, the schedule dataframe is missing some data
    if today_dt > stored_dt:
        print('\t\tCurrent season schedule data requires update from API...')

        missing_sched = pd.DataFrame()

        # add schedule data one date at a time
        for game_date in pd.date_range(start=stored_dt, end=(today_dt - pd.Timedelta(days=1)), freq='D'):
            print(f'\t\t\t... {game_date.strftime("%Y-%m-%d")} ...')
            missing_sched = pd.concat([missing_sched, nhlc.get_sched_data(game_date, 0)], ignore_index=True)

        sched_df = sched_df.loc[~sched_df[cons.game_id_col].isin(missing_sched[cons.game_id_col])]
        sched_df = pd.concat([sched_df, missing_sched], ignore_index=True)
        sched_df.sort_values(by=cons.starttime_utc_col, inplace=True)
        sched_df.reset_index(drop=True, inplace=True)

        sched_df[cons.season_name_col] = sched_df[cons.season_name_col].astype(str)

        for col in sched_df.columns:
            if isinstance(sched_df[col], np.int64):
                sched_df[col] = sched_df[col].astype(int)

        print('\t\tSaving updated season schedule to CSV file...')
        csvSave(sched_df.loc[sched_df[cons.season_name_col]==max(sched_df[cons.season_name_col])],
                cons.season_sched_folder, cons.season_sched_filename.format(season=max(sched_df[cons.season_name_col])))

    return sched_df


def create_season_df(season_name, from_csv=True, to_csv=False, debug=False):

    print(f'\tRetrieving {season_name[:4]}-{season_name[4:]} NHL season schedule...')

    season_filename = cons.season_sched_filename.format(season=season_name)

    # if the season schedule CSV file already exists in the output folder, load it instead of fetching the data from the API again
    if from_csv and season_filename in os.listdir(cons.season_sched_folder):
        if debug: print('\tSeason schedule CSV file already exists. Loading from file...')
        return csvLoad(cons.season_sched_folder, season_filename)
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
            season_sched = pd.concat([season_sched, nhlc.get_sched_data(week, dow)], ignore_index=True)

    # save the season schedule to a CSV file for future use
    if to_csv:
        if debug: print('\tSaving season schedule to CSV file...')
        csvSave(season_sched, cons.season_sched_folder, cons.season_sched_filename.format(season=season_name))

    return season_sched


def playoff_spot_predictions(today_dt, n=100, to_csv=True):

    # initialize dataframe to store the count of playoff seeds for each team across all simulations; this will be used to calculate the probabilities of each team making the playoffs and their likely seed
    count_df = pd.DataFrame(columns=[cons.team_name_col, cons.div_1_val, cons.div_2_val, cons.div_3_val, cons.wc_1_val, cons.wc_2_val, cons.missed_val, cons.make_r2_val, cons.make_r3_val, cons.make_cup_final_val, cons.win_cup_val])
    count_df[cons.team_name_col] = list(cons.team_info.keys())
    count_df.fillna(0, inplace=True)

    # run n simulations of the season and count the number of times each team finishes in each playoff seed across all simulations;
    # this will allow us to calculate the probabilities of each team making the playoffs and their likely seed
    for i in range(n):
        print(f'\nSimulation {i+1} of {n}...')
        season_results_df = predict_season(False, False, today_dt)
        season_results_points = nhlu.assign_game_points(season_results_df)
        final_standings_df = nhlu.generate_final_standings(season_results_points, today_dt)
        _, playoff_matchups = playoffs.playoff_tree_predictions(season_results_df, final_standings_df, False, today_dt, to_csv=False)

        # count the number of times each team finishes in each playoff seed across all simulations
        for _, row in final_standings_df.iterrows():
            count_df.loc[count_df[cons.team_name_col] == row[cons.team_name_col], row[cons.playoff_seed_col]] += 1

        # count the number of times each playoff team advances to each round across all simulations
        for round in playoff_matchups.keys():
            for _, matchup in playoff_matchups[round].items():
                if matchup.get_series_winner() is not None:
                    if round < 3:
                        count_df.loc[count_df[cons.team_name_col] == matchup.get_series_winner(), f'make_round_{round+1}'] += 1
                    elif round == 3:
                        count_df.loc[count_df[cons.team_name_col] == matchup.get_series_winner(), cons.make_cup_final_val] += 1
                    elif round == 4:
                        count_df.loc[count_df[cons.team_name_col] == matchup.get_series_winner(), cons.win_cup_val] += 1

    # calculate the probabilities of each team making the playoffs and their likely seed based on the counts across all simulations
    count_df[f'{cons.div_1_val}_%'] = count_df[cons.div_1_val] / n * 100
    count_df[f'{cons.div_2_val}_%'] = count_df[cons.div_2_val] / n * 100
    count_df[f'{cons.div_3_val}_%'] = count_df[cons.div_3_val] / n * 100
    count_df[f'{cons.wc_1_val}_%'] = count_df[cons.wc_1_val] / n * 100
    count_df[f'{cons.wc_2_val}_%'] = count_df[cons.wc_2_val] / n * 100
    count_df[f'{cons.missed_val}_%'] = count_df[cons.missed_val] / n * 100
    count_df[f'{cons.playoff_per_col}'] = (n - count_df[cons.missed_val]) / n * 100
    count_df[f'{cons.make_r2_val}_%'] = count_df[cons.make_r2_val] / n * 100
    count_df[f'{cons.make_r3_val}_%'] = count_df[cons.make_r3_val] / n * 100
    count_df[f'{cons.make_cup_final_val}_%'] = count_df[cons.make_cup_final_val] / n * 100
    count_df[f'{cons.win_cup_val}_%'] = count_df[cons.win_cup_val] / n * 100

    nhlu.playoff_probabilities_printer(count_df)

    if to_csv:
        print(f'\nSaving playoff spot predictions to CSV file...')
        csvSave(count_df, cons.season_pred_folder.format(date=today_dt), cons.season_results_prob_filename.format(n=n, date=today_dt))

    return count_df


if __name__ == "__main__":
    import playoffs

    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
    # today_dt = '2025-10-01' # beginning of 20252026 season
    # today_dt = '2026-02-24' # end of Olympic break

    ######################
    # create season schedule dataframe for inputted seasons
    # season_names = ['20212022', '20222023', '20232024', '20242025', '20252026']
    # for season in season_names:
    #     create_season_df(season, from_csv=False, to_csv=True, debug=True)

    ######################
    # create one set of predictions
    feature_df = predict_season(to_csv=True, set_model_random_state=True, today_dt=today_dt)
    game_result_comparison(feature_df)
    season_results_df = nhlu.generate_final_standings(nhlu.assign_game_points(feature_df), today_dt, to_csv=True)
    nhlu.nhl_team_standings(season_results_df)
    playoff_results_df = playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt)

    ######################
    # create playoff spot predictions for current season based on n simulations
    # playoff_spot_predictions(today_dt, n=50)