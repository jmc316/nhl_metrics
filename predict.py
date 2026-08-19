import os
import playoffs
import datetime

import numpy as np
import pandas as pd
import constants as cons
import utils.nhl_utils as nhlu
import utils.skl_utils as sklu
import nhl_client as nhlc

from features.features import feature_data_load
from zoneinfo import ZoneInfo
from datetime import datetime as dt
from utils.file_utils import csvLoad, csvSave
from playoff_probability import display_playoff_probability


def predict_season(to_csv, set_model_state, today_dt):

    # load all of the feature data with all actuals
    feature_df = feature_data_load()

    # running for past predictions, pre-preprocess the feature data
    if today_dt < dt.now().date().strftime(cons.date_format_yyyy_mm_dd):

        # get the season name corresponding to the inputted today_dt
        season_name = max(feature_df.loc[feature_df[cons.starttime_est_col].dt.date <= dt.strptime(today_dt, "%Y-%m-%d").date(),
                                         cons.season_name_col])

        # remove all seasons after this season from the feature dataframe
        feature_df = feature_df.loc[feature_df[cons.season_name_col] <= season_name]

        # if the today_dt is in the regular season, remove all scheduled playoff games for this season
        if not feature_df.loc[(feature_df[cons.game_type_col]==2) & (feature_df[cons.season_name_col]==season_name) &
                              (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date())].empty:
            feature_df = feature_df.loc[~((feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name))]
        # if the today_dt is in the playoffs, remove all scheduled playoff games for any rounds after the current one
        elif not feature_df.loc[(feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name) &
                              (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date())].empty:
            feature_df = feature_df.loc[~((feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name) &
                                          (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date()))]

        # nullify the results of all games beyond the today_dt
        feature_df.loc[feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date(),
                       [cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col,
                        cons.home_team_win_col, cons.win_prob_col.format(team='home'), cons.win_prob_col.format(team='away')]] = np.nan

    # pre-process the feature data
    processed_df, feature_list = sklu.preprocess_feature_data(feature_df)

    # running for past predictions, need to train a model on the past data state
    if today_dt < dt.now().date().strftime(cons.date_format_yyyy_mm_dd):
        # train a model on the past data state
        model = sklu.model_train(processed_df, feature_list, save_model=False)
        pred_df, model = sklu.model_inference(processed_df, feature_list, today_dt, model=model)

    else:
        # make predictions from the start date to the end of the schedule, and add the predictions to the feature dataframe
        pred_df, model = sklu.model_inference(processed_df, feature_list, today_dt)

    # return feature_df with the win probability columns added
    feature_df.update(pred_df[['homeTeamWin', 'homeWinProb', 'awayWinProb']])

    # display the predictions for the first date of predictions, and explain the predictions using SHAP values
    print_dt = min(feature_df.loc[feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date(),
                                  cons.starttime_est_col].dt.date)
    first_pred_dt_df = feature_df.loc[feature_df[cons.starttime_est_col].dt.date==print_dt]

    print(f'\nPredicted game results for {print_dt.strftime("%Y-%m-%d")}:')
    for idx, row in first_pred_dt_df.iterrows():

        home_team = cons.team_name_addrev_map[row[cons.home_team_name_col]]
        away_team = cons.team_name_addrev_map[row[cons.away_team_name_col]]
        ot_str = ' (OT)' if row[cons.last_period_col]=='OT' else ''

        if row[cons.win_prob_col.format(team='home')] > row[cons.win_prob_col.format(team='away')]:
            print(f"\t{away_team.lower()} {(row[cons.win_prob_col.format(team='away')]*100):.2f} at {home_team} {(row[cons.win_prob_col.format(team='home')]*100):.2f}{ot_str}")
        else:
            print(f"\t{away_team} {(row[cons.win_prob_col.format(team='away')]*100):.2f} at {home_team.lower()} {(row[cons.win_prob_col.format(team='home')]*100):.2f}{ot_str}")

        sklu.explain_predictions(pred_df.iloc[[idx]][feature_list],
                                model, home_team, away_team,
                                print_dt, today_dt)

    if to_csv:
        print('Saving season predictions to CSV file...')
        csvSave(pred_df, cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))
    
    return feature_df


def create_df_set(today_dt):

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

    return sched_df


def game_results_update(last_act_dt):

    # today's date in EST timezone
    today_dt = dt.now().date()

    # initialize empty dataframe to store any missing schedule data that needs to be updated with actual results
    missing_sched_df = pd.DataFrame()

    # if the date of the first unplayed game is in the past, the schedule dataframe is missing some data
    if today_dt > last_act_dt:
        print('\tSchedule data actuals update required...')

        # the last possible day of the season, no need to check games beyond this date
        cur_season_enddt = pd.to_datetime(f'{cons.season_enddt}-{str(today_dt.year)}').date()

        # add schedule actuals data one date at a time
        for game_date in pd.date_range(start=last_act_dt, end=min(cur_season_enddt, today_dt - pd.Timedelta(days=1)), freq='D'):
            print(f'\t\t... {game_date.strftime("%Y-%m-%d")} ...')
            new_data = nhlc.get_sched_data(game_date, 0)

            # if there were no games on this date, check the next date
            if new_data.empty:
                continue

            # if the games on this date have not yetr been played, there are no more games to check for actuals, so break out of the loop
            if new_data.loc[new_data[cons.last_period_col].notna()].empty:
                break

            # there were games on this date found with completed final scores, so add them to the missing schedule dataframe
            missing_sched_df = pd.concat([missing_sched_df, new_data], ignore_index=True)

        if missing_sched_df.empty:
            print('\tNo missing actuals data found\n')
            return missing_sched_df
        
        missing_sched_df[cons.starttime_utc_col] = pd.to_datetime(missing_sched_df[cons.starttime_utc_col], format='ISO8601')
        missing_sched_df[cons.starttime_est_col] = missing_sched_df[cons.starttime_utc_col].dt.tz_convert('EST')
        missing_sched_df[cons.season_name_col] = missing_sched_df[cons.season_name_col].astype(str)

        missing_sched_df.sort_values(by=cons.starttime_utc_col, inplace=True)
        missing_sched_df.reset_index(drop=True, inplace=True)

        for col in missing_sched_df.columns:
            if isinstance(missing_sched_df[col], np.int64):
                missing_sched_df[col] = missing_sched_df[col].astype(int)

    return missing_sched_df


def schedule_update():

    ################################################
    ### 1. LOAD THE SAVED SCHEDULE DATA WITH ACTUALS
    ################################################
    print('\nLoading saved season schedule data...')

    # a list of schedule files that have already been generated
    season_sched_list = [file for file in os.listdir(cons.season_sched_folder) if file.endswith(cons.season_sched_filename.format(season=''))]

    # initialize empty dataframe to store the saved season schedule data
    sched_df_saved = pd.DataFrame()

    # loop through each schedule file and concatenate it to the season schedule dataframe
    for filename in season_sched_list:
        if sched_df_saved.empty:
            sched_df_saved = load_season_df(filename[:8])
        else:
            sched_df_saved = pd.concat([sched_df_saved, load_season_df(filename[:8])], ignore_index=True)

    # the day after the last played game in the schedule dataframe; this is the first day that needs to be updated with actual results
    sched_last_act_dttm = pd.to_datetime(max(sched_df_saved.loc[sched_df_saved[cons.last_period_col].notna(), cons.starttime_est_col]), format='ISO8601') + pd.Timedelta(days=1)
    sched_last_act_dt = sched_last_act_dttm.date()

    # remove all data from the schedule dataframe with null results
    sched_df_act = sched_df_saved.loc[sched_df_saved[cons.starttime_est_col] <= sched_last_act_dttm]
    sched_season_name = max(sched_df_act[cons.season_name_col]) # the latest season in the schedule dataframe
    del sched_df_saved

    print('... Finished loading saved season schedule data')

    ########################################
    ### 2. CHECK FOR UPDATES TO ACTUALS DATA
    ########################################
    print('\nChecking for updates to actual game results...')
    sched_df_missing = game_results_update(sched_last_act_dt)

    # real_last_act_dt is the last date of a completed game
    if not sched_df_missing.empty:
        real_last_act_dt = pd.to_datetime(max(sched_df_missing[cons.starttime_est_col]), format='ISO8601').date() 
    else:
        real_last_act_dt = sched_last_act_dt

    print('...Finished checking for updates to actual game results')

    ####################################################
    ### 3. UPDATE FUTURE GAMES IN THE SCHEDULE DATAFRAME
    ####################################################
    print('\nUpdating future schedule data...')
    sched_df_future = pd.DataFrame()

    # find the last day of the current nhl season
    real_season_name = cons.cur_season_name # the name corresponding to the latest season schedule released by the NHL
    real_last_sched_dt = pd.to_datetime(f'{real_season_name[4:]}-{cons.season_enddt}', format=cons.date_format_yyyy_mm_dd).date()

    # loop through each week of the season and fetch the schedule data for that week, then concatenate it to the season schedule dataframe
    for week in pd.date_range(start=real_last_act_dt + pd.Timedelta(days=1), end=real_last_sched_dt, freq='W'):

        # if the week is in the offseason, skip to the next week
        cur_year_offseason_begin_dt = pd.to_datetime(f'{week.year}-{cons.season_enddt}', format=cons.date_format_yyyy_mm_dd) + pd.Timedelta(days=1)
        cur_year_offseason_end_dt = pd.to_datetime(f'{week.year}-{cons.season_stdt}', format=cons.date_format_yyyy_mm_dd) - pd.Timedelta(days=1)
        if week >= cur_year_offseason_begin_dt and week <= cur_year_offseason_end_dt:
            continue

        print(f'\t... {week.strftime("%Y-%m-%d")} ...')
        for dow in range(0, 7):
            sched_df_future = pd.concat([sched_df_future, nhlc.get_sched_data(week, dow)], ignore_index=True)

    sched_df_future = clean_season_df(sched_df_future)

    sched_df = pd.concat([sched_df_act, sched_df_missing, sched_df_future], ignore_index=True)
    sched_df.sort_values(by=cons.starttime_utc_col, inplace=True)
    sched_df.reset_index(drop=True, inplace=True)

    print('\nFinished updating future schedule data')

    ##############################################
    ### 4. SAVE ALL UPDATED SCHEDULES TO CSV FILES
    ##############################################

    # create a list of the files to save
    seasons_to_update = list(sched_df_missing[cons.season_name_col].unique()) + list(sched_df_future[cons.season_name_col].unique())

    for seasonname in seasons_to_update:
        print(f'\nSaving updated schedule data for {seasonname[:4]}-{seasonname[4:]} season to CSV file...')
        csvSave(sched_df.loc[sched_df[cons.season_name_col] == seasonname], cons.season_sched_folder, cons.season_sched_filename.format(season=seasonname))

    pass



    sched_df[cons.starttime_utc_col] = pd.to_datetime(sched_df[cons.starttime_utc_col], format='ISO8601')
    sched_df[cons.starttime_est_col] = sched_df[cons.starttime_utc_col].dt.tz_convert('EST')

    sched_df_filt = pd.DataFrame()

    # loop through each date between today and the end of the season (cons.season_enddt) and check if there are any games on that date in the schedule dataframe that have not been updated with scores; if there are, update the schedule dataframe with the scores for those games by fetching the data from the API
    for game_date in pd.date_range(start=dt.now(ZoneInfo('EST')).date() - datetime.timedelta(days=1),
                                   end=pd.to_datetime(f'{dt.now(ZoneInfo('EST')).year}-{cons.season_enddt}').date(), freq='D'):
        print(f'\tUpdating schedule data for {game_date.strftime("%Y-%m-%d")}...')
        daily_sched = nhlc.get_sched_data(game_date, 0)
        if daily_sched.empty:
            continue
        sched_df_filt = pd.concat([sched_df_filt, daily_sched], ignore_index=True)
        sched_df_filt.sort_values(by=cons.starttime_utc_col, inplace=True)
        sched_df_filt.reset_index(drop=True, inplace=True)

    sched_df_filt[cons.starttime_utc_col] = pd.to_datetime(sched_df_filt[cons.starttime_utc_col], format='ISO8601')

    sched_df_cur = pd.concat([sched_df.loc[
        (sched_df[cons.season_name_col] == sched_season_name) &
        (sched_df[cons.starttime_est_col].dt.date < (dt.now(ZoneInfo('EST')).date() - datetime.timedelta(days=1)))], sched_df_filt], ignore_index=True)
    
    sched_df_cur[cons.starttime_est_col] = sched_df_cur[cons.starttime_utc_col].dt.tz_convert('EST')

    csvSave(sched_df_cur, cons.season_sched_folder, cons.season_sched_filename.format(season=sched_season_name))

    # update the odds analysis file with the new schedule data
    odds_data = csvLoad(cons.util_data_folder, 'all_time_schedule_odds.csv')

    odds_data[cons.starttime_utc_col] = pd.to_datetime(odds_data[cons.starttime_utc_col], format='ISO8601')
    odds_data[cons.starttime_est_col] = odds_data[cons.starttime_utc_col].dt.tz_convert('EST')
    sched_df_filt[cons.starttime_est_col] = sched_df_filt[cons.starttime_utc_col].dt.tz_convert('EST')

    update_dt = odds_data.loc[
        (odds_data[cons.season_name_col] == sched_season_name) &
        (odds_data[cons.last_period_col].notna())][cons.starttime_est_col].max()

    odds_data = pd.concat([odds_data.loc[odds_data[cons.starttime_est_col] <= update_dt], sched_df_filt.loc[
            sched_df_filt[cons.starttime_est_col] > update_dt
            ]], ignore_index=True)

    csvSave(odds_data, cons.util_data_folder, 'all_time_schedule_odds.csv')

    return sched_df_cur


def load_season_df(season_name):

    season_filename = cons.season_sched_filename.format(season=season_name)
    season_df = csvLoad(cons.season_sched_folder, season_filename)
    season_df = clean_season_df(season_df)

    return season_df


def clean_season_df(season_df):

    season_df[cons.starttime_utc_col] = pd.to_datetime(season_df[cons.starttime_utc_col], format='ISO8601')
    season_df[cons.starttime_est_col] = season_df[cons.starttime_utc_col].dt.tz_convert('EST')
    season_df[cons.season_name_col] = season_df[cons.season_name_col].astype(str)

    season_df.sort_values(by=cons.starttime_utc_col, inplace=True)
    season_df.reset_index(drop=True, inplace=True)

    for col in season_df.columns:
        if pd.api.types.is_integer_dtype(season_df[col]):
            season_df[col] = season_df[col].astype(int)

    return season_df


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
        season_results_points = nhlu.assign_game_points(season_results_df.loc[season_results_df['gameType']==2])
        final_standings_df = nhlu.generate_final_standings(season_results_points, today_dt)
        _, playoff_matchups, rounds_scheduled, rounds_completed = playoffs.playoff_tree_predictions(season_results_df, final_standings_df, False, today_dt, to_csv=False)

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

    # figure out which round of the playoffs is going on, if any
    if (rounds_scheduled == 0) & (rounds_completed == 0):
        playoff_rd = 0
    elif rounds_completed == 0:
        playoff_rd = 1
    elif rounds_completed == 1:
        playoff_rd = 2
    elif rounds_completed == 2:
        playoff_rd = 3
    elif rounds_completed == 3:
        playoff_rd = 4

    if to_csv:
        print(f'\nSaving playoff spot predictions to CSV file...')
        csvSave(count_df, cons.season_pred_folder.format(date=today_dt), cons.season_results_prob_filename.format(n=n, date=today_dt))
        display_playoff_probability(today_dt, season_results_df[cons.season_name_col].max(), playoff_rd=playoff_rd, matchups=playoff_matchups, display_image=False)

    return count_df


if __name__ == "__main__":
    import playoffs

    # schedule_update()

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
    feature_df = predict_season(to_csv=True, set_model_state=True, today_dt=today_dt)
    feature_df_game_points = nhlu.assign_game_points(feature_df.loc[(feature_df[cons.game_type_col]==2) & (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col]))])
    season_results_df = nhlu.generate_final_standings(feature_df_game_points, today_dt, to_csv=True)
    if not feature_df.loc[(feature_df[cons.game_type_col]==2) &
                      (feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col])) &
                      feature_df[cons.last_period_col].isna()].empty:
        nhlu.nhl_team_standings(season_results_df)
    playoff_results_df = playoffs.playoff_tree_predictions(feature_df, season_results_df, True, today_dt)

    ######################
    # create playoff spot predictions for current season based on n simulations
    # playoff_spot_predictions(today_dt, n=50)