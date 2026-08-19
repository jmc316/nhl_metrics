import os
import numpy as np
import pandas as pd
import constants as cons
import nhl_client as nhlc
from zoneinfo import ZoneInfo
from datetime import datetime as dt
from utils.file_utils import csvLoad, csvSave


def sched_update():

    ################################################
    ### 1. LOAD THE SAVED SCHEDULE DATA WITH ACTUALS
    ################################################
    print('\nLoading saved season schedule data...')

    sched_df_saved = load_sched_df()

    # the day after the last played game in the schedule dataframe; this is the first day that needs to be updated with actual results
    sched_last_act_dttm = max(sched_df_saved.loc[sched_df_saved[cons.last_period_col].notna(), cons.starttime_utc_col]).astimezone(ZoneInfo(cons.est_tz)) + pd.Timedelta(days=1)
    sched_last_act_dt = sched_last_act_dttm.date()

    # remove all data from the schedule dataframe with null results
    sched_df_act = sched_df_saved.loc[sched_df_saved[cons.starttime_utc_col].dt.tz_convert(cons.est_tz) <= sched_last_act_dttm]
    del sched_df_saved

    print('... Finished loading saved season schedule data')

    ########################################
    ### 2. CHECK FOR UPDATES TO ACTUALS DATA
    ########################################
    print('\nChecking for updates to actual game results...')

    # today's date in EST timezone
    today_dt = dt.now().date()

    # initialize empty dataframe to store any missing schedule data that needs to be updated with actual results
    sched_df_missing = pd.DataFrame()

    # if the date of the first unplayed game is in the past, the schedule dataframe is missing some data
    if today_dt > sched_last_act_dt:
        print('\tSchedule data actuals update required...')

        # the last possible day of the season, no need to check games beyond this date
        cur_season_enddt = pd.to_datetime(f'{cons.season_enddt}-{str(today_dt.year)}').date()

        # add schedule actuals data one date at a time
        for game_date in pd.date_range(start=sched_last_act_dt, end=min(cur_season_enddt, today_dt - pd.Timedelta(days=1)), freq='D'):
            print(f'\t\t... {game_date.strftime("%Y-%m-%d")} ...')
            new_data = nhlc.get_sched_data(game_date, 0)

            # if there were no games on this date, check the next date
            if new_data.empty:
                continue

            # if the games on this date have not yet been played, there are no more games to check for actuals, so break out of the loop
            if new_data.loc[new_data[cons.last_period_col].notna()].empty:
                break

            # there were games on this date found with completed final scores, so add them to the missing schedule dataframe
            sched_df_missing = pd.concat([sched_df_missing, new_data], ignore_index=True)

        if sched_df_missing.empty:
            print('\tNo missing actuals data found\n')
        else:
            # clean the missing schedule dataframe
            sched_df_missing = clean_schedule_df(sched_df_missing)

    # real_last_act_dt is the last date of a completed game
    if not sched_df_missing.empty:
        real_last_act_dt = max(sched_df_missing[cons.starttime_utc_col]).astimezone(ZoneInfo(cons.est_tz)).date()
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

    sched_df_future = clean_schedule_df(sched_df_future)

    sched_df = pd.concat([sched_df_act, sched_df_missing, sched_df_future], ignore_index=True)
    sched_df.sort_values(by=cons.starttime_utc_col, inplace=True)
    sched_df.reset_index(drop=True, inplace=True)

    print('\nFinished updating future schedule data')

    ##############################################
    ### 4. SAVE ALL UPDATED SCHEDULES TO CSV FILES
    ##############################################

    # create a list of the files to save
    seasons_to_update = []
    if not sched_df_missing.empty:
        seasons_to_update.extend(list(sched_df_missing[cons.season_name_col].unique()))
    if not sched_df_future.empty:
        seasons_to_update.extend(list(sched_df_future[cons.season_name_col].unique()))

    for seasonname in seasons_to_update:
        print(f'\nSaving updated schedule data for {seasonname[:4]}-{seasonname[4:]} season to CSV file...')
        csvSave(sched_df.loc[sched_df[cons.season_name_col] == seasonname], cons.season_sched_folder, cons.season_sched_filename.format(season=seasonname))

    return sched_df


def load_sched_df_features():

    sched_df = load_sched_df()

    feature_df = sched_df.copy()

    # add startTimeEST column
    feature_df[cons.starttime_est_col] = feature_df[cons.starttime_utc_col].dt.tz_convert(cons.est_tz).dt.tz_localize(None)

    # add columns that the model will predict
    feature_df[cons.home_team_win_col] = np.where(feature_df[cons.home_team_score_col].isna(), np.nan,
        np.where(feature_df[cons.home_team_score_col] > feature_df[cons.away_team_score_col], 1.0, 0.0))
    feature_df[cons.win_prob_col.format(team='home')] = np.where(feature_df[cons.home_team_score_col].isna(), np.nan,
        np.where(feature_df[cons.home_team_score_col] > feature_df[cons.away_team_score_col], 1.0, 0.0))
    feature_df[cons.win_prob_col.format(team='away')] = np.where(feature_df[cons.home_team_score_col].isna(), np.nan,
        np.where(feature_df[cons.home_team_score_col] < feature_df[cons.away_team_score_col], 1.0, 0.0))

    # drop the startTimeUTC column
    feature_df.drop(columns=[cons.starttime_utc_col], inplace=True)

    # re-order the columns
    col_order = [cons.game_id_col, cons.season_name_col, cons.game_type_col, cons.starttime_est_col,
                 cons.venue_timezone_col, cons.venue_col, cons.home_team_name_col, cons.away_team_name_col,
                 cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col,
                 cons.home_team_win_col, cons.win_prob_col.format(team='home'), cons.win_prob_col.format(team='away')]
    feature_df = feature_df[col_order]

    return feature_df


def load_sched_df():

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

    sched_df_saved = clean_schedule_df(sched_df_saved)

    return sched_df_saved


def load_season_df(season_name):

    season_filename = cons.season_sched_filename.format(season=season_name)
    season_df = csvLoad(cons.season_sched_folder, season_filename)
    season_df = clean_schedule_df(season_df)

    return season_df


def clean_schedule_df(data_df):

    data_df[cons.starttime_utc_col] = pd.to_datetime(data_df[cons.starttime_utc_col], format='ISO8601')
    data_df[cons.season_name_col] = data_df[cons.season_name_col].astype(str)

    data_df.sort_values(by=cons.starttime_utc_col, inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    for col in data_df.columns:
        if pd.api.types.is_integer_dtype(data_df[col]):
            data_df[col] = data_df[col].astype(int)

    return data_df