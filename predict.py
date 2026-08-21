import os
import playoffs

import numpy as np
import pandas as pd
import constants as cons
import nhl_client as nhlc
import utils.nhl_utils as nhlu
import utils.skl_utils as sklu

from datetime import datetime as dt
from utils.file_utils import csvLoad, csvSave
from features.features import feature_data_load
from playoff_probability import display_playoff_probability


def predict_season(to_csv, set_model_state, today_dt):

    # load all of the feature data with all actuals
    feature_df = feature_data_load()

    # running for past predictions, pre-preprocess the feature data
    if today_dt < dt.now().date().strftime(cons.date_format_yyyy_mm_dd):

        feature_df, in_playoffs = preprocess_historic_data(feature_df, today_dt)

        # if playoffs are currently underway for the today_dt, move to playoff prediction method
        if in_playoffs:
            return feature_df

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
    feature_df.update(pred_df[[cons.home_team_win_col, cons.home_win_prob_col, cons.away_win_prob_col]])

    # display the predictions for the first date of predictions, and explain the predictions using SHAP values
    print_dt = min(feature_df.loc[feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date(),
                                  cons.starttime_est_col].dt.date)
    first_pred_dt_df = feature_df.loc[feature_df[cons.starttime_est_col].dt.date==print_dt]

    # print next game day's predictions into the terminal
    print(f'\nPredicted game results for {print_dt.strftime("%Y-%m-%d")}:')
    for idx, row in first_pred_dt_df.iterrows():

        home_team = cons.team_name_addrev_map[row[cons.home_team_name_col]]
        away_team = cons.team_name_addrev_map[row[cons.away_team_name_col]]
        ot_str = ' (OT)' if row[cons.last_period_col]=='OT' else ''

        if row[cons.home_win_prob_col] > row[cons.away_win_prob_col]:
            print(f"\t{away_team.lower()} {(row[cons.away_win_prob_col]*100):.2f} at {home_team} {(row[cons.home_win_prob_col]*100):.2f}{ot_str}")
        else:
            print(f"\t{away_team} {(row[cons.away_win_prob_col]*100):.2f} at {home_team.lower()} {(row[cons.home_win_prob_col]*100):.2f}{ot_str}")

        # generate SHAP values chart for each of the games
        sklu.explain_predictions(pred_df.iloc[[idx]][feature_list],
                                model, home_team, away_team,
                                print_dt, today_dt)

    # save the season predictinos to the prediction folder
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


def preprocess_historic_data(feature_df, today_dt):
    # get the season name corresponding to the inputted today_dt
    season_name = max(feature_df.loc[feature_df[cons.starttime_est_col].dt.date <= dt.strptime(today_dt, "%Y-%m-%d").date(),
                                        cons.season_name_col])

    # remove all seasons after this season from the feature dataframe
    feature_df = feature_df.loc[feature_df[cons.season_name_col] <= season_name]

    # indicates if the inputted today_dt is in the regular season or playoffs
    in_playoffs = False

    # if the today_dt is in the regular season, remove all scheduled playoff games for this season
    if not feature_df.loc[(feature_df[cons.game_type_col]==2) & (feature_df[cons.season_name_col]==season_name) &
                            (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date())].empty:
        in_playoffs = False
        feature_df = feature_df.loc[~((feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name))]

    # if the today_dt is in the playoffs, remove all scheduled playoff games for any rounds after the current one
    elif not feature_df.loc[(feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name) &
                            (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date())].empty:

        in_playoffs = True

        # get all the scheduled playoff matchups from this season
        cur_playoff_matchups = list(feature_df.loc[(feature_df[cons.game_type_col]==3) &
                                                    (feature_df[cons.season_name_col]==season_name)].sort_values(
                                                        by=cons.starttime_est_col)[[cons.home_team_name_col, cons.away_team_name_col]].drop_duplicates().values.tolist())

        # create round-specific matchups
        playoff_round_matchups = {}
        st_ind = 0
        for round in range(1, 5):
            if st_ind + 2**(5-round) > len(cur_playoff_matchups):
                break
            playoff_round_matchups[round] = cur_playoff_matchups[st_ind:st_ind + 2**(5-round)]
            st_ind += 2**(5-round)

        # find out which round of the playoffs is currently being played based on the inputted today_dt
        cur_playoff_round = 0
        fut_playoff_games_df = feature_df.loc[(feature_df[cons.game_type_col]==3) &
                                            (feature_df[cons.season_name_col]==season_name) &
                                            (feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date())]
        for round in playoff_round_matchups.keys():
            for matchup in playoff_round_matchups[round]:
                if not fut_playoff_games_df.loc[((fut_playoff_games_df[cons.home_team_name_col]==matchup[0]) &
                                             (fut_playoff_games_df[cons.away_team_name_col]==matchup[1])) |
                                             ((fut_playoff_games_df[cons.home_team_name_col]==matchup[1]) &
                                              (fut_playoff_games_df[cons.away_team_name_col]==matchup[0]))].empty:
                    cur_playoff_round = round
                    break
            if cur_playoff_round > 0:
                break

        # remove all scheduled games for future rounds from the feature dataframe
        for round in range(cur_playoff_round + 1, 5):
            if round not in playoff_round_matchups.keys():
                break
            feature_df = feature_df.loc[~((feature_df[cons.game_type_col]==3) & (feature_df[cons.season_name_col]==season_name) &
                                            ((feature_df[cons.home_team_name_col].isin([matchup[0] for matchup in playoff_round_matchups[round]])) &
                                            (feature_df[cons.away_team_name_col].isin([matchup[1] for matchup in playoff_round_matchups[round]]))))]

    # nullify the results of all games beyond the today_dt
    feature_df.loc[feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date(),
                    [cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col,
                    cons.home_team_win_col, cons.home_win_prob_col, cons.away_win_prob_col]] = np.nan

    # make sure 7 games are scheduled for each playoff series matchup
    cur_round_matchups = playoff_round_matchups[cur_playoff_round][:len(playoff_round_matchups[cur_playoff_round])//2]
    feature_df = playoffs.ensure_seven_games_scheduled(feature_df, cur_round_matchups, season_name, today_dt)

    # reset the playoff series count for all playoff matchups beyond the today_dt
    feature_df.loc[feature_df[cons.starttime_est_col].dt.date >= dt.strptime(today_dt, "%Y-%m-%d").date(), [cons.home_team_series_score_col, cons.away_team_series_score_col]] = 0

    return feature_df, in_playoffs


# def game_results_update(last_act_dt):

#     # today's date in EST timezone
#     today_dt = dt.now().date()

#     # initialize empty dataframe to store any missing schedule data that needs to be updated with actual results
#     missing_sched_df = pd.DataFrame()

#     # if the date of the first unplayed game is in the past, the schedule dataframe is missing some data
#     if today_dt > last_act_dt:
#         print('\tSchedule data actuals update required...')

#         # the last possible day of the season, no need to check games beyond this date
#         cur_season_enddt = pd.to_datetime(f'{cons.season_enddt}-{str(today_dt.year)}').date()

#         # add schedule actuals data one date at a time
#         for game_date in pd.date_range(start=last_act_dt, end=min(cur_season_enddt, today_dt - pd.Timedelta(days=1)), freq='D'):
#             print(f'\t\t... {game_date.strftime("%Y-%m-%d")} ...')
#             new_data = nhlc.get_sched_data(game_date, 0)

#             # if there were no games on this date, check the next date
#             if new_data.empty:
#                 continue

#             # if the games on this date have not yetr been played, there are no more games to check for actuals, so break out of the loop
#             if new_data.loc[new_data[cons.last_period_col].notna()].empty:
#                 break

#             # there were games on this date found with completed final scores, so add them to the missing schedule dataframe
#             missing_sched_df = pd.concat([missing_sched_df, new_data], ignore_index=True)

#         if missing_sched_df.empty:
#             print('\tNo missing actuals data found\n')
#             return missing_sched_df
        
#         missing_sched_df[cons.starttime_utc_col] = pd.to_datetime(missing_sched_df[cons.starttime_utc_col], format='ISO8601')
#         missing_sched_df[cons.starttime_est_col] = missing_sched_df[cons.starttime_utc_col].dt.tz_convert('EST')
#         missing_sched_df[cons.season_name_col] = missing_sched_df[cons.season_name_col].astype(str)

#         missing_sched_df.sort_values(by=cons.starttime_utc_col, inplace=True)
#         missing_sched_df.reset_index(drop=True, inplace=True)

#         for col in missing_sched_df.columns:
#             if isinstance(missing_sched_df[col], np.int64):
#                 missing_sched_df[col] = missing_sched_df[col].astype(int)

#     return missing_sched_df


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
        season_results_points = nhlu.assign_game_points(season_results_df.loc[season_results_df[cons.game_type_col]==2])
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