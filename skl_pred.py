import os
import random
import numpy as np
import utils as ut
import pandas as pd
import constants as cons
from tabulate import tabulate
from termcolor import colored
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


def season_predictions(to_csv=False):

    # load or create the season schedule dataframe
    season_sched = create_feature_set()

    # apply preprocessing to the model features
    season_sched = feature_preprocessing(season_sched)

    print('\tTraining model and generating predictions...')

    # handle missing values by dropping rows with missing target values and training the model on the remaining data
    season_sched = season_sched.replace({None: np.nan}, inplace=True) 
    train_df = season_sched.dropna()

    # train a random forest regressor on the training data
    X_train = train_df[cons.feature_cols]
    y_train = train_df[cons.predict_cols]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Identify rows with missing 'target' values and predict
    to_predict = season_sched[season_sched[cons.last_period_col].isna()]
    X_to_predict = to_predict[cons.feature_cols]
    predictions = model.predict(X_to_predict)

    # Add predictions back into the original DataFrame as floats
    to_predict[cons.predict_cols] = predictions
    season_sched.update(to_predict)

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        season_sched.to_csv(cons.output_folder + cons.season_sched_pred_filename, index=False)

    return season_sched

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
                weekly_sched = weekly_sched_raw[[cons.game_id_col, cons.season_col, cons.game_type_col, cons.starttime_utc_col, cons.venue_timezone_col]]

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

                # concatenate this week's schedule to the season schedule dataframe
                season_sched = pd.concat([season_sched, weekly_sched], ignore_index=True)

                break

    # save the season schedule to a CSV file for future use
    if to_csv:
        if debug: print('\tSaving season schedule to CSV file...')
        season_sched.to_csv(cons.season_sched_folder + season_name + '_' + cons.season_sched_filename, index=False)

    return season_sched


def feature_preprocessing(season_sched):

    print('\tPreprocessing feature data...')

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    season_sched[cons.starttime_utc_col] = pd.to_datetime(season_sched[cons.starttime_utc_col])
    season_sched[cons.game_year_col] = season_sched[cons.starttime_utc_col].dt.year
    season_sched[cons.game_month_col] = season_sched[cons.starttime_utc_col].dt.month
    season_sched[cons.game_day_col] = season_sched[cons.starttime_utc_col].dt.day
    season_sched[cons.game_time_col] = season_sched[cons.starttime_utc_col].dt.hour * 60 + season_sched[cons.starttime_utc_col].dt.minute

    # encode the categorical features using ordinal encoding
    encoder = OrdinalEncoder()
    season_sched[cons.venue_timezone_col] = encoder.fit_transform(season_sched[[cons.venue_timezone_col]]).astype(int)
    season_sched[cons.venue_col] = encoder.fit_transform(season_sched[[cons.venue_col]]).astype(int)

    # change the season column to a string type
    season_sched[cons.season_col] = season_sched[cons.season_col].astype(str)

    # encode the target variable 'lastPeriod' using a mapping of the period types to integers
    season_sched[cons.last_period_col] = season_sched.loc[
        season_sched[cons.last_period_col].notna(), cons.last_period_col].map(cons.last_period_map).astype(int)

    return season_sched


def assign_game_points(season_results, to_csv=False):

    print('\tAssigning game points...')

    # for any rows where each team's score evaluates to the same int:
    # assign the last period as OT or SO and add 1 to the higher score to reflect the predicted winner
    season_results.loc[
        (season_results[cons.away_team_score_col].astype(int) == season_results[cons.home_team_score_col].astype(int)) & 
        (season_results[cons.away_team_score_col].notna()), cons.last_period_col] = season_results.apply(lambda row: random.choices(
            [cons.last_period_map['OT'], cons.last_period_map['SO']], weights=[100-cons.shootout_rate, cons.shootout_rate], k=1)[0], axis=1)
    season_results.loc[
        (season_results[cons.away_team_score_col].astype(int) == season_results[cons.home_team_score_col].astype(int)) & 
        (season_results[cons.away_team_score_col].notna()), cons.home_team_score_col] = season_results[cons.home_team_score_col].astype(int) + 1

    # change the type of the score columns back to int after 'ties' were absolved above
    season_results[cons.away_team_score_col] = season_results[cons.away_team_score_col].astype(int)
    season_results[cons.home_team_score_col] = season_results[cons.home_team_score_col].astype(int)
    season_results[cons.last_period_col] = season_results[cons.last_period_col].astype(int)

    # assign points to each team based on the predicted scores and last period type
    # (2 points for a win in regulation, 1 point for an OT/SO loss, 0 points for a regulation loss)
    season_results[cons.home_team_points_col] = np.where(
        season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col], 2, np.where(
            season_results[cons.last_period_col] > 0, 1, np.where(
                season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col], 0, 0)))
    season_results[cons.away_team_points_col] = np.where(
        season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col], 2, np.where(
            season_results[cons.last_period_col] > 0, 1, np.where(
                season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col], 0, 0)))

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        col_order = [cons.game_id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col,
                     cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col,
                     cons.last_period_col, cons.home_team_points_col, cons.away_team_points_col]
        season_results[col_order].to_csv(cons.output_folder + cons.season_sched_pred_filename, index=False)

    return season_results


def generate_final_standings(season_results, to_csv=False, load_csv=False):

    # if the final standings CSV file already exists in the output folder, load it instead of re-generating the standings from the season results dataframe
    if load_csv:
        season_results = pd.read_csv(cons.output_folder + cons.season_sched_pred_filename)

    print('\tGenerating final standings...')

    # filter the final schedule on the current season
    current_season = str(int(max(season_results[cons.season_col].str[4:]))-1) + max(season_results[cons.season_col].str[4:])
    season_results = season_results.loc[season_results[cons.season_col] == current_season]

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
    home_otls = season_results.loc[(season_results[cons.last_period_col]>0) &
                                   (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                       cons.home_team_name_col).size().reset_index(name=cons.home_team_otls_col)
    away_otls = season_results.loc[(season_results[cons.last_period_col]>0) &
                                   (season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col])].groupby(
                                       cons.away_team_name_col).size().reset_index(name=cons.away_team_otls_col)
    otls_df = home_away_accumulation(home_otls, away_otls, 'OTLs', keep_segregated_cols=True)

    # calculate total regulation wins for each team (used for tiebreakers in the standings)
    home_reg_wins = season_results.loc[(season_results[cons.last_period_col]==0) &
                                       (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                           cons.home_team_name_col).size().reset_index(name=cons.home_team_reg_wins_col)
    away_reg_wins = season_results.loc[(season_results[cons.last_period_col]==0) &
                                       (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                           cons.away_team_name_col).size().reset_index(name=cons.away_team_reg_wins_col)
    reg_wins_df = home_away_accumulation(home_reg_wins, away_reg_wins, 'RegWins')

    # calculate total regulation/OT wins for each team (used for tiebreakers in the standings)
    home_reg_ot_wins = season_results.loc[(season_results[cons.last_period_col]<2) &
                                          (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                              cons.home_team_name_col).size().reset_index(name=cons.home_team_reg_ot_wins_col)
    away_reg_ot_wins = season_results.loc[(season_results[cons.last_period_col]<2) &
                                          (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                              cons.away_team_name_col).size().reset_index(name=cons.away_team_reg_ot_wins_col)
    reg_ot_wins_df = home_away_accumulation(home_reg_ot_wins, away_reg_ot_wins, 'RegOTWins')

    # calculate total shootout wins for each team
    home_so_wins = season_results.loc[(season_results[cons.last_period_col]==2) &
                                      (season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col])].groupby(
                                          cons.home_team_name_col).size().reset_index(name=cons.home_team_so_wins_col)
    away_so_wins = season_results.loc[(season_results[cons.last_period_col]==2) &
                                      (season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col])].groupby(
                                          cons.away_team_name_col).size().reset_index(name=cons.away_team_so_wins_col)
    so_wins_df = home_away_accumulation(home_so_wins, away_so_wins, 'SOWins')

    # calculate total shootout losses for each team
    home_so_losses = season_results.loc[(season_results[cons.last_period_col]==2) &
                                        (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                            cons.home_team_name_col).size().reset_index(name=cons.home_team_so_losses_col)
    away_so_losses = season_results.loc[(season_results[cons.last_period_col]==2) &
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
    

def print_wildcard_results(final_standings):

    # Eastern conference divisional dataframes
    atlantic_standings = final_standings.loc[final_standings[cons.division_name_col] == cons.atl_div_val].sort_values(by=[cons.total_points_col], ascending=[False])
    metro_standings = final_standings.loc[final_standings[cons.division_name_col] == cons.metro_div_val].sort_values(by=[cons.total_points_col], ascending=[False])
    
    # print Eastern Conference data
    print('\n-------- Eastern Conference --------')
    print('------------------------------------')
    print('--- Atlantic Division ---')
    for _, row in atlantic_standings.head(3).iterrows():
        team_printer(row)
    print('--- Metropolitan Division ---')
    for _, row in metro_standings.head(3).iterrows():
        team_printer(row)
    print('--- Wild Card ---')
    east_wildcard_standings = pd.concat([atlantic_standings.drop(atlantic_standings.index[:3]), metro_standings.drop(metro_standings.index[:3])]).sort_values(by=[cons.total_points_col], ascending=[False])
    for _, row in east_wildcard_standings.head(2).iterrows():
        team_printer(row)
    east_wildcard_standings.drop(east_wildcard_standings.index[:2], inplace=True)
    print('-----------------')
    for _, row in east_wildcard_standings.iterrows():
        team_printer(row)

    # Western conference divisional dataframes
    central_standings = final_standings.loc[final_standings[cons.division_name_col] == cons.cen_div_val].sort_values(by=[cons.total_points_col], ascending=[False])
    pacific_standings = final_standings.loc[final_standings[cons.division_name_col] == cons.pac_div_val].sort_values(by=[cons.total_points_col], ascending=[False])
    
    # print Western Conference data
    print('\n-------- Western Conference --------')
    print('------------------------------------')
    print('--- Central Division ---')
    for _, row in central_standings.head(3).iterrows():
        team_printer(row)
    print('--- Pacific Division ---')
    for _, row in pacific_standings.head(3).iterrows():
        team_printer(row)
    print('--- Wild Card ---')
    west_wildcard_standings = pd.concat([central_standings.drop(central_standings.index[:3]), pacific_standings.drop(pacific_standings.index[:3])]).sort_values(by=[cons.total_points_col], ascending=[False])
    for _, row in west_wildcard_standings.head(2).iterrows():
        team_printer(row)
    west_wildcard_standings.drop(west_wildcard_standings.index[:2], inplace=True)
    print('-----------------')
    for _, row in west_wildcard_standings.iterrows():
        team_printer(row)


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


def team_printer(row):

    # print the team name, total points, and record (wins-losses-OTLs) for this team,
    # with the text colored in the team's primary color
    if len(row[cons.team_name_col]) < 16:
        print(colored(f'\t{row[cons.team_name_col]}\t\t{row[cons.total_points_col]}\t{row[cons.total_wins_col]}-{row[cons.total_losses_col]}-{row[cons.total_otls_col]}', cons.team_colors[row[cons.team_name_col]]))
    else:
        print(colored(f'\t{row[cons.team_name_col]}\t{row[cons.total_points_col]}\t{row[cons.total_wins_col]}-{row[cons.total_losses_col]}-{row[cons.total_otls_col]}', cons.team_colors[row[cons.team_name_col]]))


def playoff_spot_predictions(n=100):

    count_df = pd.DataFrame(columns=[cons.team_name_col, cons.div_1_val, cons.div_2_val, cons.div_3_val, cons.wc_1_val, cons.wc_2_val, cons.missed_val])
    count_df[cons.team_name_col] = list(cons.team_colors.keys())
    count_df.fillna(0, inplace=True)

    for i in range(n):
        print(f'\nSimulation {i+1} of {n}...')
        season_results = season_predictions()
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

    ######################
    # create season schedule dataframe for inputted season
    create_season_df('20212022', from_csv=False, to_csv=True)

    ######################
    # create a single season prediction wiuth saving the results as csv files
    # season_results = season_predictions(to_csv=True)
    # season_results = assign_game_points(season_results, to_csv=True)
    # final_standings = generate_final_standings(pd.DataFrame(), to_csv=True, load_csv=True)

    ######################
    # create playoff spot predictions for current season based on n simulations
    playoff_spot_predictions(n=100)