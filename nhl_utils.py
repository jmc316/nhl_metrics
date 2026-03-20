import numpy as np
import pandas as pd
import features as ft
import constants as cons
import terminal_ui as tui
import skl_utils as sklu

from tabulate import tabulate
from constants import nhl_client
from file_utils import csvLoad, csvSave
from datetime import datetime as dt


def nhl_team_stats():
    print('Fetching NHL team stats...')

    # display nhl team stats ui
    team_stats_ui = tui.team_stats_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.team_stats_options.items()}

    # call the function associated with the user's choice
    func_map[team_stats_ui.get_response()]()


def nhl_team_standings(data_df=None):

    if data_df is None:
        print('Fetching live NHL team standings...')
        # Fetch the standings
        data_df = pd.DataFrame(nhl_client.standings.league_standings()['standings'])

    data_df['wildcardSeed'] = None
    data_df.loc[data_df['playoffSeed'].str[:3] == 'div', 'wildcardSeed'] = 0
    data_df.loc[data_df['playoffSeed'].str[:2] == 'wc', 'wildcardSeed'] = data_df.loc[data_df['playoffSeed'].str[:2] == 'wc', 'playoffSeed'].str[-1:].astype(int)
    data_df.loc[data_df['playoffSeed'] == 'Missed', 'wildcardSeed'] = 3
        
    # eastern conference playoff seeding
    east_conf_spots = _sort_conference_wildcard_spots(data_df.loc[data_df[cons.conference_name_col] == 'Eastern'])
    print_wildcard_standings(east_conf_spots, 'Eastern')

    # western conference playoff seeding
    west_conf_spots = _sort_conference_wildcard_spots(data_df.loc[data_df[cons.conference_name_col] == 'Western'])
    print_wildcard_standings(west_conf_spots, 'Western')

    print()


def _sort_conference_wildcard_spots(conf_df):

    sort_df = conf_df.copy()
    playoff_seed_str = sort_df[cons.playoff_seed_col].fillna(cons.missed_val).astype(str).str.lower()

    # Group ordering: division seeds first, wildcard seeds second, then missed teams.
    sort_df['_seed_group'] = np.where(
        playoff_seed_str.str.startswith('div'),
        0,
        np.where(playoff_seed_str.str.startswith('wc'), 1, 2)
    )

    # Division name should only influence ordering inside division seed rows.
    sort_df['_division_sort'] = np.where(
        sort_df['_seed_group'] == 0,
        sort_df[cons.division_name_col],
        ''
    )
    sort_df['_points_sort'] = pd.to_numeric(sort_df[cons.total_points_col], errors='coerce').fillna(-np.inf)

    sort_df.sort_values(
        by=['_seed_group', '_division_sort', '_points_sort'],
        ascending=[True, True, False],
        inplace=True
    )
    sort_df.drop(columns=['_seed_group', '_division_sort', '_points_sort'], inplace=True)

    return sort_df


def print_wildcard_standings(data_df, conference_name):

    print(f'\n--- {conference_name} Conference Wild Card ---')
    for _, team in data_df.iterrows():
        space = ''.join([ ' ' for _ in range(1, 23 - len(team[cons.team_name_col]))])
        if team[cons.playoff_seed_col] == 'div_3':
            print(team[cons.playoff_seed_col][-1:], team[cons.team_name_col], space, team['totalPoints'])
            print('----------------------------')
        elif team[cons.playoff_seed_col] == 'wc_2':
            print(team[cons.playoff_seed_col][-1:], team[cons.team_name_col], space, team['totalPoints'])
            print('----------------------------')
        elif team[cons.playoff_seed_col] != 'Missed':
            print(team[cons.playoff_seed_col][-1:], team[cons.team_name_col], space, team['totalPoints'])
        else:
            print('-', team[cons.team_name_col], space, team['totalPoints'])


def nhl_individual_team_stats():
    print('Fetching individual NHL team stats...')

    # Fetch the teams
    teams = nhl_client.teams.teams()

    print(teams)


def team_info():
    
    while True:
        try:
            # Fetch the teams
            teams_df = pd.DataFrame(cons.nhl_client.teams.teams())
            break
        except Exception as ex:
            print(f'\t\t... {ex} ...')
            continue
        
    teams_df.rename(columns={'name': cons.team_name_col}, inplace=True)

    teams_df[cons.conference_name_col] = teams_df['conference'].apply(lambda x: x['name'])
    teams_df[cons.division_name_col] = teams_df['division'].apply(lambda x: x['name'])

    teams_df = teams_df[[cons.team_name_col, cons.conference_name_col, cons.division_name_col]]

    return teams_df

    
def assign_game_points(season_results, to_csv=False):

    print('Assigning game points...')

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
    # if to_csv:
    #     col_order = [cons.game_id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col,
    #                  cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col,
    #                  cons.last_period_col, cons.home_team_points_col, cons.away_team_points_col]
    #     csvSave(season_results[col_order], cons.output_folder, cons.season_sched_pred_filename)

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
    # if load_csv:
    #     season_results = csvLoad(cons.output_folder, cons.season_sched_pred_filename)

    print('Generating final standings...')

    # filter the final schedule on the current season
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
    home_losses = season_results.loc[(season_results[cons.last_period_col]=='REG') &
                                     (season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col])].groupby(
                                         cons.home_team_name_col).size().reset_index(name=cons.home_team_losses_col)
    away_losses = season_results.loc[(season_results[cons.last_period_col]=='REG') &
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

    # merge the points, wins, losses, and OT losses dataframes together to create the final standings dataframe
    final_standings = pd.merge(points_df, wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, losses_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, otls_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, reg_wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, reg_ot_wins_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, goals_for_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, goals_against_df, on=cons.team_name_col)
    final_standings = pd.merge(final_standings, games_played_df, on=cons.team_name_col)

    # free memory by deleting the intermediate dataframes that are no longer needed
    del points_df, wins_df, losses_df, otls_df, reg_wins_df, reg_ot_wins_df, goals_for_df, goals_against_df, games_played_df

    # calculate functional columns
    final_standings[cons.goal_diff_col] = final_standings[cons.total_goals_for_col] - final_standings[cons.total_goals_against_col]
    final_standings[cons.points_percentage_col] = final_standings[cons.total_points_col] / (final_standings[cons.total_games_col] * 2)

    # load individual team info to merge with final standings for wildcard setup
    team_info_df = team_info()
    final_standings = pd.merge(final_standings, team_info_df[[cons.team_name_col, cons.division_name_col, cons.conference_name_col]], on=cons.team_name_col)

    # assign divisionSeed, conferenceSeed based on total points and tiebreakers within each division, conference
    final_standings.sort_values(by=[cons.division_name_col] + cons.tiebreaker_cols, ascending=[True] + [False]*len(cons.tiebreaker_cols), inplace=True)
    final_standings[cons.division_seed_col] = [val%8+1 for val in list(final_standings.reset_index(drop=True).reset_index().index)]
    final_standings.sort_values(by=[cons.conference_name_col] + cons.tiebreaker_cols, ascending=[True] + [False]*len(cons.tiebreaker_cols), inplace=True)
    final_standings[cons.conference_seed_col] = [val%16+1 for val in list(final_standings.reset_index(drop=True).reset_index().index)]
    final_standings.sort_values(by=cons.tiebreaker_cols, ascending=[False]*len(cons.tiebreaker_cols), inplace=True)

    # calculate playoff seed by including the top three teams from every division and then the top two remaining teams from each conference
    # division playoff spots are labelled as 'div_x', where x=[1, 2, 3] represents the division seed
    final_standings[cons.playoff_seed_col] = np.where(final_standings[cons.division_seed_col] <= 3, 'div_' + final_standings[cons.division_seed_col].astype(str), cons.missed_val)
    
    # wildcard playoff spots are labelled as 'wc_x', where x=[1, 2] represents the wildcard seed; non-playoff teams are labelled as 'Missed'
    missed_mask = final_standings[cons.playoff_seed_col] == cons.missed_val
    missed_df = final_standings.loc[missed_mask].copy()
    missed_df.sort_values(
        by=[cons.conference_name_col] + cons.tiebreaker_cols + [cons.team_name_col],
        ascending=[True] + [False] * len(cons.tiebreaker_cols) + [True],
        inplace=True
    )
    missed_df['wildcard_rank'] = missed_df.groupby(cons.conference_name_col).cumcount() + 1
    missed_df[cons.playoff_seed_col] = np.where(
        missed_df['wildcard_rank'] <= 2,
        'wc_' + missed_df['wildcard_rank'].astype(str),
        cons.missed_val
    )
    final_standings.loc[missed_df.index, cons.playoff_seed_col] = missed_df[cons.playoff_seed_col]

    # reorder the columns and sort the final standings by the tiebreaker columns in descending order
    final_standings = final_standings[cons.final_standings_col_order]
    final_standings.sort_values(by=cons.tiebreaker_cols, ascending=[False]*len(cons.tiebreaker_cols), inplace=True)

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
        csvSave(final_standings, cons.season_pred_folder.format(date=today_dt), cons.final_standings_filename.format(date=today_dt))

    return final_standings


def playoff_probabilities_printer(count_df):

    percent_cols = [cons.team_name_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%', f'{cons.missed_val}_%', f'{cons.playoff_per_col}']

    count_df = count_df.sort_values(by=[cons.playoff_per_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%'], ascending=False).reset_index(drop=True)[percent_cols]
    print('\nPlayoff Spot Probabilities:')

    print(tabulate(count_df, headers='keys', tablefmt='grid'))


def playoff_tree_predictions(regular_season_df, season_results_df, set_model_random_state):

    print('Predicting playoff tree...')
    
    # create a dataframe with the already scheduled playoff games
    scheduled_games_df = regular_season_df.loc[(regular_season_df[cons.game_type_col] == 3) &
                                        (regular_season_df[cons.season_name_col] == max(regular_season_df[cons.season_name_col]))].copy()

    # if there are no scheduled playoff games, will need to create schedules for all rounds before predictions
    if scheduled_games_df.empty:
        print('\tNo scheduled playoff games found for this season...')
        rounds_scheduled = 0
        playoff_df = pd.DataFrame()
    else:
        # can't test this yet
        # need to assume that a whole round has finished and is in the same format as the generated playoff schedules
        exit('Error: Found scheduled playoff games in the regular season schedule, but this functionality has not been implemented yet.\n' \
        'Please ensure that the regular season schedule does not contain any playoff games or that the playoff games are in the same format as the generated playoff schedules.')
    
    # load the venue map to establish each team's home venue and timezone
    venue_map_df = venue_map_load(regular_season_df)

    # initialize lists to store OOB predictions, MSE, and R-squared values for each playoff round
    oob_list, mse_list, rsq_list = [], [], []

    # loop through every playoff round
    for pl_round in range(rounds_scheduled+1, 5):
        print(f'Playoff Round {pl_round}')

        # if there is no schedule for this round, create the schedule
        if rounds_scheduled+1 <= pl_round:

            # playoff round 1
            if pl_round == 1:
                # generate the round 1 playoff matchups based off the regular season standings
                east_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Eastern'], 1)
                west_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Western'], 1)
                all_matchups = east_playoff_matchups + west_playoff_matchups

                # create the round 1 playoff schedule and add it to the regular season schedule
                playoff_df = pd.concat([regular_season_df, create_playoff_round_schedule(all_matchups, venue_map_df, regular_season_df, playoff_df)], ignore_index=True, sort=False)
            # playoff rounds 2, 3, 4
            else:
                # generate the round n playoff matchups based off the regular season standings
                all_matchups = generate_playoff_matchups(playoff_df, pl_round, all_matchups)

                # create the round n playoff schedule and add it to the regular season schedule
                playoff_df = create_playoff_round_schedule(all_matchups, venue_map_df, regular_season_df, playoff_df)

            # initialize the series score columns for the current playoff round to 0 for all games in the playoff schedule dataframe
            series_score_cols = [cons.home_team_series_score_col, cons.away_team_series_score_col]
            playoff_df[series_score_cols] = 0

            # initialize the previous game date as None
            game_dt_prev = None

            # predict games for this playoff round one day at a time
            for game_dt in playoff_df.loc[(playoff_df[cons.season_name_col] == max(playoff_df[cons.season_name_col])) &
                                          (playoff_df[cons.game_type_col] == 3) &
                                          (playoff_df[cons.last_period_col].isna()), cons.game_date_col].unique():
                
                print(f'\tPredicting games for {game_dt.strftime("%Y-%m-%d")}...')

                playoff_df_filt = playoff_df.loc[playoff_df[cons.game_date_col] <= game_dt]

                # add dependent features to the playoff schedule dataframe
                playoff_df_filt = ft.dependent_feature_add(playoff_df_filt, backfill=False, debug=False)

                # check to see if any of the series are over based on the current series scores
                playoff_df = series_final_check(playoff_df, playoff_df_filt, game_dt, game_dt_prev)

                # increment the previous game date to the current game date for the next iteration
                game_dt_prev = game_dt

                # predict games on selected date
                playoff_df_filt = sklu.make_predictions(playoff_df_filt, oob_list, mse_list, rsq_list, set_model_random_state, load_model=True, save_model=False)

                playoff_df = pd.concat([playoff_df_filt, playoff_df.loc[playoff_df[cons.game_date_col] > game_dt]], ignore_index=True)

            # declare the winner of series that finished on the last day of the first round
            playoff_df = ft.playoff_series_score_fill(playoff_df.loc[playoff_df[cons.game_date_col] == max(playoff_df[cons.game_date_col])],
                                                      playoff_df, max(playoff_df[cons.season_name_col]), debug=True)

    return playoff_df


def generate_playoff_matchups(data_df, round_num, prev_round_matchups=None):

    # create matchups for the first round of playoffs
    if round_num == 1:
        # matchup 1: division winner with better record vs wildcard 2 team
        matchup_1 = [data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.team_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.team_name_col].values[0],
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_2', cons.team_name_col].values[0],
            data_df[cons.conference_name_col].values[0],
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.division_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.division_name_col].values[0],
            'Division Winner 1 vs Wildcard 2'
            ]
        
        # matchup 2: division winner with worse record vs wildcard 1 team
        matchup_2 = [data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.team_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=True)[cons.team_name_col].values[0],
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_1', cons.team_name_col].values[0],
            data_df[cons.conference_name_col].values[0],
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.division_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.division_name_col].values[1],
            'Division Winner 2 vs Wildcard 1'
            ]
        
        # matchup 3: inter-division matchup between division 2 & 3 seeds
        div_1 = data_df[cons.division_name_col].unique()[0]
        matchup_3 = [data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') & (data_df[cons.division_name_col] == div_1), cons.team_name_col].values[0],
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') & (data_df[cons.division_name_col] == div_1), cons.team_name_col].values[0],
            data_df[cons.conference_name_col].values[0],
            div_1,
            f' Division 2 vs Division 3'
            ]
        
        # matchup 4: inter-division matchup between division 2 & 3 seeds
        div_2 = data_df[cons.division_name_col].unique()[1]
        matchup_4 = [data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') & (data_df[cons.division_name_col] == div_2), cons.team_name_col].values[0],
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') & (data_df[cons.division_name_col] == div_2), cons.team_name_col].values[0],
            data_df[cons.conference_name_col].values[0],
            div_2,
            f' Division 2 vs Division 3'
            ]
        
        all_matchups = [matchup_1, matchup_2, matchup_3, matchup_4]

    # create matchups for the second round of playoffs
    elif round_num == 2:

        all_matchups = playoff_matchups_234(
            data_df,
            0, 8, # 8 series winners from round 1
            2, # divisional id is in index 2 of the matchup list
            prev_round_matchups,
            {
                'Atlantic': ['Eastern'],
                'Metropolitan': ['Eastern'],
                'Central': ['Western'],
                'Pacific': ['Western']
            },
            'Division Final' # title of round 2
        )

    # create matchups for the third round of playoffs
    elif round_num == 3:

        all_matchups = playoff_matchups_234(
            data_df,
            8, 12, # 4 series winners from round 2
            1, # conference id is in index 2 of the matchup list
            prev_round_matchups,
            {
                'Eastern': ['NHL'],
                'Western': ['NHL']
            }, 
            'Conference Final' # title of round 3
        )

    # create matchup for the fourth round of playoffs (Stanley Cup Final)
    elif round_num == 4:

        all_matchups = playoff_matchups_234(
            data_df,
            12, 14, # 2 series winners from round 3
            1, # nhl id is in index 2 of the matchup list (irrelevant)
            prev_round_matchups,
            {
                'NHL': ['NHL']
            },
            'Stanley Cup Final' # title of round 4
        )

    return all_matchups


def playoff_matchups_234(data_df, ri_low, ri_high, round_id, prev_round_matchups, matchups_dict, round_name):

    # find all series deciding games from the first round
    series_winners_df = data_df.loc[(data_df[cons.season_name_col]==max(data_df[cons.season_name_col])) &
                    ((data_df[cons.home_team_series_score_col] == 4) | (data_df[cons.away_team_series_score_col] == 4))]
    series_winners_list = []

    # find all series winners from the first round and add them to a list
    for i in range(ri_low, ri_high):
        last_game = series_winners_df.iloc[i]
        series_winner = last_game[cons.home_team_name_col] if last_game[cons.home_team_series_score_col] == 4 else last_game[cons.away_team_name_col]
        series_winners_list.append(series_winner)

    # remove the teams that lost each series 
    for matchup in prev_round_matchups:
        if matchup[0] in series_winners_list:
            matchup.remove(matchup[1])
        elif matchup[1] in series_winners_list:
            matchup.remove(matchup[0])
        pass

    # add the winners from the first round to the second round matchups dictionary and reconfigure the result list
    for matchup in prev_round_matchups:
        matchups_dict[matchup[round_id]].append(matchup[0])
    all_matchups = [[values[1], values[2], values[0], key, round_name] for key, values in matchups_dict.items()]

    return all_matchups


def create_playoff_round_schedule(all_matchups, venue_map_df, feature_df, playoff_df):

    # if the playoff dataframe is empty, take the round start date from the regular season
    if playoff_df.empty:
        round_stdt = feature_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)
    else:
        round_stdt = playoff_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)

    # loop through matchups
    for matchup in all_matchups:

        # if the matchup is an western matchup, start series on matchday 2
        if matchup[2] == 'Western':
            game_dt = round_stdt + pd.Timedelta(days=1)
            sched_format = cons.playoff_sched_format
        # if the matchup is an eastern matchup, start series on matchday 1
        elif matchup[2] == 'Eastern':
            game_dt = round_stdt
            sched_format = cons.playoff_sched_format
        # if the matchup is the final, start series on matchday 1
        else:
            game_dt = round_stdt
            sched_format = cons.final_sched_format
        
        # list of game dates for the series
        game_dts = [game_dt + pd.Timedelta(days=val) for val in sched_format]

        # list of home and away teams for the series (higher seed is home first)
        home_teams = [matchup[0]] * 2 + [matchup[1]] * 2 + [matchup[0]] + [matchup[1]] + [matchup[0]]
        away_teams = [matchup[1]] * 2 + [matchup[0]] * 2 + [matchup[1]] + [matchup[0]] + [matchup[1]]

        # list of venues for the sesries based off the home team for each game
        venues = [list(venue_map_df.loc[venue_map_df[cons.home_team_name_col]==home_team][[cons.venue_col, cons.venue_timezone_col]].values[0]) for home_team in home_teams]

        # game ID does not exist for unscheduled playoff games
        game_id = np.nan

        # game type for playoff games is 3
        game_type = 3

        # season name is the current season for the feature dataframe
        season_name = max(feature_df[cons.season_name_col])

        # get the most popular game time per venue for the current season to use as the game time for the playoff matchups
        game_time_df = pd.DataFrame(feature_df.loc[
            feature_df[cons.season_name_col]==max(feature_df[cons.season_name_col])][[
                cons.game_time_col, cons.venue_col]].value_counts(), columns=['count'])
        game_time_df = game_time_df.loc[game_time_df['count'] > 5]
        game_time_df = game_time_df.loc[game_time_df.groupby(cons.venue_col)['count'].idxmax()]
        game_time_df.reset_index(inplace=True)
        game_time_utc = [int(game_time_df.loc[game_time_df[cons.venue_col]==venue[0]][cons.game_time_col].values[0]) for venue in venues]

        # add all data to a dataframe for the current matchup and append to the playoff dataframe
        matchup_df = pd.DataFrame({
            cons.game_id_col: [game_id] * len(game_dts),
            cons.season_name_col: [season_name] * len(game_dts),
            cons.game_type_col: [game_type] * len(game_dts),
            cons.venue_timezone_col: [venue[1] for venue in venues],
            cons.venue_col: [venue[0] for venue in venues],
            cons.home_team_name_col: home_teams,
            cons.away_team_name_col: away_teams,
            cons.away_team_score_col: [np.nan] * len(game_dts),
            cons.home_team_score_col: [np.nan] * len(game_dts),
            cons.last_period_col: [np.nan] * len(game_dts),
            cons.game_date_col: game_dts,
            cons.game_time_col: game_time_utc
        })

        # add the matchup games to the playoff dataframe
        playoff_df = pd.concat([playoff_df, matchup_df], ignore_index=True).sort_values(by=[cons.game_date_col, cons.game_time_col])

    return playoff_df


def venue_map_load(regular_season_df):

    venue_map_df = pd.DataFrame(regular_season_df.loc[
        regular_season_df[cons.season_name_col]==max(regular_season_df[cons.season_name_col])][[
            cons.home_team_name_col, cons.venue_col, cons.venue_timezone_col]].value_counts(), columns=['count'])
    venue_map_df = venue_map_df.loc[venue_map_df['count'] > 20]
    venue_map_df.drop(columns=['count'], inplace=True)
    venue_map_df.reset_index(inplace=True)

    return venue_map_df


def series_final_check(playoff_df, playoff_df_filt, game_dt, game_dt_prev):

    # gather all games from the previous playoff game date
    series_score_checker_df = playoff_df_filt.loc[(playoff_df_filt[cons.season_name_col] == max(playoff_df_filt[cons.season_name_col])) &
                                                (playoff_df_filt[cons.game_type_col] == 3) &
                                                (playoff_df_filt[cons.game_date_col] == game_dt_prev)
                                                ]
    
    # loop through all games from previous game date
    for _, row in series_score_checker_df.iterrows():

        # if there is a winner and it was not game 7, drop any remaining scheduled games in this series
        if ((row[cons.home_team_series_score_col] == 4) and (row[cons.home_team_series_score_col] != 3)) or \
            ((row[cons.away_team_series_score_col] == 4) and (row[cons.away_team_series_score_col] != 3)):

            # drop all future games between these two teams in the playoff schedule dataframe
            indeces_drop = playoff_df.loc[(playoff_df[cons.game_date_col] > game_dt) &
                                                    (((playoff_df[cons.home_team_name_col] == row[cons.home_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.away_team_name_col])) |
                                                    ((playoff_df[cons.home_team_name_col] == row[cons.away_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.home_team_name_col])))].index
            playoff_df.drop(index=indeces_drop, inplace=True)

        # print out the series results
        if (row[cons.home_team_series_score_col] == 4) or (row[cons.away_team_series_score_col] == 4):
            if row[cons.home_team_series_score_col] == 4:
                print(f'\t\t{row[cons.home_team_name_col]} advance past {row[cons.away_team_name_col]}: {row[cons.home_team_series_score_col]}-{row[cons.away_team_series_score_col]}')
            else:
                print(f'\t\t{row[cons.away_team_name_col]} advance past {row[cons.home_team_name_col]}: {row[cons.away_team_series_score_col]}-{row[cons.home_team_series_score_col]}')
    
    return playoff_df


if __name__ == '__main__':
    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
    feature_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))
    final_standings_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.final_standings_filename.format(date=today_dt))

    playoff_tree_predictions(feature_df, final_standings_df, False)