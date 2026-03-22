import numpy as np
import pandas as pd
import features as ft
import constants as cons
import terminal_ui as tui
import skl_utils as sklu

from tabulate import tabulate
from nhl_client import nhl_client
from file_utils import csvLoad, csvSave
from datetime import datetime as dt
from playoff_matchup import PlayoffMatchup


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
            teams_df = pd.DataFrame(nhl_client.teams.teams())
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


def generate_final_standings(season_results, to_csv=False):

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
        print('Saving final standings to CSV...')
        today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
        csvSave(final_standings, cons.season_pred_folder.format(date=today_dt), cons.final_standings_filename.format(date=today_dt))

    return final_standings


def playoff_probabilities_printer(count_df):

    percent_cols = [cons.team_name_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%', f'{cons.missed_val}_%', f'{cons.playoff_per_col}']

    count_df = count_df.sort_values(by=[cons.playoff_per_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%'], ascending=False).reset_index(drop=True)[percent_cols]
    print('\nPlayoff Spot Probabilities:')

    print(tabulate(count_df, headers='keys', tablefmt='grid'))


def playoff_tree_predictions(regular_season_df, season_results_df, set_model_random_state, to_csv=True):

    print('Predicting playoff tree...')

    # if the points columns are in the season results dataframe, remove them to avoid confusing the model
    if cons.home_team_points_col in regular_season_df.columns:
        regular_season_df.drop(columns=[cons.home_team_points_col], inplace=True)
    if cons.away_team_points_col in regular_season_df.columns:
        regular_season_df.drop(columns=[cons.away_team_points_col], inplace=True)
    
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
        print(f'\nPlayoffs Round {pl_round}')

        # if there is no schedule for this round, create the schedule
        if rounds_scheduled+1 <= pl_round:

            # playoff round 1
            if pl_round == 1:
                # generate the round 1 playoff matchups based off the regular season standings
                east_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Eastern'], 1)
                west_playoff_matchups = generate_playoff_matchups(season_results_df.loc[season_results_df[cons.conference_name_col] == 'Western'], 1)

                all_matchups = east_playoff_matchups.copy()
                for matchup_num, matchup in west_playoff_matchups.items():
                    all_matchups.update({matchup_num+4: matchup})

                # create the round 1 playoff schedule and add it to the regular season schedule
                playoff_df = pd.concat([regular_season_df, create_playoff_round_schedule(all_matchups, venue_map_df, regular_season_df, playoff_df)], ignore_index=True, sort=False)
            # playoff rounds 2, 3, 4
            else:
                # generate the round n playoff matchups based off the regular season standings
                all_matchups = generate_playoff_matchups(playoff_df, pl_round, all_matchups)

                # create the round n playoff schedule and add it to the regular season schedule
                playoff_df = create_playoff_round_schedule(all_matchups, venue_map_df, regular_season_df, playoff_df)

            # predict games for this playoff round one day at a time
            for game_dt in playoff_df.loc[(playoff_df[cons.season_name_col] == max(playoff_df[cons.season_name_col])) &
                                          (playoff_df[cons.game_type_col] == 3) &
                                          (playoff_df[cons.last_period_col].isna()), cons.game_date_col].unique():
                
                # if there were scheduled games on this date that no longer exist, skip to the next date
                if playoff_df.loc[playoff_df[cons.game_date_col] == game_dt].empty:
                    continue

                playoff_df_filt = playoff_df.loc[playoff_df[cons.game_date_col] <= game_dt]

                # add dependent features to the playoff schedule dataframe
                playoff_df_filt = ft.dependent_feature_add(playoff_df_filt, backfill=False, debug=False)

                # predict games on selected date
                print(f'\tPredicting games for {game_dt.strftime("%Y-%m-%d")}...')
                playoff_df_filt = sklu.make_predictions(playoff_df_filt, oob_list, mse_list, rsq_list, set_model_random_state, load_model=True, save_model=False)

                playoff_df = pd.concat([playoff_df_filt, playoff_df.loc[playoff_df[cons.game_date_col] > game_dt]], ignore_index=True)

                # check to see if any of the series are over based on the current series scores
                playoff_df, all_matchups = series_final_check(playoff_df, playoff_df_filt, all_matchups, game_dt)

    # save playoff predictions to CSV
    if to_csv:
        print('\nSaving playoff predictions to CSV...')
        today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
        csvSave(playoff_df, cons.season_pred_folder.format(date=today_dt), cons.playoff_pred_filename.format(date=today_dt))

    return playoff_df


def generate_playoff_matchups(data_df, round_num, prev_round_matchups=None):

    matchups_dict = {}

    # create matchups for the first round of playoffs
    if round_num == 1:
        # matchup 1: division winner with better record vs wildcard 2 team
        matchups_dict.update({0: PlayoffMatchup(
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.team_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.team_name_col].values[0], # highest seed in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_2', cons.team_name_col].values[0], # second wildcard team in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.conference_seed_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.conference_seed_col].values[0], # division winner conference seed
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_2', cons.conference_seed_col].values[0], # second wildcard team conference seed
            1, # playoff round number
            data_df[cons.conference_name_col].values[0], # conference name
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.division_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=False)[cons.division_name_col].values[0] # division name (based off division winner)
        )})
        
        # matchup 2: division winner with worse record vs wildcard 1 team
        matchups_dict.update({1: PlayoffMatchup(
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.team_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=True)[cons.team_name_col].values[0], # division winner with worse record in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_1', cons.team_name_col].values[0], # first wildcard team in the conference
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.conference_seed_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=True)[cons.conference_seed_col].values[0], # division winner conference seed
            data_df.loc[data_df[cons.playoff_seed_col] == 'wc_1', cons.conference_seed_col].values[0], # first wildcard team conference seed
            1, # playoff round number
            data_df[cons.conference_name_col].values[0], # conference name
            data_df.loc[data_df[cons.playoff_seed_col] == 'div_1', [cons.division_name_col, cons.total_points_col]].sort_values(
                by=cons.total_points_col, ascending=True)[cons.division_name_col].values[0] # division name (based off division winner)
        )})

        # matchup 3: inter-division matchup between division 2 & 3 seeds
        matchups_dict.update({2: PlayoffMatchup(
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[0]), cons.team_name_col].values[0], # division 2 seed from division 1
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[0]), cons.team_name_col].values[0], # division 3 seed from division 1
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[0]), cons.conference_seed_col].values[0], # division 2 seed conference seed
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[0]), cons.conference_seed_col].values[0], # division 3 seed conference seed
            1, # playoff round number
            data_df[cons.conference_name_col].values[0], # conference name
            data_df[cons.division_name_col].unique()[0] # division name
        )})
        
        # matchup 4: inter-division matchup between division 2 & 3 seeds
        matchups_dict.update({3: PlayoffMatchup(
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[1]), cons.team_name_col].values[0], # division 2 seed from division 2
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[1]), cons.team_name_col].values[0], # division 3 seed from division 2
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_2') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[1]), cons.conference_seed_col].values[0], # division 2 seed conference seed
            data_df.loc[(data_df[cons.playoff_seed_col] == 'div_3') &
                        (data_df[cons.division_name_col] == data_df[cons.division_name_col].unique()[1]), cons.conference_seed_col].values[0], # division 3 seed conference seed
            1, # playoff round number
            data_df[cons.conference_name_col].values[0], # conference name
            data_df[cons.division_name_col].unique()[1] # division name
        )})

    # create matchups for the second, third, and fourth rounds of the playoffs based off the winners from the previous round
    else:
        matchups_dict = playoff_matchups_234(round_num, prev_round_matchups)

    return matchups_dict


def playoff_matchups_234(round_id, prev_round_matchups):

    # find all series winners from the previous round
    series_winners_list = [[
        matchup.get_series_winner(),
        matchup.get_winner_conf_seed(),
        matchup.get_division(),
        matchup.get_conference(),
        'NHL']
        for _, matchup in prev_round_matchups.items()
        ]
    
    series_matchups_preview = {}
    matchups_dict = {}

    for prev_series_winner in series_winners_list:
        if prev_series_winner[round_id] not in series_matchups_preview.keys():
            series_matchups_preview.update({prev_series_winner[round_id]: [[prev_series_winner[0], prev_series_winner[1], prev_series_winner[2], prev_series_winner[3]]]})
        else:
            series_matchups_preview[prev_series_winner[round_id]].append([prev_series_winner[0], prev_series_winner[1], prev_series_winner[2], prev_series_winner[3]])

    # add the winners from the first round to the second round matchups dictionary and reconfigure the result list
    for ind, (_, matchup) in enumerate(series_matchups_preview.items()):
        if matchup[0][1] < matchup[1][1]: # if the first team has a better seed than the second team, they are the home team
            matchups_dict.update({ind: PlayoffMatchup(
                matchup[0][0], # team 1 name
                matchup[1][0], # team 2 name
                matchup[0][1], # team 1 conference seed
                matchup[1][1], # team 2 conference seed
                round_id, # playoff round number
                matchup[0][3], # conference name
                matchup[0][2] # division name
            )})
        else:
            matchups_dict.update({ind: PlayoffMatchup(
                matchup[1][0], # team 1 name
                matchup[0][0], # team 2 name
                matchup[1][1], # team 1 conference seed
                matchup[0][1], # team 2 conference seed
                round_id, # playoff round number
                matchup[0][3], # conference name
                matchup[0][2] # division name
            )})
    
    return matchups_dict


def create_playoff_round_schedule(all_matchups, venue_map_df, feature_df, playoff_df):

    # if the playoff dataframe is empty, take the round start date from the regular season
    if playoff_df.empty:
        round_stdt = feature_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)
    else:
        round_stdt = playoff_df[cons.game_date_col].max() + pd.Timedelta(days=cons.playoff_round_buffer)

    # loop through matchups
    for matchup_num, matchup in all_matchups.items():

        # if the matchup is an western matchup, start series on matchday 2
        if matchup.get_conference() == 'Western':
            game_dt = round_stdt + pd.Timedelta(days=1)
            sched_format = cons.playoff_sched_format
        # if the matchup is an eastern matchup, start series on matchday 1
        elif matchup.get_conference() == 'Eastern':
            game_dt = round_stdt
            sched_format = cons.playoff_sched_format
        # if the matchup is the final, start series on matchday 1
        else:
            game_dt = round_stdt
            sched_format = cons.final_sched_format
        
        # list of game dates for the series
        game_dts = [game_dt + pd.Timedelta(days=val) for val in sched_format]

        # list of home and away teams for the series (higher seed is home first)
        home_teams = [matchup.get_team1()] * 2 + [matchup.get_team2()] * 2 + [matchup.get_team1()] + [matchup.get_team2()] + [matchup.get_team1()]
        away_teams = [matchup.get_team2()] * 2 + [matchup.get_team1()] * 2 + [matchup.get_team2()] + [matchup.get_team1()] + [matchup.get_team2()]

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


def series_final_check(playoff_df, playoff_df_filt, all_matchups, game_dt):

    # check if there were any games played where a series could have been won
    series_win_check_df = playoff_df_filt.loc[(playoff_df_filt[cons.game_date_col]==max(playoff_df_filt[cons.game_date_col])) &
                                                  ((playoff_df_filt[cons.home_team_series_score_col] == 3) |
                                                   (playoff_df_filt[cons.away_team_series_score_col] == 3))]
    
    matchup_map = [[matchup.get_teams(), matchup_num] for matchup_num, matchup in all_matchups.items()]
    
    # loop through all games from previous game date
    for _, row in series_win_check_df.iterrows():

        # find the index for this matchup in the matchups list
        for matchup in matchup_map:
            if row[cons.home_team_name_col] in matchup[0]:
                matchup_ind = matchup[1]
                break

        # initialize series win flags for both teams, game 7 indicator
        home_team_wins, away_team_wins = False, False
        game_seven = bool((row[cons.home_team_series_score_col] == 3) and (row[cons.away_team_series_score_col] == 3))

        # if the team that was leading in the series won, the series is over
        if (row[cons.home_team_series_score_col] == 3 and row[cons.home_team_score_col] > row[cons.away_team_score_col]):
            home_team_wins = True
            all_matchups[matchup_ind].set_series_results(row[cons.home_team_name_col], row[cons.away_team_name_col])
            
        elif (row[cons.away_team_series_score_col] == 3 and row[cons.away_team_score_col] > row[cons.home_team_score_col]):
            away_team_wins = True
            all_matchups[matchup_ind].set_series_results(row[cons.away_team_name_col], row[cons.home_team_name_col])

        # if either team won and it was not game 7, remove all future scheduled series games
        if (home_team_wins or away_team_wins) and not game_seven:
            indeces_drop = playoff_df.loc[(playoff_df[cons.game_date_col] > game_dt) &
                                                    (((playoff_df[cons.home_team_name_col] == row[cons.home_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.away_team_name_col])) |
                                                    ((playoff_df[cons.home_team_name_col] == row[cons.away_team_name_col]) &
                                                    (playoff_df[cons.away_team_name_col] == row[cons.home_team_name_col])))].index
            playoff_df.drop(index=indeces_drop, inplace=True)

        # print out the series results
        if home_team_wins:
            print(f'\t\t{row[cons.home_team_name_col]} advance past {row[cons.away_team_name_col]}: {int(row[cons.home_team_series_score_col]+1)}-{int(row[cons.away_team_series_score_col])}')
        elif away_team_wins:
            print(f'\t\t{row[cons.away_team_name_col]} advance past {row[cons.home_team_name_col]}: {int(row[cons.away_team_series_score_col]+1)}-{int(row[cons.home_team_series_score_col])}')
    
    return playoff_df, all_matchups


if __name__ == '__main__':
    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)
    feature_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))
    
    final_standings_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.final_standings_filename.format(date=today_dt))

    playoff_tree_predictions(feature_df, final_standings_df, False)