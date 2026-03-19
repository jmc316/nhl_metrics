import numpy as np
import pandas as pd
import constants as cons
import terminal_ui as tui

from file_utils import csvLoad, csvSave
from tabulate import tabulate
from constants import nhl_client


def nhl_team_stats():
    print('Fetching NHL team stats...')

    # display nhl team stats ui
    team_stats_ui = tui.team_stats_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.team_stats_options.items()}

    # call the function associated with the user's choice
    func_map[team_stats_ui.get_response()]()


def nhl_team_standings():
    print('Fetching NHL team standings...')

    # Fetch the standings
    league_df = pd.DataFrame(nhl_client.standings.league_standings()['standings'])

    # eastern conference playoff seeding
    east_conf_spots = league_df.loc[
        (league_df[cons.conference_name_col]=='Eastern')].sort_values(by=[cons.wildcard_sequence_col, cons.division_name_col])
    
    # western conference playoff seeding
    west_conf_spots = league_df.loc[
        (league_df[cons.conference_name_col]=='Western')].sort_values(by=[cons.wildcard_sequence_col, cons.division_name_col])

    print('--- Eastern Conference Wild Card ---')
    for _, team in east_conf_spots.iterrows():
        if team[cons.wildcard_sequence_col] == 2:
            print(team[cons.wildcard_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
            print('------------')
        elif team[cons.division_sequence_col] == 3:
            print(team[cons.division_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
            print('------------')
        elif team[cons.wildcard_sequence_col] == 0:
            print(team[cons.division_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
        else:
            print(team[cons.wildcard_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])

    print('\n--- Western Conference Wild Card ---')
    for _, team in west_conf_spots.iterrows():
        if team[cons.wildcard_sequence_col] == 2:
            print(team[cons.wildcard_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
            print('------------')
        elif team[cons.division_sequence_col] == 3:
            print(team[cons.division_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
            print('------------')
        elif team[cons.wildcard_sequence_col] == 0:
            print(team[cons.division_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])
        else:
            print(team[cons.wildcard_sequence_col], team[cons.team_name_col][cons.default_col], '\t', team['points'])

    print()


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
    if to_csv:
        col_order = [cons.game_id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col,
                     cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col,
                     cons.last_period_col, cons.home_team_points_col, cons.away_team_points_col]
        csvSave(season_results[col_order], cons.output_folder, cons.season_sched_pred_filename)

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
        season_results = csvLoad(cons.output_folder, cons.season_sched_pred_filename)

    print('Generating final standings...')

    # filter the final schedule on the current season
    # season_results[cons.season_name_col] = season_results[cons.season_name_col].astype(str)
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
    team_info_df = team_info()
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
        csvSave(final_standings, cons.output_folder, cons.final_standings_filename)

    return final_standings


def playoff_probabilities_printer(count_df):

    percent_cols = [cons.team_name_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%', f'{cons.missed_val}_%', f'{cons.playoff_per_col}']

    count_df = count_df.sort_values(by=[cons.playoff_per_col, f'{cons.div_1_val}_%', f'{cons.div_2_val}_%', f'{cons.div_3_val}_%', f'{cons.wc_1_val}_%', f'{cons.wc_2_val}_%'], ascending=False).reset_index(drop=True)[percent_cols]
    print('\nPlayoff Spot Probabilities:')

    print(tabulate(count_df, headers='keys', tablefmt='grid'))