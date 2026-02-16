import constants as cons
import terminal_ui as tui
import pandas as pd
import numpy as np

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

    for col in league_df.columns:
        if isinstance(league_df[col], np.int64):
            league_df[col] = league_df[col].astype(int)

    # eastern conference playoff seeding
    east_conf_spots = league_df.loc[
        (league_df['conferenceName']=='Eastern')].sort_values(by=['wildcardSequence', 'divisionName'])
    
    # western conference playoff seeding
    west_conf_spots = league_df.loc[
        (league_df['conferenceName']=='Western')].sort_values(by=['wildcardSequence', 'divisionName'])

    print('--- Eastern Conference Wild Card ---')
    for _, team in east_conf_spots.iterrows():
        if team['wildcardSequence'] == 2:
            print(team['wildcardSequence'], team['teamName']['default'], '\t', team['points'])
            print('------------')
        elif team['divisionSequence'] == 3:
            print(team['divisionSequence'], team['teamName']['default'], '\t', team['points'])
            print('------------')
        elif team['wildcardSequence'] == 0:
            print(team['divisionSequence'], team['teamName']['default'], '\t', team['points'])
        else:
            print(team['wildcardSequence'], team['teamName']['default'], '\t', team['points'])

    print('\n--- Western Conference Wild Card ---')
    for _, team in west_conf_spots.iterrows():
        if team['wildcardSequence'] == 2:
            print(team['wildcardSequence'], team['teamName']['default'], '\t', team['points'])
            print('------------')
        elif team['divisionSequence'] == 3:
            print(team['divisionSequence'], team['teamName']['default'], '\t', team['points'])
            print('------------')
        elif team['wildcardSequence'] == 0:
            print(team['divisionSequence'], team['teamName']['default'], '\t', team['points'])
        else:
            print(team['wildcardSequence'], team['teamName']['default'], '\t', team['points'])

    print()


def nhl_individual_team_stats():
    print('Fetching individual NHL team stats...')

    # Fetch the teams
    teams = nhl_client.teams.teams()

    print(teams)