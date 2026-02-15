import constants as cons
import terminal_ui as tui

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
    standings = nhl_client.standings.league_standings()

    print(standings)


def nhl_individual_team_stats():
    print('Fetching individual NHL team stats...')

    # Fetch the teams
    teams = nhl_client.teams.teams()

    print(teams)