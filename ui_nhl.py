import constants as cons
import terminal_ui as tui
import nhl_client as nhlc


def nhl_team_stats():
    print('Fetching NHL team stats...')

    # display nhl team stats ui
    team_stats_ui = tui.team_stats_screen()

    func_map = {option: getattr(__import__(module), func) for option, (module, func) in cons.team_stats_options.items()}

    # call the function associated with the user's choice
    func_map[team_stats_ui.get_response()]()


def nhl_individual_team_stats():
    print('Fetching individual NHL team stats...')

    # Fetch the teams
    teams = nhlc.get_team_stats()

    print(teams)


def team_info():
    
    teams_df = nhlc.get_team_stats()
        
    teams_df.rename(columns={'name': cons.team_name_col}, inplace=True)

    teams_df[cons.conference_name_col] = teams_df['conference'].apply(lambda x: x['name'])
    teams_df[cons.division_name_col] = teams_df['division'].apply(lambda x: x['name'])

    teams_df = teams_df[[cons.team_name_col, cons.conference_name_col, cons.division_name_col]]

    return teams_df