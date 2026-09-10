import pandas as pd
import constants as cons
from nhl_client import get_team_goalies

def predict_starting_goalies_simple(sched_df, player_df):

    # data taken from PuckPedia
    set_starting_goalies = {
        'Anaheim Ducks': 'L. Dostal',
        'Boston Bruins': 'J. Swayman',
        'Buffalo Sabres': 'U. Luukkonen',
        'Calgary Flames': 'D. Wolf',
        'Carolina Hurricanes': 'B. Bussi',
        'Chicago Blackhawks': 'S. Knight',
        'Colorado Avalanche': 'S. Wedgewood',
        'Columbus Blue Jackets': 'J. Greaves',
        'Dallas Stars': 'J. Oettinger',
        'Detroit Red Wings': 'J. Gibson',
        'Edmonton Oilers': 'T. Jarry',
        'Florida Panthers': 'J. Markstrom',
        'Los Angeles Kings': 'D. Kuemper',
        'Minnesota Wild': 'J. Wallstedt',
        'Montréal Canadiens': 'J. Dobes',
        'Nashville Predators': 'J. Saros',
        'New Jersey Devils': 'J. Allen',
        'New York Islanders': 'I. Sorokin',
        'New York Rangers': 'I. Shesterkin',
        'Ottawa Senators': 'L. Ullmark',
        'Philadelphia Flyers': 'D. Vladar',
        'Pittsburgh Penguins': 'A. Silovs',
        'San Jose Sharks': 'Y. Askarov',
        'Seattle Kraken': 'J. Daccord',
        'St. Louis Blues': 'J. Hofer',
        'Tampa Bay Lightning': 'A. Vasilevskiy',
        'Toronto Maple Leafs': 'S. Bobrovsky',
        'Utah Mammoth': 'K. Vejmelka',
        'Vancouver Canucks': 'K. Lankinen',
        'Vegas Golden Knights': 'C. Hart',
        'Washington Capitals': 'L. Thompson',
        'Winnipeg Jets': 'C. Hellebuyck',
    }

    sched_df_cur = sched_df.loc[sched_df[cons.season_name_col] == sched_df[cons.season_name_col].max()]

    goalies_per_team = {}

    # get the current goalies for every team
    print('\nGenerating current goalies per team...')
    for team_name in sorted(sched_df_cur[cons.home_team_name_col].unique()):
        
        goalies_per_team[team_name] = get_team_goalies(sched_df, player_df=player_df, team_name=team_name)

    # get the starting goalie for each team based on each team's highest valued goalie
    starting_goalies = {}
    for team_name, goalies in goalies_per_team.items():

        top_goalie = [goalie for goalie in goalies if set_starting_goalies[team_name]==goalie['goalie_name']][0]

        # top_goalie = max(goalies, key=lambda g: g['value'])
        # print(f"{team_name} starting goalie: {top_goalie['goalie_name']} (Value: {top_goalie['value']})")
        starting_goalies[team_name] = top_goalie

    # SIMPLE SOLUTION, MORE INTELLIGENT PREDICTION CAN BE IMPLEMENTED LATER BASED ON ADDITIONAL DATA
    # add the starting goalie as the starter for all future games
    for team_name, top_goalie in starting_goalies.items():
        sched_df.loc[(sched_df[cons.last_period_col].isna()) & (sched_df[cons.home_team_name_col] == team_name), 'home_goalie_id'] = top_goalie['goalie_id']
        sched_df.loc[(sched_df[cons.last_period_col].isna()) & (sched_df[cons.home_team_name_col] == team_name), 'home_goalie_name'] = top_goalie['goalie_name']
        sched_df.loc[(sched_df[cons.last_period_col].isna()) & (sched_df[cons.away_team_name_col] == team_name), 'away_goalie_id'] = top_goalie['goalie_id']
        sched_df.loc[(sched_df[cons.last_period_col].isna()) & (sched_df[cons.away_team_name_col] == team_name), 'away_goalie_name'] = top_goalie['goalie_name']

    return sched_df


def predict_starting_goalies(sched_df, player_df):

    sched_df_cur = sched_df.loc[sched_df[cons.season_name_col] == sched_df[cons.season_name_col].max()]

    goalies_per_team = {}

    # get the current goalies for every team
    print('\nGenerating current goalies per team...')
    for team_name in sched_df_cur[cons.home_team_name_col].unique():
        
        goalies_per_team[team_name] = get_team_goalies(sched_df, player_df=player_df, team_name=team_name)

    print('Creating starting goalie data...')
    start_goalie_df = create_starting_goalie_data(sched_df)

    pass

    return sched_df


def create_starting_goalie_data(sched_df):

    start_goalie_df = pd.DataFrame()

    # for every game in sched_df, create two rows in the start_goalie_df: one for the home team and one for the away team
    for _, game in sched_df.iterrows():
        start_goalie_df = pd.concat([
            start_goalie_df,
            pd.DataFrame([{
                cons.game_id_col: game[cons.game_id_col],
                cons.team_name_col: game[cons.home_team_name_col],
                cons.start_goalie_id_col: game['home_goalie_id'],
                cons.start_goalie_name_col: game['home_goalie_name']
            }])
        ], ignore_index=True)
        start_goalie_df = pd.concat([
            start_goalie_df,
            pd.DataFrame([{
                cons.game_id_col: game[cons.game_id_col],
                cons.team_name_col: game[cons.away_team_name_col],
                cons.start_goalie_id_col: game['away_goalie_id'],
                cons.start_goalie_name_col: game['away_goalie_name']
            }])
        ], ignore_index=True)

    return start_goalie_df.sort_values(by=[cons.game_id_col, cons.team_name_col])