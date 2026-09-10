import pandas as pd
import constants as cons
from nhl_client import get_team_goalies


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