import pandas as pd
import requests
import constants as cons

from nhlpy import NHLClient
from schedule import load_sched_df

# Create an instance of the NHLClient
nhl_client = NHLClient()


def backfill():

    # load all schedule data
    # sched_df = load_sched_df()
    new_data_df = pd.DataFrame()
    valid_event_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'penalty']
    all_event_types = []
    gameids_2020 = range(2020020001, 2020040001)  # Example range for 2020 season game IDs
    # 2021020001, 2022020001

    for gameId in gameids_2020:  # sched_df[cons.game_id_col].unique():

        print(gameId)

        if str(gameId)[:4] != '2020':
            continue

        url = f"https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore"
        try:
            boxscore_data = requests.get(url).json()
        except:
            print(f"Failed to fetch data for game {gameId}. Skipping...")
            continue
        home_goalies = boxscore_data['playerByGameStats']['homeTeam']['goalies']
        away_goalies = boxscore_data['playerByGameStats']['awayTeam']['goalies']

        if not home_goalies and not away_goalies:
            print(f'No goalies found for game {gameId}. Skipping...')
            continue

        for goalie in home_goalies + away_goalies:

            if goalie['toi'] == '00:00':
                continue

            if goalie in home_goalies:
                team = 'home'
            else:
                team = 'away'

            new_data_df = pd.concat([new_data_df, pd.DataFrame([{
                cons.game_id_col: gameId,
                cons.starttime_est_col: pd.to_datetime(boxscore_data['startTimeUTC']).tz_convert(cons.est_tz).tz_localize(None),
                'team': team,
                'goalie_name': goalie['name']['default'],
                'goalie_id': goalie['playerId'],
                'starter': goalie['starter'],
                'toi_secs': int(goalie['toi'].split(':')[0])*60+int(goalie['toi'].split(':')[1]),
                'ev_shots_against': int(goalie['evenStrengthShotsAgainst'].split('/')[1]),
                'ev_saves': int(goalie['evenStrengthShotsAgainst'].split('/')[0]),
                'ev_goals_against': int(goalie['evenStrengthGoalsAgainst']),
                'sh_shots_against': int(goalie['shorthandedShotsAgainst'].split('/')[1]),
                'sh_saves': int(goalie['shorthandedShotsAgainst'].split('/')[0]),
                'sh_goals_against': int(goalie['shorthandedGoalsAgainst']),
                'pp_shots_against': int(goalie['powerPlayShotsAgainst'].split('/')[1]),
                'pp_saves': int(goalie['powerPlayShotsAgainst'].split('/')[0]),
                'pp_goals_against': int(goalie['powerPlayGoalsAgainst']),
                'tot_shots_against': goalie['shotsAgainst'],
                'tot_saves': goalie['saves'],
                'tot_goals_against': goalie['goalsAgainst'],
                'decision': goalie['decision'] if 'decision' in goalie else None,
            }])], ignore_index=True)

            if not goalie['starter']:
                pass

        pass

    new_data_df.to_csv('goalie_data.csv', index=False)

    # new_data_df = new_data_df.loc[new_data_df['starter'] == True].copy()

    # # change the columns of new_data_df to be home and away-specific
    # new_data_df_home = new_data_df.loc[new_data_df['team'] == 'home'].copy()
    # new_data_df_home = new_data_df_home.rename(columns={
    #     'goalie_name': 'home_goalie_name',
    #     'goalie_id': 'home_goalie_id',
    #     'starter': 'home_starter',
    #     'toi_secs': 'home_toi_secs',
    #     'ev_shots_against': 'home_ev_shots_against',
    #     'ev_saves': 'home_ev_saves',
    #     'ev_goals_against': 'home_ev_goals_against',
    #     'sh_shots_against': 'home_sh_shots_against',
    #     'sh_saves': 'home_sh_saves',
    #     'sh_goals_against': 'home_sh_goals_against',
    #     'pp_shots_against': 'home_pp_shots_against',
    #     'pp_saves': 'home_pp_saves',
    #     'pp_goals_against': 'home_pp_goals_against',
    #     'tot_shots_against': 'home_tot_shots_against',
    #     'tot_saves': 'home_tot_saves',
    #     'tot_goals_against': 'home_tot_goals_against',
    #     'decision': 'home_decision'
    # })
    # new_data_df_home = new_data_df_home.drop(columns=['team', cons.starttime_est_col])
    # sched_df = sched_df.merge(new_data_df_home, on=cons.game_id_col, how='left')

    # new_data_df_away = new_data_df.loc[new_data_df['team'] == 'away'].copy()
    # new_data_df_away = new_data_df_away.rename(columns={
    #     'goalie_name': 'away_goalie_name',
    #     'goalie_id': 'away_goalie_id',
    #     'starter': 'away_starter',
    #     'toi_secs': 'away_toi_secs',
    #     'ev_shots_against': 'away_ev_shots_against',
    #     'ev_saves': 'away_ev_saves',
    #     'ev_goals_against': 'away_ev_goals_against',
    #     'sh_shots_against': 'away_sh_shots_against',
    #     'sh_saves': 'away_sh_saves',
    #     'sh_goals_against': 'away_sh_goals_against',
    #     'pp_shots_against': 'away_pp_shots_against',
    #     'pp_saves': 'away_pp_saves',
    #     'pp_goals_against': 'away_pp_goals_against',
    #     'tot_shots_against': 'away_tot_shots_against',
    #     'tot_saves': 'away_tot_saves',
    #     'tot_goals_against': 'away_tot_goals_against',
    #     'decision': 'away_decision'
    # })
    # new_data_df_away = new_data_df_away.drop(columns=['team', cons.starttime_est_col])
    # sched_df = sched_df.merge(new_data_df_away, on=cons.game_id_col, how='left')

    # # for season in sched_df[cons.season_name_col].unique():
    # #     print(f'Writing new data for season: {season}...')
    # #     sched_df.loc[sched_df[cons.season_name_col] == season].to_csv(f'output/season_feature_sets/Goalie_features/{season}_goalie_features.csv', index=False)

    # pass

if __name__ == '__main__':
    backfill()