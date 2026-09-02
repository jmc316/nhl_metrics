import pandas as pd
import requests
import constants as cons
import os

from nhlpy import NHLClient
from schedule import load_sched_df

# Create an instance of the NHLClient
nhl_client = NHLClient()


def backfill():

    # load all schedule data
    sched_df = load_sched_df()
    new_data_df = pd.DataFrame()
    valid_event_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'penalty']
    all_event_types = []

    # data_df = pd.read_csv('season_roster_data.csv')

    # season_roster_df = pd.DataFrame()
    # for seasonName in sched_df[cons.season_name_col].unique():

    #     # if seasonName in ['20212022', '20222023']:
    #     #     continue

    #     for team_name in cons.team_name_addrev_map.keys():

    #         if team_name not in sched_df.loc[sched_df[cons.season_name_col]==seasonName, cons.home_team_name_col].unique():
    #             continue

    #         print(f'{seasonName} - {team_name}')

    #         team_abbrev = cons.team_name_addrev_map[team_name]
    #         season_roster_data = nhl_client.teams.team_roster(team_abbr=team_abbrev, season=seasonName)

    #         for position in ['forwards', 'defensemen', 'goalies']:
    #             for player in season_roster_data[position]:

    #                 if not player['lastName']:
    #                     continue

    #                 player_career_stats = nhl_client.stats.player_career_stats(player_id=player['id'])['seasonTotals']
    #                 player_career_stats = [season for season in player_career_stats if season['leagueAbbrev'] == 'NHL']

    #                 if not player_career_stats:
    #                     continue

    #                 if not season_roster_df.empty:
    #                     if player['id'] in season_roster_df['playerId'].values:
    #                          continue
    #                 print(f"\t... {player['firstName']['default']} {player['lastName']['default']}")

    #                 player_career_stats_df = pd.DataFrame(player_career_stats)
    #                 player_career_stats_df['teamName'] = player_career_stats_df['teamName'].str.get('default')
    #                 player_career_stats_df['playerId'] = player['id']
    #                 player_career_stats_df['firstName'] = player['firstName']['default']
    #                 player_career_stats_df['lastName'] = player['lastName']['default']
    #                 player_career_stats_df['position'] = position

    #                 if position != 'goalies':
    #                     player_career_stats_df['avgToi'] = pd.to_timedelta("00:" + player_career_stats_df['avgToi']).dt.total_seconds().astype(int)
    #                     player_career_stats_df = player_career_stats_df[['playerId', 'firstName', 'lastName', 'position', 'season', 'sequence',
    #                                                                      'teamName', 'gameTypeId', 'gamesPlayed', 'assists', 'avgToi',
    #                                                                      'faceoffWinningPctg', 'gameWinningGoals', 'goals', 'otGoals',
    #                                                                      'pim', 'plusMinus', 'points', 'powerPlayGoals', 'powerPlayPoints',
    #                                                                      'shootingPctg', 'shorthandedGoals',
    #                                                                      'shorthandedPoints', 'shots']]
    #                 else:
    #                     if 'savePctg' not in player_career_stats_df.columns:
    #                         player_career_stats_df['savePctg'] = 1.0
    #                     player_career_stats_df['timeOnIce'] = pd.to_timedelta("00:" + player_career_stats_df['timeOnIce']).dt.total_seconds().astype(int)
    #                     player_career_stats_df = player_career_stats_df[['playerId', 'firstName', 'lastName', 'position', 'season', 'sequence',
    #                                                                      'teamName', 'gameTypeId', 'gamesPlayed', 'gamesStarted', 'assists',
    #                                                                      'goals', 'goalsAgainst', 'goalsAgainstAvg', 'losses', 'otLosses',
    #                                                                      'pim', 'savePctg', 'shotsAgainst', 'shutouts',
    #                                                                      'timeOnIce', 'wins']]
                    
    #                 season_roster_df = pd.concat([season_roster_df, player_career_stats_df], ignore_index=True)

    #                 pass

    #     season_roster_df = season_roster_df.sort_values(by=['lastName', 'firstName', 'season', 'sequence', 'gameTypeId'], ascending=True)
    #     season_roster_df.to_csv(f'season_roster_data_{seasonName}.csv', index=False)
                
    sched_df['homeTeamLineup'] = None
    sched_df['awayTeamLineup'] = None
    sched_df.set_index('gameId', inplace=True)

    for gameId in sched_df.index.unique():

        print(gameId)

        url = f"https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore"
        try:
            boxscore_data = requests.get(url).json()
        except:
            print(f"Failed to fetch data for game {gameId}. Skipping...")
            continue

        game_data = nhl_client.game_center.play_by_play(gameId)
        home_team_id = game_data['homeTeam']['id']
        away_team_id = game_data['awayTeam']['id']

        roster_data = game_data['rosterSpots']
        home_team_roster = [player['playerId'] for player in roster_data if player['teamId']==home_team_id]
        away_team_roster = [player['playerId'] for player in roster_data if player['teamId']==away_team_id]

        sched_df.at[gameId, 'homeTeamLineup'] = sorted(home_team_roster)
        sched_df.at[gameId, 'awayTeamLineup'] = sorted(away_team_roster)

    for season in sched_df[cons.season_name_col].unique():
        print(f'Writing new data for season: {season}...')
        sched_df.loc[sched_df[cons.season_name_col] == season].to_csv(f'output/season_schedules/{season}_season_sched.csv', index=False)

    # pass

if __name__ == '__main__':
    backfill()