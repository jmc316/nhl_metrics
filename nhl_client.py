import time

import pandas as pd
import requests
import constants as cons
import utils.player_feat_utils as pl_ut

from nhlpy import NHLClient

# Create an instance of the NHLClient
nhl_client = NHLClient()


def get_sched_data(week, dow, is_historical):
    while True:
        try:
            # fetch the schedule data for this week and day of week from the NHL API
            weekly_sched_raw = pd.DataFrame(nhl_client.schedule.weekly_schedule(date=week.strftime("%Y-%m-%d"))['gameWeek'][dow]['games'])
        except Exception as ex:
            # re-try this week's schedule if there was a timeout error
            print(f'\t\t... {ex} ...')
            time.sleep(cons.api_timeout_wait_time)

            continue

        # if there are no games in this week, skip to the next week
        if weekly_sched_raw.empty:
            return pd.DataFrame()

        # initialize columns that are the same as the raw data
        weekly_sched = weekly_sched_raw[['id', cons.season_col, cons.game_type_col, cons.starttime_utc_col, cons.venue_timezone_col]]

        weekly_sched.rename(columns={'id': cons.game_id_col, 'season': cons.season_name_col}, inplace=True)

        # only include NHL games (gameType 2 = regular season, 3 = playoffs)
        weekly_sched = weekly_sched.loc[weekly_sched[cons.game_type_col].isin([2, 3])]
        weekly_sched_raw = weekly_sched_raw.loc[weekly_sched_raw[cons.game_type_col].isin([2, 3])]

        # if there are no valid NHL games in this week, skip to the next week
        if weekly_sched.empty:
            return pd.DataFrame()

        # create columns that are derived from the raw data
        weekly_sched[cons.venue_col] = [item['default'] for item in weekly_sched_raw['venue']]
        weekly_sched[cons.away_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw[cons.away_team_col]]
        weekly_sched[cons.home_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw[cons.home_team_col]]

        # if the game has already been played, extract the scores and last period type from the raw data; otherwise, set these columns to None for now and they will be filled in with predictions later
        if cons.game_outcome_col in weekly_sched_raw.columns:
            weekly_sched[cons.away_team_score_col] = [item['score'] for item in weekly_sched_raw[cons.away_team_col]]
            weekly_sched[cons.home_team_score_col] = [item['score'] for item in weekly_sched_raw[cons.home_team_col]]
            weekly_sched[cons.last_period_col] = [item['lastPeriodType'] for item in weekly_sched_raw[cons.game_outcome_col]]
        else:
            weekly_sched[cons.away_team_score_col] = None
            weekly_sched[cons.home_team_score_col] = None
            weekly_sched[cons.last_period_col] = None

        # if there have been games that have already been played, need to fill in more data
        if is_historical:
            for gameId in weekly_sched[cons.game_id_col]:

                pbp_data = nhl_client.game_center.play_by_play(gameId)

                # get the lineup data
                home_team_roster, away_team_roster = get_game_lineups(gameId, pbp_data=pbp_data)
                weekly_sched.at[gameId, 'homeTeamLineup'] = home_team_roster
                weekly_sched.at[gameId, 'awayTeamLineup'] = away_team_roster

                # get the goalie data
                goalie_df = get_goalie_data(gameId)
                weekly_sched = pd.merge(weekly_sched, goalie_df, how='left', on=cons.game_id_col)

                # get the team stats data
                team_df = get_team_data(gameId, pbp_data=pbp_data)
                weekly_sched = pd.merge(weekly_sched, team_df, how='left', on=cons.game_id_col)

        if not is_historical:
            for gameId in weekly_sched[cons.game_id_col]:

                # fill in any known starting goalies for future games
                weekly_sched = fill_known_starting_goalies(weekly_sched, gameId)


        return weekly_sched


def fill_known_starting_goalies(weekly_sched, gameId):

    # implement this in future when data becomes available
    pass

    return weekly_sched


def get_team_goalies(sched_df, player_df, team_name):

    goalies_list = []
    team_abbv = cons.team_name_addrev_map[team_name]
    cur_season = sched_df[cons.season_name_col].max()

    # get the goalies listed on the team roster
    team_roster_goalies_data = nhl_client.teams.team_roster(team_abbr=team_abbv, season=cur_season)['goalies']
    team_roster_goalies = {goalie['id']: f'{goalie['firstName']['default'][0]}. {goalie['lastName']['default']}' for goalie in team_roster_goalies_data}

    # get the last two goalies to start games for this team in home or away games
    home_starts = sched_df.loc[(sched_df[cons.home_team_name_col] == team_name) & (sched_df['home_starter'])]
    home_starts['goalie_name'] = home_starts['home_goalie_name']
    home_starts['goalie_id'] = home_starts['home_goalie_id']
    away_starts = sched_df.loc[(sched_df[cons.away_team_name_col] == team_name) & (sched_df['away_starter'])]
    away_starts['goalie_name'] = away_starts['away_goalie_name']
    away_starts['goalie_id'] = away_starts['away_goalie_id']
    team_starts = pd.concat([home_starts, away_starts]).sort_values(by=cons.starttime_utc_col, ascending=False)[['goalie_name', 'goalie_id']].drop_duplicates()
    prior_start_goalies = {int(row['goalie_id']): row['goalie_name'] for _, row in team_starts.head(2).iterrows()}

    pass

    # loop through the prior starting goalies
    for goalie_id in prior_start_goalies.keys():
        if goalie_id in team_roster_goalies:
            goalies_list.append({'goalie_id': goalie_id, 'goalie_name': team_roster_goalies[goalie_id]})

    if len(goalies_list) < 2:

        # add player value to each goalie on the team roster
        for goalie_id in team_roster_goalies.keys():
            if goalie_id in list(player_df['playerId']):
                value = player_df.loc[(player_df['playerId']==goalie_id), 'value'].values[0]
            else:
                value = pl_ut.DEFAULT_VALUE

            team_roster_goalies[goalie_id] = {
                'goalie_name': team_roster_goalies[goalie_id],
                'value': value
            }
    
        # sort the goalies by value
        team_roster_goalies = dict(sorted(team_roster_goalies.items(), key=lambda item: item[1]['value'], reverse=True))

        # fill in with the highest value goalies from the team roster
        for goalie_id, goalie_info in team_roster_goalies.items():
            if goalie_id not in [g['goalie_id'] for g in goalies_list]:
                # print(f'\tAdding goalie {team_roster_goalies[goalie_id]["goalie_name"]}')
                goalies_list.append({'goalie_id': goalie_id, 'goalie_name': goalie_info['goalie_name']})
            if len(goalies_list) >= 2:
                break

    return goalies_list


def get_team_data(gameId, pbp_data=None):

    if pbp_data is None:
        pbp_data = nhl_client.game_center.play_by_play(gameId)

    team_data = {}
    valid_event_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'penalty']

    home_id = pbp_data['homeTeam']['id']
    away_id = pbp_data['awayTeam']['id']

    team_data[gameId] = {
        'home_shot-on-goal': 0,
        'away_shot-on-goal': 0,
        'home_missed-shot': 0,
        'away_missed-shot': 0,
        'home_blocked-shot': 0,
        'away_blocked-shot': 0,
        'home_penalty': 0,
        'away_penalty': 0,
        'home_pp_goal': 0,
        'away_pp_goal': 0,
    }

    # loop through each recorded play in the game and accumulate important stats
    for play in pbp_data['plays']:
        event_type = play.get('typeDescKey')
        event_team = play.get('details', {}).get('eventOwnerTeamId')

        if event_type in valid_event_types:
            if event_team == home_id:
                if event_type == 'shot-on-goal':
                    team_data[gameId]['home_shot-on-goal'] += 1
                elif event_type == 'missed-shot':
                    team_data[gameId]['home_missed-shot'] += 1
                elif event_type == 'blocked-shot':
                    team_data[gameId]['home_blocked-shot'] += 1
                elif event_type == 'penalty':
                    team_data[gameId]['home_penalty'] += 1
            elif event_team == away_id:
                if event_type == 'shot-on-goal':
                    team_data[gameId]['away_shot-on-goal'] += 1
                elif event_type == 'missed-shot':
                    team_data[gameId]['away_missed-shot'] += 1
                elif event_type == 'blocked-shot':
                    team_data[gameId]['away_blocked-shot'] += 1
                elif event_type == 'penalty':
                    team_data[gameId]['away_penalty'] += 1

    # need match summary for power play goals
    match_summary = nhl_client.game_center.match_up(game_id=gameId)

    # loop through the scoring summary and accumulate a count of PP goals
    i = 0
    while i < len(match_summary['summary']['scoring']):
        goals_by_period = match_summary['summary']['scoring'][i]['goals']
        if len(goals_by_period) == 0:
            i += 1
            continue
        for goal_play in goals_by_period:
            if goal_play.get('strength') == 'pp':
                if goal_play.get('isHome') == True:
                    team_data[gameId]['home_pp_goal'] += 1
                else:
                    team_data[gameId]['away_pp_goal'] += 1
        i += 1

    team_data_df = pd.DataFrame(team_data).T.reset_index(names=cons.game_id_col)

    return team_data_df


def get_goalie_data(gameId):

    goalie_df = pd.DataFrame()

    url = f"https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore"
    try:
        boxscore_data = requests.get(url).json()
    except:
        print(f"Failed to fetch data for game {gameId}. Skipping...")
        return None

    home_goalies = boxscore_data['playerByGameStats']['homeTeam']['goalies']
    away_goalies = boxscore_data['playerByGameStats']['awayTeam']['goalies']

    # loop through all goalies listed for this game and record info for the starters
    for goalie in home_goalies + away_goalies:

        if (goalie['toi'] == '00:00') or (not goalie['starter']):
            continue

        if goalie in home_goalies:
            team = 'home'
        else:
            team = 'away'

        goalie_df = pd.concat([goalie_df, pd.DataFrame([{
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

    return goalie_df


def get_game_lineups(gameId, pbp_data=None):

    if pbp_data == None:
        pbp_data = nhl_client.game_center.play_by_play(gameId)
                    
    home_team_id = pbp_data['homeTeam']['id']
    away_team_id = pbp_data['awayTeam']['id']

    roster_data = pbp_data['rosterSpots']
    home_team_roster = [player['playerId'] for player in roster_data if player['teamId']==home_team_id]
    away_team_roster = [player['playerId'] for player in roster_data if player['teamId']==away_team_id]

    return sorted(home_team_roster), sorted(away_team_roster)


def fill_future_lineup(sched_df, player_df):

    # the dataframe for the current season
    sched_df_cur = sched_df.loc[sched_df[cons.season_name_col] == sched_df[cons.season_name_col].max()]

    # if there have been no games played yet this season, construct lineups based off of offseason rosters
    if sched_df_cur.loc[sched_df_cur[cons.last_period_col].notna()].empty:

        # get the current lineup availability for each team
        team_lineups = {}
        cur_season = sched_df_cur[cons.season_name_col].max()

        print('\nGenerating team lineups from offseason rosters...')
        for team_name in sched_df_cur.loc[sched_df_cur[cons.season_name_col] == cur_season, cons.home_team_name_col].unique():
            team_roster_values = {}
            team_abbv = cons.team_name_addrev_map[team_name]
            team_roster = nhl_client.teams.team_roster(team_abbr=team_abbv, season=cur_season)
    
            # get the value for each player
            for position in team_roster:
                for player in team_roster[position]:
                    player_id = player['id']
                    if player_id in player_df['playerId'].values:
                        value = player_df.loc[player_df['playerId']==player_id, 'value'].iloc[0]
                    else:
                        value = pl_ut.DEFAULT_VALUE
                    team_roster_values.setdefault(position, {})[player_id] = value

            # get the top 12 forwards, 6 defensemen, and 2 goalies from the roster
            team_lineups[team_name] = {
                'forwards': sorted(team_roster_values.get('forwards', {}).items(), key=lambda x: x[1], reverse=True)[:12],
                'defensemen': sorted(team_roster_values.get('defensemen', {}).items(), key=lambda x: x[1], reverse=True)[:6],
                'goalies': sorted(team_roster_values.get('goalies', {}).items(), key=lambda x: x[1], reverse=True)[:2],
            }

            pass

        # fill in the future lineup data based on the current best roster
        for idx, row in sched_df.loc[sched_df[cons.season_name_col] == cur_season].iterrows():
            home_team_name = row[cons.home_team_name_col]
            away_team_name = row[cons.away_team_name_col]
            sched_df.at[idx, cons.home_lineup_col] = str([player_id for player_id, _ in team_lineups.get(home_team_name, {}).get('forwards', []) +
                                                                 team_lineups.get(home_team_name, {}).get('defensemen', []) +
                                                                 team_lineups.get(home_team_name, {}).get('goalies', [])])
            sched_df.at[idx, cons.away_lineup_col] = str([player_id for player_id, _ in team_lineups.get(away_team_name, {}).get('forwards', []) +
                                                                 team_lineups.get(away_team_name, {}).get('defensemen', []) +
                                                                 team_lineups.get(away_team_name, {}).get('goalies', [])])

    # if games have been played this season, construct lineups based on previous lineups
    else:
        # get the last available lineup for each team from the previous games
        last_lineups = {}

        for idx, row in sched_df_cur.iterrows():
            home_team_name = row[cons.home_team_name_col]
            away_team_name = row[cons.away_team_name_col]
            last_lineups[home_team_name] = row[cons.home_lineup_col]
            last_lineups[away_team_name] = row[cons.away_lineup_col]

        # fill in the future lineup data based on the last available lineups
        for idx, row in sched_df.loc[sched_df[cons.season_name_col] == cur_season].iterrows():
            home_team_name = row[cons.home_team_name_col]
            away_team_name = row[cons.away_team_name_col]
            sched_df.at[idx, cons.home_lineup_col] = last_lineups.get(home_team_name, '[]')
            sched_df.at[idx, cons.away_lineup_col] = last_lineups.get(away_team_name, '[]')

    return sched_df



def get_nhl_team_standings():
    print('Fetching live NHL team standings...')
    while True:
        try:
            # Fetch the standings
            data_df = pd.DataFrame(nhl_client.standings.league_standings()['standings'])
        except Exception as ex:
            # re-try this week's schedule if there was a timeout error
            print(f'\t\t... {ex} ...')
            time.sleep(cons.api_timeout_wait_time)

        finally:
            return data_df
        

# def get_team_stats():
#     print('Fetching live NHL team stats...')
#     while True:
#         try:
#             # Fetch the team stats
#             data_df = pd.DataFrame(nhl_client.teams.teams())
#         except Exception as ex:
#             # re-try this week's schedule if there was a timeout error
#             print(f'\t\t... {ex} ...')
#             time.sleep(cons.api_timeout_wait_time)

#         finally:
#             return data_df