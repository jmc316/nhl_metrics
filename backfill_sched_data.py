import pandas as pd
import constants as cons

from nhlpy import NHLClient
from schedule import load_sched_df

# Create an instance of the NHLClient
nhl_client = NHLClient()


def backfill():

    # load all schedule data
    sched_df = load_sched_df()
    new_data = {}
    valid_event_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'penalty']
    all_event_types = []

    for gameId in sched_df[cons.game_id_col].unique():

        print(gameId)

        game_data = nhl_client.game_center.play_by_play(gameId)
        home_id = game_data['homeTeam']['id']
        away_id = game_data['awayTeam']['id']

        new_data[gameId] = {
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

        for play in game_data['plays']:
            event_type = play.get('typeDescKey')
            event_team = play.get('details', {}).get('eventOwnerTeamId')

            if event_type not in all_event_types:
                all_event_types.append(event_type)

            if event_type in valid_event_types:
                if event_team == home_id:
                    if event_type == 'shot-on-goal':
                        new_data[gameId]['home_shot-on-goal'] += 1
                    elif event_type == 'missed-shot':
                        new_data[gameId]['home_missed-shot'] += 1
                    elif event_type == 'blocked-shot':
                        new_data[gameId]['home_blocked-shot'] += 1
                    elif event_type == 'penalty':
                        new_data[gameId]['home_penalty'] += 1
                elif event_team == away_id:
                    if event_type == 'shot-on-goal':
                        new_data[gameId]['away_shot-on-goal'] += 1
                    elif event_type == 'missed-shot':
                        new_data[gameId]['away_missed-shot'] += 1
                    elif event_type == 'blocked-shot':
                        new_data[gameId]['away_blocked-shot'] += 1
                    elif event_type == 'penalty':
                        new_data[gameId]['away_penalty'] += 1

        match_summary = nhl_client.game_center.match_up(game_id=gameId)

        i = 0

        while i < len(match_summary['summary']['scoring']):
            goals_by_period = match_summary['summary']['scoring'][i]['goals']
            if len(goals_by_period) == 0:
                i += 1
                continue
            for goal_play in goals_by_period:
                if goal_play.get('strength') == 'pp':
                    if goal_play.get('isHome') == True:
                        new_data[gameId]['home_pp_goal'] += 1
                    else:
                        new_data[gameId]['away_pp_goal'] += 1
            i += 1

        pass

    new_data_df = pd.DataFrame(new_data).T.reset_index(names=cons.game_id_col)

    sched_df = sched_df.merge(new_data_df, on=cons.game_id_col, how='left')

    for season in sched_df[cons.season_name_col].unique():
        print(f'Writing new data for season: {season}...')
        sched_df.loc[sched_df[cons.season_name_col] == season].to_csv(f'output/season_schedules/{season}_season_sched_new.csv', index=False)

if __name__ == '__main__':
    backfill()