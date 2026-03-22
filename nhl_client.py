import time

import pandas as pd
import constants as cons

from nhlpy import NHLClient

# Create an instance of the NHLClient
nhl_client = NHLClient()

def get_sched_data(week, dow):
    while True:
        try:
            # fetch the schedule data for this week and day of week from the NHL API
            weekly_sched_raw = pd.DataFrame(nhl_client.schedule.weekly_schedule(date=week.strftime("%Y-%m-%d"))['gameWeek'][dow]['games'])
        except Exception as ex:
            # re-try this week's schedule if there was a timeout error
            print(f'\t\t... {ex} ...')
            time.sleep(3)

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

        return weekly_sched