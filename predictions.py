import os
import constants as cons
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder


# def nhl_predictions():
#     print('Fetching NHL predictions...')

#     complete_games = pd.read_csv('annual_sched_new.csv')


def nhl_predictions():

    annual_sched = create_season_df()

    annual_sched['startTimeUTC'] = pd.to_datetime(annual_sched['startTimeUTC'])
    annual_sched['gameYear'] = annual_sched['startTimeUTC'].dt.year
    annual_sched['gameMonth'] = annual_sched['startTimeUTC'].dt.month
    annual_sched['gameDay'] = annual_sched['startTimeUTC'].dt.day
    annual_sched['gameTime'] = annual_sched['startTimeUTC'].dt.hour * 60 + annual_sched['startTimeUTC'].dt.minute
    annual_sched.drop(columns=['startTimeUTC'], inplace=True)

    encoder = OrdinalEncoder()
    annual_sched['venueTimezone'] = encoder.fit_transform(annual_sched[['venueTimezone']])
    annual_sched['venue'] = encoder.fit_transform(annual_sched[['venue']])
    annual_sched['lastPeriod'] = encoder.fit_transform(annual_sched[['lastPeriod']])

    feature_cols = ['id', 'season', 'gameYear', 'gameMonth', 'gameDay', 'gameTime', 'venueTimezone', 'venue', 'awayTeamId', 'homeTeamId']
    predict_cols = ['awayTeamScore', 'homeTeamScore', 'lastPeriod']

    annual_sched = annual_sched.replace({None: np.nan}, inplace=True) 
    train_df = annual_sched.dropna()

    X_train = train_df[feature_cols]
    y_train = train_df[predict_cols]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Identify rows with missing 'target' values and predict
    to_predict = annual_sched[annual_sched['lastPeriod'].isna()]
    X_to_predict = to_predict[feature_cols]
    predictions = model.predict(X_to_predict)

    # Add predictions back into the original DataFrame
    to_predict[predict_cols] = predictions 
    annual_sched.update(to_predict)

    annual_sched.to_csv('annual_sched_with_predictions.csv', index=False)

    print()


def create_season_df():
    print('Fetching NHL predictions...')

    if 'annual_sched_new.csv' in os.listdir():
        return pd.read_csv('annual_sched_new.csv')

    weekly_sched = cons.nhl_client.schedule.weekly_schedule(date="2026-01-04")

    # get the first day of the season
    first_day = '2025-10-01'
    last_day = '2026-04-30'

    annual_sched = pd.DataFrame()

    for week in pd.date_range(start=first_day, end=last_day, freq='W'):
        print(f'\t... {week.strftime('%Y-%m-%d')} ...')
        for dow in range(0, 6):
            weekly_sched_raw = pd.DataFrame(cons.nhl_client.schedule.weekly_schedule(date=week.strftime('%Y-%m-%d'))['gameWeek'][dow]['games'])

            # if there are no games in this week, skip to the next week
            if weekly_sched_raw.empty:
                continue

            # initialize columns that are the same as the raw data
            weekly_sched = weekly_sched_raw[['id', 'season', 'startTimeUTC', 'venueTimezone']]

            # create columns that are derived from the raw data
            weekly_sched['venue'] = [item['default'] for item in weekly_sched_raw['venue']]
            weekly_sched['awayTeamId'] = [item['id'] for item in weekly_sched_raw['awayTeam']]
            weekly_sched['awayTeamName'] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw['awayTeam']]
            weekly_sched['homeTeamId'] = [item['id'] for item in weekly_sched_raw['homeTeam']]
            weekly_sched['homeTeamName'] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw['homeTeam']]

            if 'gameOutcome' in weekly_sched_raw.columns:
                weekly_sched['awayTeamScore'] = [item['score'] for item in weekly_sched_raw['awayTeam']]
                weekly_sched['homeTeamScore'] = [item['score'] for item in weekly_sched_raw['homeTeam']]
                weekly_sched['lastPeriod'] = [item['lastPeriodType'] for item in weekly_sched_raw['gameOutcome']]
            else:
                weekly_sched['awayTeamScore'] = None
                weekly_sched['homeTeamScore'] = None
                weekly_sched['lastPeriod'] = None
            
            # only include NHL games (team ids are less than 33)
            weekly_sched = weekly_sched.loc[(weekly_sched['awayTeamId'] < 33) & (weekly_sched['awayTeamId'] >= 0)]

            annual_sched = pd.concat([annual_sched, weekly_sched], ignore_index=True)

    # annual_sched.to_csv('annual_sched_new.csv', index=False)

    return annual_sched