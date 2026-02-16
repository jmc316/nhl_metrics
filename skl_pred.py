import os
import random
import constants as cons
import pandas as pd
import numpy as np
import utils as ut
from termcolor import colored
from httpcore import ReadTimeout, ConnectError, ConnectTimeout
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder


def season_predictions(to_csv=False):

    # load or create the season schedule dataframe
    season_sched = create_season_df()

    # apply preprocessing to the model features
    season_sched = feature_preprocessing(season_sched)

    print('\nTraining model and generating predictions...')

    # handle missing values by dropping rows with missing target values and training the model on the remaining data
    season_sched = season_sched.replace({None: np.nan}, inplace=True) 
    train_df = season_sched.dropna()

    # train a random forest regressor on the training data
    X_train = train_df[cons.feature_cols]
    y_train = train_df[cons.predict_cols]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Identify rows with missing 'target' values and predict
    to_predict = season_sched[season_sched[cons.last_period_col].isna()]
    X_to_predict = to_predict[cons.feature_cols]
    predictions = model.predict(X_to_predict)

    # Add predictions back into the original DataFrame as floats
    to_predict[cons.predict_cols] = predictions
    season_sched.update(to_predict)

    # save the updated season schedule with predictions to a new CSV file
    # drop the feature columns that are no longer important
    if to_csv:
        col_order = [cons.id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col, cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col]
        season_sched[col_order].to_csv(cons.output_folder + cons.season_sched_pred_filename, index=False)

    return season_sched


def create_season_df():
    print('\nFetching NHL season schedule...')

    if cons.season_sched_filename in os.listdir(cons.output_folder):
        return pd.read_csv(cons.output_folder + cons.season_sched_filename)

    # get the first day of the season
    # this needs to be variable later on, but for now we can hardcode it
    first_day = '2025-10-01'
    last_day = '2026-04-30'

    season_sched = pd.DataFrame()

    for week in pd.date_range(start=first_day, end=last_day, freq='W'):
        print(f'\t... {week.strftime('%Y-%m-%d')} ...')
        for dow in range(0, 7):
            while True:
                try:
                    weekly_sched_raw = pd.DataFrame(cons.nhl_client.schedule.weekly_schedule(date=week.strftime('%Y-%m-%d'))['gameWeek'][dow]['games'])
                except Exception as ex:
                    # re-try this week's schedule if there was a timeout error
                    print(f'\t\t... {ex} ...')
                    continue

                # if there are no games in this week, skip to the next week
                if weekly_sched_raw.empty:
                    break

                # initialize columns that are the same as the raw data
                weekly_sched = weekly_sched_raw[[cons.id_col, cons.season_col, cons.starttime_utc_col, cons.venue_timezone_col]]

                # create columns that are derived from the raw data
                weekly_sched[cons.venue_col] = [item['default'] for item in weekly_sched_raw['venue']]
                weekly_sched[cons.away_team_id_col] = [item['id'] for item in weekly_sched_raw['awayTeam']]
                weekly_sched[cons.away_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw['awayTeam']]
                weekly_sched[cons.home_team_id_col] = [item['id'] for item in weekly_sched_raw['homeTeam']]
                weekly_sched[cons.home_team_name_col] = [item['placeName']['default'] + ' ' + item['commonName']['default'] for item in weekly_sched_raw['homeTeam']]

                if 'gameOutcome' in weekly_sched_raw.columns:
                    weekly_sched[cons.away_team_score_col] = [item['score'] for item in weekly_sched_raw['awayTeam']]
                    weekly_sched[cons.home_team_score_col] = [item['score'] for item in weekly_sched_raw['homeTeam']]
                    weekly_sched[cons.last_period_col] = [item['lastPeriodType'] for item in weekly_sched_raw['gameOutcome']]
                else:
                    weekly_sched[cons.away_team_score_col] = None
                    weekly_sched[cons.home_team_score_col] = None
                    weekly_sched[cons.last_period_col] = None
                
                # only include NHL games (team ids are less than 69) Why 69? Probably need a better way to filter out non-NHL games, but this works for now
                weekly_sched = weekly_sched.loc[(weekly_sched[cons.away_team_id_col] < 69) & (weekly_sched[cons.away_team_id_col] >= 0)]

                season_sched = pd.concat([season_sched, weekly_sched], ignore_index=True)

                break

    season_sched.to_csv(cons.output_folder + cons.season_sched_filename, index=False)

    return season_sched


def feature_preprocessing(season_sched):

    print('\nPreprocessing feature data...')

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    season_sched[cons.starttime_utc_col] = pd.to_datetime(season_sched[cons.starttime_utc_col])
    season_sched[cons.game_year_col] = season_sched[cons.starttime_utc_col].dt.year
    season_sched[cons.game_month_col] = season_sched[cons.starttime_utc_col].dt.month
    season_sched[cons.game_day_col] = season_sched[cons.starttime_utc_col].dt.day
    season_sched[cons.game_time_col] = season_sched[cons.starttime_utc_col].dt.hour * 60 + season_sched[cons.starttime_utc_col].dt.minute

    # encode the categorical features using ordinal encoding
    encoder = OrdinalEncoder()
    season_sched[cons.venue_timezone_col] = encoder.fit_transform(season_sched[[cons.venue_timezone_col]]).astype(int)
    season_sched[cons.venue_col] = encoder.fit_transform(season_sched[[cons.venue_col]]).astype(int)

    # encode the target variable 'lastPeriod' using a mapping of the period types to integers
    season_sched[cons.last_period_col] = season_sched.loc[
        season_sched[cons.last_period_col].notna(), cons.last_period_col].map(cons.last_period_map).astype(int)

    return season_sched


def assign_game_points(season_results, to_csv=False):

    print('\nAssigning game points...')

    # for any rows where each team's score evaluates to the same int:
    # assign the last period as OT or SO and add 1 to the higher score to reflect the predicted winner
    season_results.loc[
        (season_results[cons.away_team_score_col].astype(int) == season_results[cons.home_team_score_col].astype(int)) & 
        (season_results[cons.away_team_score_col].notna()), cons.last_period_col] = season_results.apply(lambda row: random.choices(
            [cons.last_period_map['OT'], cons.last_period_map['SO']], weights=[100-cons.shootout_rate, cons.shootout_rate], k=1)[0], axis=1)
    season_results.loc[
        (season_results[cons.away_team_score_col].astype(int) == season_results[cons.home_team_score_col].astype(int)) & 
        (season_results[cons.away_team_score_col].notna()), cons.home_team_score_col] = season_results[cons.home_team_score_col].astype(int) + 1

    # change the type of the score columns back to int after 'ties' were absolved above
    season_results[cons.away_team_score_col] = season_results[cons.away_team_score_col].astype(int)
    season_results[cons.home_team_score_col] = season_results[cons.home_team_score_col].astype(int)
    season_results[cons.last_period_col] = season_results[cons.last_period_col].astype(int)

    season_results['homeTeamPoints'] = np.where(
        season_results[cons.home_team_score_col] > season_results[cons.away_team_score_col], 2, np.where(
            season_results[cons.last_period_col] > 0, 1, np.where(
                season_results[cons.home_team_score_col] < season_results[cons.away_team_score_col], 0, 0)))
    season_results['awayTeamPoints'] = np.where(
        season_results[cons.away_team_score_col] > season_results[cons.home_team_score_col], 2, np.where(
            season_results[cons.last_period_col] > 0, 1, np.where(
                season_results[cons.away_team_score_col] < season_results[cons.home_team_score_col], 0, 0)))

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        col_order = [cons.id_col, cons.season_col, cons.starttime_utc_col, cons.home_team_name_col,
                     cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col,
                     cons.last_period_col, 'homeTeamPoints', 'awayTeamPoints']
        season_results[col_order].to_csv(cons.output_folder + cons.season_sched_pred_filename, index=False)

    return season_results


def generate_final_standings(season_results, to_csv=False):

    print('\nGenerating final standings...')

    home_points = season_results.groupby(cons.home_team_name_col)['homeTeamPoints'].sum().reset_index()
    away_points = season_results.groupby(cons.away_team_name_col)['awayTeamPoints'].sum().reset_index()
    points_df = home_away_accumulation(home_points, away_points, 'Points')

    home_wins = season_results.groupby(cons.home_team_name_col).apply(lambda x: (x[cons.home_team_score_col] > x[cons.away_team_score_col]).sum()).reset_index(name='homeTeamWins')
    away_wins = season_results.groupby(cons.away_team_name_col).apply(lambda x: (x[cons.away_team_score_col] > x[cons.home_team_score_col]).sum()).reset_index(name='awayTeamWins')
    wins_df = home_away_accumulation(home_wins, away_wins, 'Wins')

    home_losses = season_results.loc[season_results[cons.last_period_col]==0].groupby(cons.home_team_name_col).apply(lambda x: (x[cons.home_team_score_col] < x[cons.away_team_score_col]).sum()).reset_index(name='homeTeamLosses')
    away_losses = season_results.loc[season_results[cons.last_period_col]==0].groupby(cons.away_team_name_col).apply(lambda x: (x[cons.away_team_score_col] < x[cons.home_team_score_col]).sum()).reset_index(name='awayTeamLosses')
    losses_df = home_away_accumulation(home_losses, away_losses, 'Losses')

    home_otls = season_results.loc[season_results[cons.last_period_col]>0].groupby(cons.home_team_name_col).apply(lambda x: (x[cons.home_team_score_col] < x[cons.away_team_score_col]).sum()).reset_index(name='homeTeamOTLs')
    away_otls = season_results.loc[season_results[cons.last_period_col]>0].groupby(cons.away_team_name_col).apply(lambda x: (x[cons.away_team_score_col] < x[cons.home_team_score_col]).sum()).reset_index(name='awayTeamOTLs')
    otls_df = home_away_accumulation(home_otls, away_otls, 'OTLs')

    final_standings_pw = pd.merge(points_df, wins_df, on='teamName')
    final_standings_pwl = pd.merge(final_standings_pw, losses_df, on='teamName')
    final_standings = pd.merge(final_standings_pwl, otls_df, on='teamName')

    # free memory by deleting the intermediate dataframes that are no longer needed
    del final_standings_pw, final_standings_pwl, points_df, wins_df, losses_df, otls_df

    # load individual team info to merge with final standings for wildcard setup
    team_info_df = ut.team_info()

    final_standings = pd.merge(final_standings, team_info_df[['teamName', 'divisionName', 'conferenceName']], on='teamName')

    # save the updated season schedule with predictions to a new CSV file
    if to_csv:
        final_standings.to_csv(cons.output_folder + 'final_standings.csv', index=False)

    return final_standings

def print_wildcard_results(final_standings):
    # Eastern conference wildcard chart
    atlantic_standings = final_standings.loc[final_standings['divisionName'] == 'Atlantic'].sort_values(by=['totalPoints'], ascending=[False])
    metro_standings = final_standings.loc[final_standings['divisionName'] == 'Metropolitan'].sort_values(by=['totalPoints'], ascending=[False])
    
    print('\n-------- Eastern Conference --------')
    print('------------------------------------')
    print('--- Atlantic Division ---')
    for _, row in atlantic_standings.head(3).iterrows():
        team_printer(row)
    print('--- Metropolitan Division ---')
    for _, row in metro_standings.head(3).iterrows():
        team_printer(row)
    print('--- Wild Card ---')
    east_wildcard_standings = pd.concat([atlantic_standings.drop(atlantic_standings.index[:3]), metro_standings.drop(metro_standings.index[:3])]).sort_values(by=['totalPoints'], ascending=[False])
    for _, row in east_wildcard_standings.head(2).iterrows():
        team_printer(row)
    east_wildcard_standings.drop(east_wildcard_standings.index[:2], inplace=True)
    print('-----------------')
    for _, row in east_wildcard_standings.iterrows():
        team_printer(row)

    # Western conference wildcard chart
    central_standings = final_standings.loc[final_standings['divisionName'] == 'Central'].sort_values(by=['totalPoints'], ascending=[False])
    pacific_standings = final_standings.loc[final_standings['divisionName'] == 'Pacific'].sort_values(by=['totalPoints'], ascending=[False])
    
    print('\n-------- Western Conference --------')
    print('------------------------------------')
    print('--- Central Division ---')
    for _, row in central_standings.head(3).iterrows():
        team_printer(row)
    print('--- Pacific Division ---')
    for _, row in pacific_standings.head(3).iterrows():
        team_printer(row)
    print('--- Wild Card ---')
    west_wildcard_standings = pd.concat([central_standings.drop(central_standings.index[:3]), pacific_standings.drop(pacific_standings.index[:3])]).sort_values(by=['totalPoints'], ascending=[False])
    for _, row in west_wildcard_standings.head(2).iterrows():
        team_printer(row)
    west_wildcard_standings.drop(west_wildcard_standings.index[:2], inplace=True)
    print('-----------------')
    for _, row in west_wildcard_standings.iterrows():
        team_printer(row)


def home_away_accumulation(home_df, away_df, stat_col):
    merged_df = pd.merge(home_df, away_df, left_on=cons.home_team_name_col, right_on=cons.away_team_name_col)
    merged_df[f'total{stat_col}'] = merged_df[f'homeTeam{stat_col}'] + merged_df[f'awayTeam{stat_col}']
    merged_df.drop(columns=[cons.away_team_name_col, f'homeTeam{stat_col}', f'awayTeam{stat_col}'], inplace=True)
    merged_df.rename(columns={cons.home_team_name_col: 'teamName'}, inplace=True)
    merged_df.sort_values(by=f'total{stat_col}', ascending=False, inplace=True)
    return merged_df.reset_index(drop=True)


def team_printer(row):
    if len(row['teamName']) < 16:
        print(colored(f'\t{row['teamName']}\t\t{row['totalPoints']}\t{row['totalWins']}-{row['totalLosses']}-{row['totalOTLs']}', cons.team_colors[row['teamName']]))
    else:
        print(colored(f'\t{row['teamName']}\t{row['totalPoints']}\t{row['totalWins']}-{row['totalLosses']}-{row['totalOTLs']}', cons.team_colors[row['teamName']]))



if __name__ == "__main__":
    # generate the season schedule with predictions and save to a new CSV file
    season_results = season_predictions(to_csv=True)

    # assign points to each team based on the predicted scores and save to a new CSV file
    season_results = assign_game_points(season_results, to_csv=True)

    # generate the final standings based on the predicted points and save to a new CSV file
    final_standings = generate_final_standings(season_results, to_csv=True)

    # print out the wildcard standings based on the predicted points
    print_wildcard_results(final_standings)