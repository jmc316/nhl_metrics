import numpy as np
import pandas as pd
import constants as cons

from file_utils import csvLoad
from datetime import datetime as dt

def game_result_comparison(predict_df, actual_df=None):

    print('Comparing predicted game results to actual game results...')

    # filter the predict dataframe to only include predictions
    predict_df = predict_df.loc[predict_df[cons.game_date_col] > cons.last_actual_game_date]
    predict_min_dt = predict_df[cons.game_date_col].min()
    predict_max_dt = predict_df[cons.game_date_col].max()

    # get the monday before the min prediction date
    predict_min_dt_mon = predict_min_dt - pd.to_timedelta(predict_min_dt.weekday(), unit='D')

    # get the monday after the max prediction date
    predict_max_dt_mon = predict_max_dt + pd.to_timedelta(7 - predict_max_dt.weekday(), unit='D')

    if actual_df is None:

        # print('Actual results dataframe does not exist, creating it...')

        current_dt = predict_min_dt_mon
        actual_df = pd.DataFrame()
        # loop through every monday and append the nhl schedule and results
        while current_dt <= predict_max_dt_mon:
            # print(f'\tGetting actual game results for week of {current_dt.strftime(cons.date_format_yyyy_mm_dd)}...')

            for dow in range(0, 7):
                while True:
                    try:
                        # fetch the schedule data for this week and day of week from the NHL API
                        weekly_sched_raw = pd.DataFrame(cons.nhl_client.schedule.weekly_schedule(date=current_dt.strftime(cons.date_format_yyyy_mm_dd))['gameWeek'][dow]['games'])

                    except Exception as ex:
                        # re-try this week's schedule if there was a timeout error
                        print(f'\t\t... {ex} ...')
                        continue
                    # if there are no games in this week, skip to the next week
                    if weekly_sched_raw.empty:
                        break

                    # initialize columns that are the same as the raw data
                    weekly_sched = weekly_sched_raw[['id', cons.season_col, cons.game_type_col, cons.starttime_utc_col, cons.venue_timezone_col]]

                    weekly_sched.rename(columns={'id': cons.game_id_col, 'season': cons.season_name_col}, inplace=True)

                    # only include NHL games (gameType 2 = regular season, 3 = playoffs)
                    weekly_sched = weekly_sched.loc[weekly_sched[cons.game_type_col].isin([2, 3])]
                    weekly_sched_raw = weekly_sched_raw.loc[weekly_sched_raw[cons.game_type_col].isin([2, 3])]

                    # if there are no valid NHL games in this week, skip to the next week
                    if weekly_sched.empty:
                        break

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

                    actual_df = pd.concat([actual_df, weekly_sched], ignore_index=True)

                    break
                    
            current_dt += pd.to_timedelta(7, unit='D')

        # add game date column
        actual_df[cons.game_date_col] = pd.to_datetime(actual_df[cons.starttime_utc_col]).dt.date

        actual_df = actual_df.loc[(actual_df[cons.game_date_col] > predict_min_dt) &
                                (actual_df[cons.game_date_col] <= predict_max_dt) &
                                (actual_df[cons.away_team_score_col].notna()) &
                                (actual_df[cons.home_team_score_col].notna())]
        
    if actual_df.empty:
        print('No actual game results found for the prediction date range.\n')
        return None
        
    merge_cols = [cons.game_id_col, cons.game_date_col, cons.home_team_name_col, cons.away_team_name_col, cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col]

    # merge the predict and actual dataframes on the game ID column
    comparison_df = pd.merge(predict_df[merge_cols], actual_df[merge_cols], on=[cons.game_id_col, cons.game_date_col, cons.home_team_name_col, cons.away_team_name_col], suffixes=('_predicted', '_actual'))

    comparison_df['correct_outcome'] = np.where(
        (comparison_df[cons.home_team_score_col+'_actual'] > comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] > comparison_df[cons.away_team_score_col+'_predicted']) |
        (comparison_df[cons.home_team_score_col+'_actual'] < comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] < comparison_df[cons.away_team_score_col+'_predicted']) |
        (comparison_df[cons.home_team_score_col+'_actual'] == comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] == comparison_df[cons.away_team_score_col+'_predicted']),
        1, 0
        )
    
    print(f'Games with correct outcome prediction: {sum(comparison_df["correct_outcome"])} / {len(comparison_df)} ({sum(comparison_df["correct_outcome"]) / len(comparison_df):.2%})\n')

    # give the percent accuracy of the correct outcomes per day and the number of correct outcomes
    daily_accuracy = comparison_df.groupby(cons.game_date_col)['correct_outcome'].mean()
    daily_correct = comparison_df.groupby(cons.game_date_col)['correct_outcome'].sum()
    daily_games = comparison_df.groupby(cons.game_date_col)['correct_outcome'].count()
    for date, accuracy in daily_accuracy.items():
        print(f' {date}: ({daily_correct[date]}/{daily_games[date]}) {accuracy:.2%}')


    return comparison_df


if __name__ == '__main__':

    cons.last_actual_game_date = pd.to_datetime('2026-02-05').date()
    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    season_prediction_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))
    
    game_result_comparison(season_prediction_df)