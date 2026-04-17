import os
import numpy as np
import pandas as pd
import constants as cons
import nhl_client as nhlc

from file_utils import csvLoad
from datetime import datetime as dt

def game_result_comparison(predict_df, actual_df=None):

    print('Comparing predicted game results to actual game results...')

    # filter the predict dataframe to only include predictions
    predict_df = predict_df.loc[predict_df[cons.game_date_col] > cons.last_actual_game_date]

    if predict_df.loc[predict_df[cons.last_period_col].isna()].empty:
        print('No predictions found for games that have not yet been played.\n')
        return None

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

            weekly_sched = nhlc.get_sched_data(current_dt, 0)

            actual_df = pd.concat([actual_df, weekly_sched], ignore_index=True)
                    
            current_dt += pd.to_timedelta(7, unit='D')

        # add game date column
        actual_df[cons.game_date_col] = pd.to_datetime(actual_df[cons.starttime_est_col]).dt.date

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

    # create a column that indicates whether the predicted outcome was correct (1) or not (0)
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


def prediction_analysis(actuals_df, date_since, date_until):

    predict_df = pd.DataFrame()

    # loop through every folder in the season prediction folder and create a dataframe with all predictions for the most recent game date
    for folder in os.listdir('output/season_predictions/'):
        if (folder < date_since) | (folder >= date_until):
            continue
        predict_df_indiv = pd.read_csv('output/season_predictions/' + folder + '/regularseason_predictions_' + folder + '.csv')
        print(f'Analyzing predictions for {folder}...')
        min_predict_date = predict_df_indiv.loc[predict_df_indiv[cons.game_date_col] == folder, cons.game_date_col].min()
        predict_df = pd.concat([predict_df, predict_df_indiv.loc[predict_df_indiv[cons.game_date_col] == min_predict_date]], ignore_index=True)

        if predict_df_indiv.loc[predict_df_indiv[cons.game_date_col] == min_predict_date].empty:
            print(f'\t... No games found')

    comparison_df = pd.merge(predict_df, actuals_df, on=[cons.game_id_col], suffixes=('_predicted', '_actual'))

    comparison_df = comparison_df[[cons.game_id_col, cons.game_date_col+'_predicted', cons.home_team_name_col+'_predicted', cons.away_team_name_col+'_predicted', cons.home_team_score_col+'_predicted', cons.away_team_score_col+'_predicted', cons.last_period_col+'_predicted',
                            cons.home_team_score_col+'_actual', cons.away_team_score_col+'_actual', cons.last_period_col+'_actual']]
    
    comparison_df.rename(columns={cons.game_date_col+'_predicted': cons.game_date_col, cons.home_team_name_col+'_predicted': cons.home_team_name_col, cons.away_team_name_col+'_predicted': cons.away_team_name_col}, inplace=True)

    comparison_df['correct_outcome'] = np.where(
        ((comparison_df[cons.home_team_score_col+'_actual'] > comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] > comparison_df[cons.away_team_score_col+'_predicted'])) |
        ((comparison_df[cons.home_team_score_col+'_actual'] < comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] < comparison_df[cons.away_team_score_col+'_predicted'])) |
        ((comparison_df[cons.home_team_score_col+'_actual'] == comparison_df[cons.away_team_score_col+'_actual']) &
        (comparison_df[cons.home_team_score_col+'_predicted'] == comparison_df[cons.away_team_score_col+'_predicted'])),
        1, 0
        )
    
    print(f"\nGames with correct outcome prediction: {sum(comparison_df['correct_outcome'])} / {len(comparison_df)} ({sum(comparison_df['correct_outcome']) / len(comparison_df):.2%})\n")

    return comparison_df


if __name__ == '__main__':

    # cons.last_actual_game_date = pd.to_datetime('2026-04-13').date()
    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    season_prediction_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))

    prediction_analysis(season_prediction_df, '2026-02-24', today_dt) # last day before Olympic Break ended