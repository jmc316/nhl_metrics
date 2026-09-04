import os

import numpy as np
import pandas as pd
import constants as cons

from datetime import datetime as dt
from utils.file_utils import csvLoad
from features.features import clean_feature_df


def prediction_analysis(actuals_df, date_since, date_until):

    predict_df = pd.DataFrame()

    # loop through every folder in the season prediction folder and create a dataframe with all predictions for the most recent game date
    for pred_date in os.listdir(cons.season_pred_base_folder):
        if (pred_date < date_since) | (pred_date >= date_until):
            continue
        if os.path.exists(cons.season_pred_base_folder + pred_date + '/' + cons.season_pred_filename.format(date=pred_date)):
            predict_df_indiv = csvLoad(cons.season_pred_base_folder + pred_date + '/', cons.season_pred_filename.format(date=pred_date))
        else:
            predict_df_indiv = csvLoad(cons.season_pred_base_folder + pred_date + '/', cons.playoff_pred_filename.format(date=pred_date))
        # print(f'Analyzing predictions for {pred_date}...')
        min_predict_date = pd.to_datetime(predict_df_indiv.loc[pd.to_datetime(predict_df_indiv[cons.starttime_est_col]).dt.date == dt.strptime(pred_date, '%Y-%m-%d').date(), cons.starttime_est_col]).dt.date.min()
        predict_df = pd.concat([predict_df, predict_df_indiv.loc[pd.to_datetime(predict_df_indiv[cons.starttime_est_col]).dt.date == min_predict_date]], ignore_index=True)

        if predict_df_indiv.loc[pd.to_datetime(predict_df_indiv[cons.starttime_est_col]).dt.date == min_predict_date].empty:
            # print(f'\t... No games found')
            pass

    # not a valid date range to do comparison for
    if date_since > date_until:
        return pd.DataFrame()

    if predict_df.empty:
        print(f'No predictions found for the date range {date_since} to {date_until}.')
        return pd.DataFrame()
    else:
        predict_df = clean_feature_df(predict_df)

    home_prob_col = cons.home_win_prob_col
    away_prob_col = cons.away_win_prob_col

    comparison_df = pd.merge(predict_df, actuals_df, on=[cons.game_id_col, cons.starttime_est_col], suffixes=(cons.pred_suf, cons.act_suf))

    comparison_df = comparison_df[[cons.game_id_col, cons.starttime_est_col, cons.home_team_name_col+cons.pred_suf,
                                   cons.away_team_name_col+cons.pred_suf, home_prob_col, away_prob_col,
                                   cons.home_team_score_col+cons.act_suf, cons.away_team_score_col+cons.act_suf,
                                   cons.last_period_col+cons.act_suf]]
    
    comparison_df.rename(columns={cons.game_date_col+cons.pred_suf: cons.game_date_col, cons.home_team_name_col+cons.pred_suf: cons.home_team_name_col,
                                  cons.away_team_name_col+cons.pred_suf: cons.away_team_name_col, cons.home_team_score_col+cons.act_suf: cons.home_team_score_col,
                                  cons.away_team_score_col+cons.act_suf: cons.away_team_score_col, cons.last_period_col+cons.act_suf: cons.last_period_col}, inplace=True)

    comparison_df[cons.cor_outcome_col] = np.where(
        ((comparison_df[cons.home_team_score_col] > comparison_df[cons.away_team_score_col]) &
        (comparison_df[home_prob_col] > comparison_df[away_prob_col])) |
        ((comparison_df[cons.home_team_score_col] < comparison_df[cons.away_team_score_col]) &
        (comparison_df[home_prob_col] < comparison_df[away_prob_col])) |
        ((comparison_df[cons.home_team_score_col] == comparison_df[cons.away_team_score_col]) &
        (comparison_df[home_prob_col] == comparison_df[away_prob_col])),
        1, 0
        )
    
    # print(f"\nGames with correct outcome prediction: {sum(comparison_df[cons.cor_outcome_col])} / {len(comparison_df)} ({sum(comparison_df[cons.cor_outcome_col]) / len(comparison_df):.2%})\n")

    return comparison_df


if __name__ == '__main__':

    # cons.last_actual_game_date = pd.to_datetime('2026-04-13').date()
    today_dt = dt.now().date().strftime(cons.date_format_yyyy_mm_dd)

    season_prediction_df = csvLoad(cons.season_pred_folder.format(date=today_dt), cons.season_pred_filename.format(date=today_dt))

    prediction_analysis(season_prediction_df, '2026-02-24', today_dt) # last day before Olympic Break ended