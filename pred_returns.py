import datetime
import matplotlib

import pandas as pd
import constants as cons
import matplotlib.pyplot as plt

from datetime import datetime as dt
from schedule import clean_schedule_df
from analyze import prediction_analysis
from utils.file_utils import csvLoad, csvSave

# avoid garbage collection
matplotlib.use('Agg')


def daily_probability(today_dt, date_since, season, display_graphic=True):

    # load the schedule data with game results
    season_actual_df = csvLoad(cons.season_sched_folder, cons.season_sched_filename.format(season=season))
    season_actual_df = clean_schedule_df(season_actual_df)
    season_actual_df[cons.starttime_est_col] = season_actual_df[cons.starttime_utc_col].dt.tz_convert(cons.est_tz).dt.tz_localize(None)

    # load a dataframe with all predictions in this date range
    pred_df = prediction_analysis(season_actual_df, date_since, today_dt)

    if pred_df.empty:
        print('Prediction analysis not available - invalid date range\n')
        return

    # load dataframe containing all odds data
    odds_data = pd.read_csv(cons.util_data_folder + cons.sched_odds_filename)

    # merge the predictions into the odds data
    merge_cols = [cons.game_id_col, cons.home_team_name_col, cons.away_team_name_col]
    odds_data = pd.merge(pred_df, odds_data[merge_cols+[cons.home_odds_col, cons.away_odds_col]], on=merge_cols, how='left')

    if odds_data[cons.home_odds_col].isnull().any() or odds_data[cons.away_odds_col].isnull().any():
        missing_games = odds_data[cons.home_odds_col].isnull().sum() + odds_data[cons.away_odds_col].isnull().sum()
        print(f"*** WARNING: Missing odds data for {missing_games} entries! ***")

    # add a row that is the odds of the winner if the correct prediction was made, else 0
    odds_data['winner_odds'] = odds_data.apply(
        lambda row: row[cons.home_odds_col] if row[cons.cor_outcome_col] == 1 and row[cons.home_team_score_col] > row[cons.away_team_score_col]
            else (row[cons.away_odds_col] if row[cons.cor_outcome_col] == 1 and row[cons.home_team_score_col] < row[cons.away_team_score_col]
                else 0), axis=1)

    # initialize the winnings column as losing every prediction
    odds_data[cons.winnings_col] = -1.0

    # add winning values to the winnings column for when the winner odds are not 0
    odds_data.loc[odds_data[cons.winner_odds_col] > 0, cons.winnings_col] = odds_data[cons.winner_odds_col] / 100
    odds_data.loc[odds_data[cons.winner_odds_col] < 0, cons.winnings_col] = 100 / abs(odds_data[cons.winner_odds_col])

    # save the prediction return data to the prediction folder
    csvSave(odds_data, cons.season_pred_folder.format(date=today_dt), cons.pred_ret_filename.format(date_since=date_since, today_dt=today_dt))

    # the date that playoffs starts for this season
    playoff_st_dt = season_actual_df.loc[season_actual_df[cons.game_type_col]==3, cons.starttime_est_col].dt.date.min()

    total_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), cons.winnings_col].sum()
    if playoff_st_dt is not pd.NaT:
        playoff_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= playoff_st_dt, cons.winnings_col].sum()

    # print a bunch of stats to the terminal for two date ranges: total season and playoffs
    print('Yesterday\'s games and returns:')
    for _, row in odds_data.loc[odds_data[cons.starttime_est_col].dt.date == (pd.to_datetime(today_dt) - datetime.timedelta(days=1)).date()].iterrows():
        ot_str = ' (OT)' if row[cons.last_period_col]=='OT' else ''
        if row[cons.home_team_score_col] > row[cons.away_team_score_col]:
            print(f"\t{cons.team_name_addrev_map[row[cons.away_team_name_col]].lower()} {int(row[cons.away_team_score_col])} at {cons.team_name_addrev_map[row[cons.home_team_name_col]]} {int(row[cons.home_team_score_col])}{ot_str}: {row[cons.winnings_col]:.2f}")
        else:
            print(f"\t{cons.team_name_addrev_map[row[cons.away_team_name_col]]} {int(row[cons.away_team_score_col])} at {cons.team_name_addrev_map[row[cons.home_team_name_col]].lower()} {int(row[cons.home_team_score_col])}{ot_str}: {row[cons.winnings_col]:.2f}")

    num_correct_preds = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), cons.cor_outcome_col].sum()
    num_games = len(odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date()])
    print(f'\nTotal return on $1 bet since {date_since}: ${total_return:.2f} ({num_correct_preds/num_games:.2%} win rate)\n')

    if pd.to_datetime(today_dt).date() > playoff_st_dt:
        num_games = len(odds_data.loc[odds_data[cons.starttime_est_col].dt.date>=playoff_st_dt])
        print(f'Total return on $1 bet since playoffs start ({num_games} games): ${playoff_return:.2f}\n')

    _, ax = plt.subplots()

    # create a column that is the sum of each day's winnings
    odds_data[cons.game_date_col] = odds_data[cons.starttime_est_col].dt.date
    odds_data_daily = odds_data.groupby(cons.game_date_col)[cons.winnings_col].sum().reset_index()
    odds_data_daily['daily_return'] = odds_data_daily[cons.winnings_col].cumsum()

    # create y=0 line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # create x=playoff_st_dt line (playoffs start)
    if pd.to_datetime(today_dt).date() > playoff_st_dt:
        ax.axvline(playoff_st_dt, color='green', linewidth=0.8, linestyle='--', label='Playoffs Start')

    # Define condition: green if perfect, red if 0% accuracy, blue otherwise
    colors = []
    for date in odds_data_daily[cons.game_date_col]:
        if odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), cons.cor_outcome_col].mean() == 1.0:
            colors.append('green')
        elif odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), cons.cor_outcome_col].mean() == 0.0:
            colors.append('red')
        else:
            colors.append('lightblue')

    # Create bar chart
    ax.bar(list(odds_data_daily[cons.game_date_col]), list(odds_data_daily[cons.winnings_col]), color=colors, label='Revenue')

    # Create line chart on the same axis
    ax.plot(list(odds_data_daily[cons.game_date_col]), list(odds_data_daily['daily_return']), color='red', label='Cumulative Winnings')

    # generate a plot chart for returns over the course of the time frame
    ax.set_title(f'Predictions Returns for {date_since} to {today_dt}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(cons.season_pred_folder.format(date=today_dt) + cons.pred_ret_filename.format(date_since=date_since, today_dt=today_dt).replace('.csv', '.png'), bbox_inches='tight')
    # plt.show()
    plt.close()