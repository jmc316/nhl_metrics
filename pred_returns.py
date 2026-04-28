import pandas as pd
import constants as cons
import matplotlib.pyplot as plt

from file_utils import csvLoad, csvSave
from datetime import datetime as dt
from analyze import prediction_analysis


def daily_probability(today_dt, date_since, display_graphic=True):

    season_actual_df = csvLoad(cons.season_sched_folder, cons.season_sched_filename.format(season='20252026'))

    pred_df = prediction_analysis(season_actual_df, date_since, today_dt)

    odds_data = pd.read_csv(cons.util_data_folder + 'all_time_schedule_odds.csv')

    merge_cols = [cons.game_id_col, cons.home_team_name_col, cons.away_team_name_col]
    odds_data = pd.merge(pred_df, odds_data[merge_cols+['homeTeamOdds', 'awayTeamOdds']], on=merge_cols, how='left')

    if odds_data['homeTeamOdds'].isnull().any() or odds_data['awayTeamOdds'].isnull().any():
        missing_games = odds_data['homeTeamOdds'].isnull().sum() + odds_data['awayTeamOdds'].isnull().sum()
        print(f"Warning: Missing odds data for {missing_games} entries!")

    odds_data['winner_odds'] = odds_data['winner_odds'] = odds_data.apply(lambda row: row['homeTeamOdds'] if row['correct_outcome'] == 1 and row['homeTeamScore_predicted'] > row['awayTeamScore_predicted'] else (row['awayTeamOdds'] if row['correct_outcome'] == 1 and row['homeTeamScore_predicted'] < row['awayTeamScore_predicted'] else 0), axis=1)

    odds_data['winnings'] = -1.0

    odds_data.loc[odds_data['winner_odds'] > 0, 'winnings'] = odds_data['winner_odds'] / 100
    odds_data.loc[odds_data['winner_odds'] < 0, 'winnings'] = 100 / abs(odds_data['winner_odds'])

    csvSave(odds_data, f'output/season_predictions/{today_dt}/', f'prediction_returns_{date_since}_to_{today_dt}.csv')

    total_return = odds_data.loc[pd.to_datetime(odds_data['gameDate']) > date_since, 'winnings'].sum()

    if total_return > 0:
        print(f'Total return on $1 bet since {date_since}: ${total_return:.2f} (Profit of ${total_return - 1:.2f})\n')
    else:
        print(f'Total return on $1 bet since {date_since}: ${total_return:.2f} (Loss of ${1 - total_return:.2f})\n')

    _, ax = plt.subplots()

    # create a column that is the sum of each day's winnings
    odds_data_daily = odds_data.groupby(cons.game_date_col)['winnings'].sum().reset_index()
    odds_data_daily['daily_return'] = odds_data_daily['winnings'].cumsum()

    # create y=0 line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # create x='2026-04-17' line (playoffs start)
    ax.axvline(pd.to_datetime('2026-04-17').date(), color='green', linewidth=0.8, linestyle='--', label='Playoffs Start')

    # Define condition: green if perfect, red if 0% accuracy, blue otherwise
    colors = []
    for date in odds_data_daily[cons.game_date_col]:
        if odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), 'correct_outcome'].mean() == 1.0:
            colors.append('green')
        elif odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), 'correct_outcome'].mean() == 0.0:
            colors.append('red')
        else:
            colors.append('lightblue')

    # Create bar chart
    ax.bar(list(odds_data_daily[cons.game_date_col]), list(odds_data_daily['winnings']), color=colors, label='Revenue')

    # Create line chart on the same axis
    ax.plot(list(odds_data_daily[cons.game_date_col]), list(odds_data_daily['daily_return']), color='red', label='Cumulative Winnings')

    ax.set_title(f'Predictions Returns for {date_since} to {today_dt}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'output/season_predictions/{today_dt}/prediction_returns_{date_since}_to_{today_dt}.png')
    if display_graphic:
        plt.show()