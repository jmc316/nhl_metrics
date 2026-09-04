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

    # add implied probability based off odds
    odds_data.loc[odds_data[cons.home_odds_col] > 0, 'homeTeamImpliedProb'] = (100 / (odds_data['homeTeamOdds']+100)) * 100
    odds_data.loc[odds_data[cons.away_odds_col] > 0, 'awayTeamImpliedProb'] = (100 / (odds_data['awayTeamOdds']+100)) * 100
    odds_data.loc[odds_data[cons.home_odds_col] < 0, 'homeTeamImpliedProb'] = (abs(odds_data['homeTeamOdds']) / (abs(odds_data['homeTeamOdds'])+100)) * 100
    odds_data.loc[odds_data[cons.away_odds_col] < 0, 'awayTeamImpliedProb'] = (abs(odds_data['awayTeamOdds']) / (abs(odds_data['awayTeamOdds'])+100)) * 100

    # add a row that is the odds of the winner if the correct prediction was made, else 0
    odds_data['winner_odds'] = odds_data.apply(
        lambda row: row[cons.home_odds_col] if row[cons.cor_outcome_col] == 1 and row[cons.home_team_score_col] > row[cons.away_team_score_col]
            else (row[cons.away_odds_col] if row[cons.cor_outcome_col] == 1 and row[cons.home_team_score_col] < row[cons.away_team_score_col]
                else 0), axis=1)

    # create value column
    odds_data.loc[odds_data[cons.home_odds_col] > 0, 'homeNetOdds'] = odds_data[cons.home_odds_col] / 100
    odds_data.loc[odds_data[cons.home_odds_col] < 0, 'homeNetOdds'] = 100 / abs(odds_data[cons.home_odds_col])
    odds_data.loc[odds_data[cons.away_odds_col] > 0, 'awayNetOdds'] = odds_data[cons.away_odds_col] / 100
    odds_data.loc[odds_data[cons.away_odds_col] < 0, 'awayNetOdds'] = 100 / abs(odds_data[cons.away_odds_col])
    odds_data.loc[odds_data[cons.home_odds_col] > odds_data[cons.away_odds_col], 'ExpectedValue'] = odds_data[cons.home_win_prob_col] * odds_data['homeNetOdds'] - odds_data[cons.away_win_prob_col]
    odds_data.loc[odds_data[cons.away_odds_col] > odds_data[cons.home_odds_col], 'ExpectedValue'] = odds_data[cons.away_win_prob_col] * odds_data['awayNetOdds'] - odds_data[cons.home_win_prob_col]

    odds_data.loc[(odds_data[cons.home_odds_col] > odds_data[cons.away_odds_col]) &
                  (odds_data['ExpectedValue'] > 0), 'kellyPer_full'] = (odds_data['homeNetOdds']*odds_data[cons.home_win_prob_col] - odds_data[cons.away_win_prob_col]) / odds_data['homeNetOdds']
    odds_data.loc[(odds_data[cons.home_odds_col] < odds_data[cons.away_odds_col]) &
                  (odds_data['ExpectedValue'] > 0), 'kellyPer_full'] = (odds_data['awayNetOdds']*odds_data[cons.away_win_prob_col] - odds_data[cons.home_win_prob_col]) / odds_data['awayNetOdds']
    odds_data['kellyPer_qtr'] = odds_data['kellyPer_full'] / 4
    odds_data['kellyPer_hlf'] = odds_data['kellyPer_full'] / 2

    # implement betting strategy here: track a compounding bankroll, wagering kellyStake_full% of the
    # bankroll on each game, where a day's bankroll reflects the prior day's wins/losses
    odds_data[cons.game_date_col] = odds_data[cons.starttime_est_col].dt.date

    # multiplier applied to a game's stake: odds payout if won, -1 if lost, 0 if no bet was placed
    odds_data['stakeMultiplier'] = -1.0
    odds_data.loc[odds_data[cons.winner_odds_col] > 0, 'stakeMultiplier'] = odds_data[cons.winner_odds_col] / 100
    odds_data.loc[odds_data[cons.winner_odds_col] < 0, 'stakeMultiplier'] = 100 / abs(odds_data[cons.winner_odds_col])
    odds_data.loc[odds_data['kellyPer_full'].isna(), 'stakeMultiplier'] = 0.0

    bankroll_init = 100.0
    bankroll_full = bankroll_init
    bankroll_hlf = bankroll_init
    bankroll_qtr = bankroll_init
    bankroll_full_by_date = {}
    bankroll_hlf_by_date = {}
    bankroll_qtr_by_date = {}
    for game_date in sorted(odds_data[cons.game_date_col].unique()):
        bankroll_full_by_date[game_date] = bankroll_full
        bankroll_hlf_by_date[game_date] = bankroll_hlf
        bankroll_qtr_by_date[game_date] = bankroll_qtr
        day_mask = odds_data[cons.game_date_col] == game_date
        day_full_stakes = 100 * bankroll_full * odds_data.loc[day_mask, 'kellyPer_full'].fillna(0) / 100
        bankroll_full += (day_full_stakes * odds_data.loc[day_mask, 'stakeMultiplier']).sum()
        bankroll_hlf_stakes = 100 * bankroll_hlf * odds_data.loc[day_mask, 'kellyPer_hlf'].fillna(0) / 100
        bankroll_hlf += (bankroll_hlf_stakes * odds_data.loc[day_mask, 'stakeMultiplier']).sum()
        bankroll_qtr_stakes = 100 * bankroll_qtr * odds_data.loc[day_mask, 'kellyPer_qtr'].fillna(0) / 100
        bankroll_qtr += (bankroll_qtr_stakes * odds_data.loc[day_mask, 'stakeMultiplier']).sum()

    odds_data[f'{cons.bankroll_col}_full'] = odds_data[cons.game_date_col].map(bankroll_full_by_date)
    odds_data[f'{cons.bankroll_col}_hlf'] = odds_data[cons.game_date_col].map(bankroll_hlf_by_date)
    odds_data[f'{cons.bankroll_col}_qtr'] = odds_data[cons.game_date_col].map(bankroll_qtr_by_date)

    odds_data[f'{cons.winnings_col}_full'] = odds_data['kellyPer_full'] * odds_data['bankroll_full'] * odds_data['stakeMultiplier']
    odds_data[f'{cons.winnings_col}_hlf'] = odds_data['kellyPer_hlf'] * odds_data['bankroll_hlf'] * odds_data['stakeMultiplier']
    odds_data[f'{cons.winnings_col}_qtr'] = odds_data['kellyPer_qtr'] * odds_data['bankroll_qtr'] * odds_data['stakeMultiplier']

    odds_data.drop(columns=['stakeMultiplier', 'homeTeamImpliedProb', 'awayTeamImpliedProb', 'homeNetOdds', 'awayNetOdds'], inplace=True)

    # save the prediction return data to the prediction folder
    csvSave(odds_data, cons.season_pred_folder.format(date=today_dt), cons.pred_ret_filename.format(date_since=date_since, today_dt=today_dt))

    # the date that playoffs starts for this season
    playoff_st_dt = season_actual_df.loc[season_actual_df[cons.game_type_col]==3, cons.starttime_est_col].dt.date.min()

    full_kelly_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), f'{cons.winnings_col}_full'].sum()
    half_kelly_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), f'{cons.winnings_col}_hlf'].sum()
    qtr_kelly_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), f'{cons.winnings_col}_qtr'].sum()

    if playoff_st_dt is not pd.NaT:
        full_kelly_playoff_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= playoff_st_dt, f'{cons.winnings_col}_full'].sum()
        half_kelly_playoff_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= playoff_st_dt, f'{cons.winnings_col}_hlf'].sum()
        qtr_kelly_playoff_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= playoff_st_dt, f'{cons.winnings_col}_qtr'].sum()

    # print a bunch of stats to the terminal for two date ranges: total season and playoffs
    total_daily_return = odds_data.loc[odds_data[cons.starttime_est_col].dt.date == (pd.to_datetime(today_dt) - datetime.timedelta(days=1)).date(), f'{cons.winnings_col}_qtr'].sum()
    if odds_data.loc[odds_data[cons.starttime_est_col].dt.date == (pd.to_datetime(today_dt) - datetime.timedelta(days=1)).date()].empty:
        print('No games yesterday.')
    else:
        print(f'Yesterday\'s games and returns ({total_daily_return:.2f}):')
        for _, row in odds_data.loc[odds_data[cons.starttime_est_col].dt.date == (pd.to_datetime(today_dt) - datetime.timedelta(days=1)).date()].iterrows():
            ot_str = ' (OT)' if row[cons.last_period_col]=='OT' else ''
            if row[cons.home_team_score_col] > row[cons.away_team_score_col]:
                print(f"\t{cons.team_name_addrev_map[row[cons.away_team_name_col]].lower()} {int(row[cons.away_team_score_col])} at {cons.team_name_addrev_map[row[cons.home_team_name_col]]} {int(row[cons.home_team_score_col])}{ot_str}: {row[f'{cons.winnings_col}_qtr']:.2f}")
            else:
                print(f"\t{cons.team_name_addrev_map[row[cons.away_team_name_col]]} {int(row[cons.away_team_score_col])} at {cons.team_name_addrev_map[row[cons.home_team_name_col]].lower()} {int(row[cons.home_team_score_col])}{ot_str}: {row[f'{cons.winnings_col}_qtr']:.2f}")

    num_correct_preds = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date(), cons.cor_outcome_col].sum()
    num_games = len(odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= pd.to_datetime(date_since).date()])
    print(f'\nSince {date_since}:')
    print(f'\tModel Win Accuracy: {num_correct_preds/num_games:.2%} ({num_correct_preds}/{num_games})')
    print(f'\tFull Kelly return: {full_kelly_return:.2f}')
    print(f'\tHalf Kelly return: {half_kelly_return:.2f}')
    print(f'\tQuarter Kelly return: {qtr_kelly_return:.2f}\n')

    if pd.to_datetime(today_dt).date() > playoff_st_dt:
        num_correct_preds = odds_data.loc[odds_data[cons.starttime_est_col].dt.date >= playoff_st_dt, cons.cor_outcome_col].sum()
        num_games = len(odds_data.loc[odds_data[cons.starttime_est_col].dt.date>=playoff_st_dt])
        print(f'Since playoffs start ({playoff_st_dt}):')
        print(f'\tModel Win Accuracy: {num_correct_preds/num_games:.2%} ({num_correct_preds}/{num_games})')
        print(f'\tFull Kelly return: {full_kelly_playoff_return:.2f}')
        print(f'\tHalf Kelly return: {half_kelly_playoff_return:.2f}')
        print(f'\tQuarter Kelly return: {qtr_kelly_playoff_return:.2f}\n')

    _, ax = plt.subplots()

    # create a column that is the sum of each day's winnings
    odds_data_daily_full = odds_data.groupby(cons.game_date_col)[[f'{cons.winnings_col}_full', f'{cons.winnings_col}_hlf', f'{cons.winnings_col}_qtr']].sum().reset_index()
    odds_data_daily_full['full_daily_return'] = odds_data_daily_full[f'{cons.winnings_col}_full'].cumsum()
    odds_data_daily_full['hlf_daily_return'] = odds_data_daily_full[f'{cons.winnings_col}_hlf'].cumsum()
    odds_data_daily_full['qtr_daily_return'] = odds_data_daily_full[f'{cons.winnings_col}_qtr'].cumsum()

    # create y=0 line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # create x=playoff_st_dt line (playoffs start)
    if pd.to_datetime(today_dt).date() > playoff_st_dt:
        ax.axvline(playoff_st_dt, color='green', linewidth=0.8, linestyle='--', label='Playoffs Start')

    # Define condition: green if perfect, red if 0% accuracy, blue otherwise
    colors = []
    for date in odds_data_daily_full[cons.game_date_col]:
        if odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), cons.cor_outcome_col].mean() == 1.0:
            colors.append('green')
        elif odds_data.loc[pd.to_datetime(odds_data[cons.game_date_col]) == dt.strftime(date, cons.date_format_yyyy_mm_dd), cons.cor_outcome_col].mean() == 0.0:
            colors.append('red')
        else:
            colors.append('lightblue')

    # Create bar chart
    daily_revenue = list(odds_data_daily_full[f'{cons.winnings_col}_full'] + odds_data_daily_full[f'{cons.winnings_col}_hlf'] + odds_data_daily_full[f'{cons.winnings_col}_qtr'])
    ax.bar(list(odds_data_daily_full[cons.game_date_col]), daily_revenue, color=colors, label='Revenue')

    # Create line chart on the same axis
    ax.plot(list(odds_data_daily_full[cons.game_date_col]), list(odds_data_daily_full['full_daily_return']), color='red', label='Full Kelly')
    ax.plot(list(odds_data_daily_full[cons.game_date_col]), list(odds_data_daily_full['hlf_daily_return']), color='blue', label='Half Kelly')
    ax.plot(list(odds_data_daily_full[cons.game_date_col]), list(odds_data_daily_full['qtr_daily_return']), color='purple', label='Quarter Kelly')

    # generate a plot chart for returns over the course of the time frame
    ax.set_title(f'Predictions Returns for {date_since} to {today_dt}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(cons.season_pred_folder.format(date=today_dt) + cons.pred_ret_filename.format(date_since=date_since, today_dt=today_dt).replace('.csv', '.png'), bbox_inches='tight')
    # plt.show()
    plt.close()