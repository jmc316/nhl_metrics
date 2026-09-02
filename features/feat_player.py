import json

import numpy as np
import pandas as pd
import constants as cons

from utils.file_utils import csvLoad, csvSave
from schedule import load_sched_df_features


def player_features_update(data_df_in=pd.DataFrame, verbose=False):

    if data_df_in.empty:
        data_df = load_sched_df_features(feat_set_label='player')
    else:
        data_df = data_df_in[cons.player_feature_cols]

    # add player-based features
    player_features = ['lineup_strength']

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    player_df = load_player_df()

    print(f'\tComputing Player values...')
    player_df = compute_player_value(player_df)

    for feature in player_features:

        print(f'\tAdding {feature}...')

        if feature == 'lineup_strength':
            data_df = lineup_strength(data_df, player_df)

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            print(f'Writing player features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.player_features_folder, cons.player_features_filename.format(season=season))

    return data_df, player_features


def load_player_df():

    # load the saved goalie features dataframe
    player_df = csvLoad(cons.player_features_folder, cons.player_data_filename)
    player_df[cons.season_name_col] = player_df[cons.season_name_col].astype('string')
    
    return player_df


def compute_player_value(player_df):

    # merge the data with different game types for each playerId/seasonName/teamName
    player_df = player_df.groupby([cons.season_name_col, 'playerId'], as_index=False).agg({
        'gamesPlayed': 'sum',
        'assists': 'sum',
        'totToi': 'sum',
        'gameWinningGoals': 'sum',
        'goals': 'sum',
        'otGoals': 'sum',
        'pim': 'sum',
        'plusMinus': 'sum',
        'points': 'sum',
        'powerPlayGoals': 'sum',
        'powerPlayPoints': 'sum',
        'shorthandedGoals': 'sum',
        'shorthandedPoints': 'sum',
        'shots': 'sum',
        'gamesStarted': 'sum',
        'goalsAgainst': 'sum',
        'losses': 'sum',
        'otLosses': 'sum',
        'shotsAgainst': 'sum',
        'shutouts': 'sum',
        'wins': 'sum',
        'position': 'first'
    })

    # re-compute average features, except for faceoffWinningPctg
    player_df.loc[player_df['position'] != 'goalie', 'shootingPctg'] = player_df['goals'] / player_df['shots']
    player_df.loc[player_df['position'] == 'goalie', 'goalsAgainstAvg'] = player_df['goalsAgainst'] / (player_df['totToi'] / 60)
    player_df.loc[player_df['position'] == 'goalie', 'savePctg'] = 1 - (player_df['goalsAgainst'] / player_df['shotsAgainst'])

    # compute skater value based on features
    player_df.loc[player_df['position'] != 'goalie', 'value'] = (
        player_df['goals'] * 1.0 +
        player_df['assists'] * 0.7 +
        player_df['powerPlayPoints'] * 0.3 +      # PP points already counted in points; adjust to avoid double-count
        player_df['shorthandedPoints'] * 0.5 +     # SH points are rarer/higher-leverage
        player_df['gameWinningGoals'] * 0.3 +
        player_df['shots'] * 0.05 # +                # volume/possession proxy
        # player_df['plusMinus'] * 0.1               # weight lightly — see caveat below
    )

    player_df.loc[player_df['position'] != 'goalie', 'value_per60'] = player_df['value'] / player_df['totToi']
    player_df.loc[player_df['totToi'] <= 0, 'value_per60'] = 0

    # compute goalie value based on features
    player_df.loc[player_df['position'] == 'goalie', 'value'] = (
        player_df['savePctg'] * 100 +              # dominant term, most direct skill signal
        (player_df['shutouts'] * 2) -
        player_df['goalsAgainstAvg'] * 5 +          # penalize high GAA
        (player_df['wins'] / player_df['gamesStarted'].replace(0, 1)) * 10  # win rate as starter
    )

    player_df.loc[player_df['position'] == 'goalie', 'value_per60'] = player_df['value'] / player_df['totToi']
    player_df.loc[player_df['totToi'] <= 0, 'value_per60'] = 0

    player_df = player_df.sort_values(by=['playerId', cons.season_name_col], ascending=True)

    return player_df


def compute_player_asof_value(values_df, playerid, decay_rate=0.7):

    # initialized to non-null average across all player data
    value_per60 = 0.001056387
    value = 35.77355929

    if values_df.empty:
        return value_per60, value

    weights = [decay_rate ** i for i in range(len(values_df))]
    value_per60 = np.average(values_df['value_per60'], weights=weights)
    value = np.average(values_df['value'], weights=weights)

    return value_per60, value


def lineup_strength(data_df, player_df):

    # for every matchup, compute the sum value of each lineup's player values based off a weighted average of each player's previous season's value and value_per60
    home_lineup_strengths_per60 = []
    away_lineup_strengths_per60 = []
    home_lineup_strengths = []
    away_lineup_strengths = []

    for index, row in data_df.iterrows():
        home_lineup = json.loads(row[cons.home_lineup_col])
        away_lineup = json.loads(row[cons.away_lineup_col])

        home_strength_per60 = 0
        away_strength_per60 = 0
        home_strength = 0
        away_strength = 0

        for player in home_lineup:
            player_career_values = player_df.loc[(player_df['playerId'] == player) &
                                                     (player_df[cons.season_name_col] < row[cons.season_name_col])][[cons.season_name_col, 'value', 'value_per60']]

            value_per60, value = compute_player_asof_value(player_career_values, player)
            home_strength_per60 += value_per60 if value_per60 is not None else 0
            home_strength += value if value is not None else 0

        for player in away_lineup:
            player_career_values = player_df.loc[(player_df['playerId'] == player) &
                                                                 (player_df[cons.season_name_col] < row[cons.season_name_col])][[cons.season_name_col, 'value', 'value_per60']]
            value_per60, value = compute_player_asof_value(player_career_values, player)
            away_strength_per60 += value_per60 if value_per60 is not None else 0
            away_strength += value if value is not None else 0

        home_lineup_strengths_per60.append(home_strength_per60)
        away_lineup_strengths_per60.append(away_strength_per60)
        home_lineup_strengths.append(home_strength)
        away_lineup_strengths.append(away_strength)

    data_df['home_lineup_strength_per60'] = home_lineup_strengths_per60
    data_df['away_lineup_strength_per60'] = away_lineup_strengths_per60
    data_df['home_lineup_strength'] = home_lineup_strengths
    data_df['away_lineup_strength'] = away_lineup_strengths

    return data_df