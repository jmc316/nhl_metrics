import pandas as pd
import constants as cons

from utils.file_utils import csvLoad, csvSave
from schedule import load_sched_df_features


def player_features_update(data_df_in=pd.DataFrame, verbose=False):

    if data_df_in.empty:
        data_df = load_sched_df_features()
    else:
        data_df = data_df_in[cons.sched_feature_cols]

    # add player-based features
    player_features = []

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    player_df = load_player_df()

    print(f'\tComputing Player values...')
    player_df = compute_player_value(player_df)

    for feature in player_features:

        print(f'\tAdding {feature}...')

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

    total_toi_hours = (player_df['avgToi'] * player_df['gamesPlayed']) / 60
    player_df.loc[player_df['position'] != 'goalie', 'value_per60'] = player_df['value'] / total_toi_hours
    player_df.loc[total_toi_hours <= 0, 'value_per60'] = 0

    # compute goalie value based on features
    player_df.loc[player_df['position'] == 'goalie', 'value'] = (
        player_df['savePctg'] * 100 +              # dominant term, most direct skill signal
        (player_df['shutouts'] * 2) -
        player_df['goalsAgainstAvg'] * 5 +          # penalize high GAA
        (player_df['wins'] / player_df['gamesStarted'].replace(0, 1)) * 10  # win rate as starter
    )

    total_toi_hours = player_df['timeOnIce'] / 60
    player_df.loc[player_df['position'] == 'goalie', 'value_per60'] = player_df['value'] / total_toi_hours
    player_df.loc[total_toi_hours <= 0, 'value_per60'] = 0

    return player_df