import pandas as pd
import constants as cons

from utils.file_utils import csvSave
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

    for feature in player_features:

        print(f'\tAdding {feature}...')

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            print(f'Writing player features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.player_features_folder, cons.player_features_filename.format(season=season))

    return data_df, player_features


def load_player_df():

    pass