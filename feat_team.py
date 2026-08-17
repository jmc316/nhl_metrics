import pandas as pd
import constants as cons
from file_utils import csvSave
from schedule import load_sched_df_features


def team_features_update(data_df_in=pd.DataFrame, verbose=False):

    if data_df_in.empty:
        data_df = load_sched_df_features()
    else:
        data_df = data_df_in[cons.base_feature_cols]

    # add team-based features
    team_features = []

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    for feature in team_features:

        print(f'\tAdding {feature}...')

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            print(f'Writing team features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.team_features_folder, cons.team_features_filename.format(season=season))

    return data_df, team_features


def load_team_df():

    pass