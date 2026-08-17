import constants as cons
from file_utils import csvSave


def player_features_update():

    # load all of the player data
    data_df = load_player_df()

    # add player-based features
    player_features = []

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    for feature in player_features:

        print(f'\tAdding {feature}...')

    for season in data_df[cons.season_name_col].unique():
        print(f'Writing player features for season: {season}...')
        data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
        csvSave(data_df_season, f'{cons.season_feature_sets_folder}Player_features/', f'{season}_player_features.csv')

    return data_df, player_features


def load_player_df():

    pass