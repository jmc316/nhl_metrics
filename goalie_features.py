import constants as cons
from file_utils import csvSave


def goalie_features_update():

    # load all of the goalie data
    data_df = load_goalie_df()

    # add goalie-based features
    goalie_features = []

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    for feature in goalie_features:

        print(f'\tAdding {feature}...')

    for season in data_df[cons.season_name_col].unique():
        print(f'Writing goalie features for season: {season}...')
        data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
        csvSave(data_df_season, f'{cons.season_feature_sets_folder}Goalie_features/', f'{season}_goalie_features.csv')

    return data_df, goalie_features


def load_goalie_df():

    pass