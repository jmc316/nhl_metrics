import constants as cons
from file_utils import csvSave


def team_features_update():

    # load all of the team data
    data_df = load_team_df()

    # add team-based features
    team_features = []

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    for feature in team_features:

        print(f'\tAdding {feature}...')

    for season in data_df[cons.season_name_col].unique():
        print(f'Writing team features for season: {season}...')
        data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
        csvSave(data_df_season, f'{cons.season_feature_sets_folder}Team_features/', f'{season}_team_features.csv')

    return data_df, team_features


def load_team_df():

    pass