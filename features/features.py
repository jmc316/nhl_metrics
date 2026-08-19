import os

import pandas as pd
import constants as cons

from utils.file_utils import csvLoad, csvSave
from features.feat_team import team_features_update
from features.feat_schedule import sched_features_update
from features.feat_player import player_features_update
from features.feat_goalie import goalie_features_update


def feature_data_update(sched_feat_df, team_feat_df, player_feat_df, goalie_feat_df, save_feat_data):

    # merge all feature dataframes into a single dataframe
    merge_cols = [cons.game_id_col, cons.season_name_col, cons.game_type_col, cons.starttime_est_col,
                  cons.venue_timezone_col, cons.venue_col, cons.home_team_name_col, cons.away_team_name_col,
                  cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col, cons.home_team_win_col,
                  cons.win_prob_col.format(team='home'), cons.win_prob_col.format(team='away')]
    sched_df = sched_feat_df.merge(team_feat_df, how='left', on=merge_cols)
    sched_df = sched_df.merge(player_feat_df, how='left', on=merge_cols)
    sched_df = sched_df.merge(goalie_feat_df, how='left', on=merge_cols)

    if save_feat_data:
        for season in sched_df[cons.season_name_col].unique():
            season_df = sched_df[sched_df[cons.season_name_col] == season]
            csvSave(cons.season_feature_sets_folder, f"{season}_feature_data.csv", season_df, index=False)

    return sched_df


def clean_feature_df(data_df):

    data_df[cons.starttime_est_col] = pd.to_datetime(data_df[cons.starttime_est_col], format='ISO8601')

    data_df.sort_values(by=cons.starttime_est_col, inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    for col in data_df.columns:
        if pd.api.types.is_integer_dtype(data_df[col]):
            data_df[col] = data_df[col].astype(int)

    return data_df


def feature_data_load():

    # the list of feature files that have already been generated
    season_sched_list = [file for file in os.listdir(cons.season_feature_sets_folder) if file.endswith('_feature_data.csv')]

    # initialize empty dataframe to store the feature data
    feat_df = pd.DataFrame()

    # loop through each feature file and concatenate it to the feature data dataframe;
    # if there are no feature files, throw an error
    if not season_sched_list:
        raise FileNotFoundError(f"No feature files found in {cons.season_feature_sets_folder}")
    for filename in season_sched_list:
        temp_df = csvLoad(cons.season_feature_sets_folder, filename)
        feat_df = pd.concat([feat_df, temp_df], ignore_index=True)

    clean_feature_df(feat_df)

    return feat_df


def feat_update(data_df=pd.DataFrame, save_feat_data=False, verbose=False):
    print('Updating all feature data...')

    sched_feat_df, sched_features = sched_features_update(data_df, verbose)
    team_feat_df, team_features = team_features_update(data_df, verbose)
    player_feat_df, player_features = player_features_update(data_df, verbose)
    goalie_feat_df, goalie_features = goalie_features_update(data_df, verbose)

    feature_df = feature_data_update(sched_feat_df, team_feat_df, player_feat_df, goalie_feat_df, save_feat_data)

    print('All feature data updated.\n')

    feature_df.sort_values(by=cons.starttime_est_col, inplace=True)

    return feature_df


def update_schedule_feature_data():
    print('Updating schedule feature data...')

    sched_features_update()

    print('Schedule feature data updated.\n')


def update_team_feature_data():
    print('Updating team feature data...')

    team_features_update()

    print('Team feature data updated.\n')


def update_player_feature_data():
    print('Updating player feature data...')

    player_features_update()

    print('Player feature data updated.\n')


def update_goalie_feature_data():
    print('Updating goalie feature data...')

    goalie_features_update()

    print('Goalie feature data updated.\n')