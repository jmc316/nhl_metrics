"""Orchestrates building, merging, saving, and loading the full (schedule/team/player/goalie) feature set."""

import os

import pandas as pd
import constants as cons

from utils.file_utils import csvLoad, csvSave
from features.feat_team import team_features_update
from features.feat_player import player_features_update
from features.feat_goalie import goalie_features_update
from features.feat_schedule import sched_features_update


def feature_data_update(sched_feat_df, team_feat_df, player_feat_df, goalie_feat_df, save_feat_data):
    """Merge the schedule/team/player/goalie feature dataframes into one, optionally saving per season."""

    # merge all feature dataframes into a single dataframe
    merge_cols = [cons.game_id_col, cons.season_name_col, cons.game_type_col, cons.starttime_est_col,
                  cons.venue_timezone_col, cons.venue_col, cons.home_team_name_col, cons.away_team_name_col,
                  cons.home_team_score_col, cons.away_team_score_col, cons.last_period_col, cons.home_team_win_col,
                  cons.home_win_prob_col, cons.away_win_prob_col]
    sched_df = sched_feat_df.merge(team_feat_df, how='left', on=merge_cols)
    sched_df = sched_df.merge(player_feat_df, how='left', on=merge_cols)
    sched_df = sched_df.merge(goalie_feat_df, how='left', on=merge_cols)

    if save_feat_data:
        sched_df.sort_values(by=[cons.game_id_col, cons.starttime_est_col, cons.home_team_name_col], inplace=True)
        for season in sched_df[cons.season_name_col].unique():
            season_df = sched_df[sched_df[cons.season_name_col] == season]
            csvSave(season_df, cons.season_feature_sets_folder, cons.feat_data_filename.format(season=season))

    return sched_df


def clean_feature_df(data_df):
    """Normalize dtypes/sort order of a loaded feature dataframe."""

    data_df[cons.starttime_est_col] = pd.to_datetime(data_df[cons.starttime_est_col], format='mixed')

    data_df.sort_values(by=[cons.game_id_col, cons.starttime_est_col, cons.home_team_name_col], inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    for col in data_df.columns:
        if pd.api.types.is_integer_dtype(data_df[col]):
            data_df[col] = data_df[col].astype(int)

    return data_df


def feature_data_load():
    """Load and concatenate all previously saved per-season feature files into one dataframe."""

    # the list of feature files that have already been generated
    season_sched_list = [file for file in os.listdir(cons.season_feature_sets_folder) if file.endswith(cons.feat_data_filename.format(season='$').split('$')[1])]

    # initialize empty dataframe to store the feature data
    feat_df = pd.DataFrame()

    # loop through each feature file and concatenate it to the feature data dataframe;
    # if there are no feature files, throw an error
    if not season_sched_list:
        raise FileNotFoundError(f"No feature files found in {cons.season_feature_sets_folder}")
    for filename in season_sched_list:
        temp_df = csvLoad(cons.season_feature_sets_folder, filename)
        feat_df = pd.concat([feat_df, temp_df], ignore_index=True)

    feat_df = clean_feature_df(feat_df)

    return feat_df


def feat_update(data_df=pd.DataFrame, term_out=True, save_feat_data=False, verbose=False, player_value_formula=None,
                existing_sched_feat_df=None, existing_team_feat_df=None, existing_player_feat_df=None, existing_goalie_feat_df=None,
                append_mode=False):
    """Rebuild schedule, team, player, and goalie features and merge them into one feature dataframe.

    Each existing_*_feat_df param, if provided, is the previously computed feature dataframe (in that
    domain's own output format) to append newly appearing games onto, instead of recomputing every game.
    If append_mode is True, any existing_*_feat_df left as None is auto-loaded from the previously
    saved per-domain feature CSVs on disk, so the caller doesn't have to load them manually.
    """

    if term_out: print('Updating all feature data...')

    if append_mode:
        if existing_sched_feat_df is None:
            existing_sched_feat_df = load_domain_feature_data(cons.sched_features_folder, cons.sched_features_filename)
        if existing_team_feat_df is None:
            existing_team_feat_df = load_domain_feature_data(cons.team_features_folder, cons.team_features_filename)
        if existing_player_feat_df is None:
            existing_player_feat_df = load_domain_feature_data(cons.player_features_folder, cons.player_features_filename)
        if existing_goalie_feat_df is None:
            existing_goalie_feat_df = load_domain_feature_data(cons.goalie_features_folder, cons.goalie_features_filename)

    sched_feat_df, sched_features = sched_features_update(data_df, verbose, existing_feat_df=existing_sched_feat_df)
    team_feat_df, team_features = team_features_update(data_df, verbose, existing_feat_df=existing_team_feat_df)
    player_feat_df, player_features = player_features_update(data_df, verbose, player_value_formula=player_value_formula, existing_feat_df=existing_player_feat_df)
    goalie_feat_df, goalie_features = goalie_features_update(data_df, verbose, existing_feat_df=existing_goalie_feat_df)

    feature_df = feature_data_update(sched_feat_df, team_feat_df, player_feat_df, goalie_feat_df, save_feat_data)

    if term_out: print('All feature data updated.\n')

    feature_df.sort_values(by=cons.starttime_est_col, inplace=True)

    return feature_df


def load_domain_feature_data(folder, filename_format):
    """Load previously saved per-season feature files for one feature domain (empty if none exist yet)."""

    if not os.path.isdir(folder):
        return pd.DataFrame()

    filename_suffix = filename_format.format(season='$').split('$')[1]
    season_files = [file for file in os.listdir(folder) if file.endswith(filename_suffix)]
    if not season_files:
        return pd.DataFrame()

    domain_df = pd.DataFrame()
    for filename in season_files:
        domain_df = pd.concat([domain_df, csvLoad(folder, filename)], ignore_index=True)

    domain_df[cons.starttime_est_col] = pd.to_datetime(domain_df[cons.starttime_est_col], format='mixed')

    return domain_df


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