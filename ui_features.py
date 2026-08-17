from feat_schedule import sched_features_update
from feat_team import team_features_update
from feat_player import player_features_update
from feat_goalie import goalie_features_update
from features import feature_data_update


def ui_update_all_feature_data():
    print('Updating all feature data...')

    sched_feat_df, sched_features = sched_features_update()
    team_feat_df, team_features = team_features_update()
    player_feat_df, player_features = player_features_update()
    goalie_feat_df, goalie_features = goalie_features_update()

    feature_data_update(sched_feat_df, team_feat_df, player_feat_df, goalie_feat_df)

    print('All feature data updated.\n')


def ui_update_schedule_feature_data():
    print('Updating schedule feature data...')

    sched_features_update()

    print('Schedule feature data updated.\n')


def ui_update_team_feature_data():
    print('Updating team feature data...')

    team_features_update()

    print('Team feature data updated.\n')


def ui_update_player_feature_data():
    print('Updating player feature data...')

    player_features_update()

    print('Player feature data updated.\n')


def ui_update_goalie_feature_data():
    print('Updating goalie feature data...')

    goalie_features_update()

    print('Goalie feature data updated.\n')
