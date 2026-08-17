from schedule_features import sched_features_update
from team_features import team_features_update
from player_features import player_features_update
from goalie_features import goalie_features_update


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
