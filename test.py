import constants as cons

from schedule import load_sched_df
from utils.player_feat_utils import load_player_df, compute_player_value
from start_goalie import predict_starting_goalies

if __name__ == "__main__":

    sched_df = load_sched_df()
    player_df = load_player_df()
    player_df = compute_player_value(player_df)
    player_df.sort_values(by=['playerId', cons.season_name_col], ascending=[False, False], inplace=True)

    sched_df = predict_starting_goalies(sched_df, player_df=player_df)
