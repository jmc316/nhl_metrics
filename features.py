import pandas as pd
import haversine as hs
import constants as cons
from file_utils import fileLoad


def dependent_feature_add(feature_df, backfill=True, debug=True):

    # calculate the number of wins for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team wins ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_wins_col, cons.home_team_name_col)

    # calculate the number of losses for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team losses ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_losses_col, cons.home_team_name_col)

    # calculate the number of OTLs for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team OTLs ...')
    feature_df = season_result(feature_df, backfill, cons.home_team_otls_col, cons.home_team_name_col)

    # calculate the points percentage of the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team points percentage ...')
    if cons.home_team_points_percentage_col not in feature_df.columns:
        feature_df[cons.home_team_points_percentage_col] = (feature_df[cons.home_team_wins_col] * 2 + feature_df[cons.home_team_otls_col]) / ((feature_df[cons.home_team_wins_col] + feature_df[cons.home_team_otls_col] + feature_df[cons.home_team_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[cons.home_team_points_percentage_col].isna()]
        feature_df_new[cons.home_team_points_percentage_col] = (feature_df_new[cons.home_team_wins_col] * 2 + feature_df_new[cons.home_team_otls_col]) / ((feature_df_new[cons.home_team_wins_col] + feature_df_new[cons.home_team_otls_col] + feature_df_new[cons.home_team_losses_col]) * 2)
        feature_df.update(feature_df_new[cons.home_team_points_percentage_col])
    feature_df.drop(columns=[cons.home_team_wins_col, cons.home_team_otls_col, cons.home_team_losses_col], inplace=True)

    # calculate the number of wins for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team wins ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_wins_col, cons.away_team_name_col)

    # calculate the number of losses for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team losses ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_losses_col, cons.away_team_name_col)

    # calculate the number of OTLs for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team OTLs ...')
    feature_df = season_result(feature_df, backfill, cons.away_team_otls_col, cons.away_team_name_col)

    # calculate the points percentage of the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team points percentage ...')
    if cons.away_team_points_percentage_col not in feature_df.columns:
        feature_df[cons.away_team_points_percentage_col] = (feature_df[cons.away_team_wins_col] * 2 + feature_df[cons.away_team_otls_col]) / ((feature_df[cons.away_team_wins_col] + feature_df[cons.away_team_otls_col] + feature_df[cons.away_team_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[cons.away_team_points_percentage_col].isna()]
        feature_df_new[cons.away_team_points_percentage_col] = (feature_df_new[cons.away_team_wins_col] * 2 + feature_df_new[cons.away_team_otls_col]) / ((feature_df_new[cons.away_team_wins_col] + feature_df_new[cons.away_team_otls_col] + feature_df_new[cons.away_team_losses_col]) * 2)
        feature_df.update(feature_df_new[cons.away_team_points_percentage_col])
    feature_df.drop(columns=[cons.away_team_wins_col, cons.away_team_otls_col, cons.away_team_losses_col], inplace=True)

    # calculate the number of wins in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 wins ...')
    home_team_prev_7_wins_col = cons.home_team_prev_n_wins_col + '7'
    feature_df = prevN_result(feature_df, backfill, home_team_prev_7_wins_col, cons.home_team_name_col, 7)

    # calculate the number of losses in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 losses ...')
    home_team_prev_7_losses_col = cons.home_team_prev_n_losses_col + '7'
    feature_df = prevN_result(feature_df, backfill, home_team_prev_7_losses_col, cons.home_team_name_col, 7)

    # calculate the number of OTLs in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 7 OTLs ...')
    home_team_prev_7_otls_col = cons.home_team_prev_n_otls_col + '7'
    feature_df = prevN_result(feature_df, backfill, home_team_prev_7_otls_col, cons.home_team_name_col, 7)

    # calculate the points percentage in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 7 points percentage ...')
    home_team_prev_7_points_percentage_col = cons.home_team_prev_n_points_percentage_col + '7'
    if home_team_prev_7_points_percentage_col not in feature_df.columns:
        feature_df[home_team_prev_7_points_percentage_col] = (feature_df[home_team_prev_7_wins_col] * 2 + feature_df[home_team_prev_7_otls_col]) / ((feature_df[home_team_prev_7_wins_col] + feature_df[home_team_prev_7_otls_col] + feature_df[home_team_prev_7_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[home_team_prev_7_points_percentage_col].isna()]
        feature_df_new[home_team_prev_7_points_percentage_col] = (feature_df_new[home_team_prev_7_wins_col] * 2 + feature_df_new[home_team_prev_7_otls_col]) / ((feature_df_new[home_team_prev_7_wins_col] + feature_df_new[home_team_prev_7_otls_col] + feature_df_new[home_team_prev_7_losses_col]) * 2)
        feature_df.update(feature_df_new[home_team_prev_7_points_percentage_col])
    feature_df.drop(columns=[home_team_prev_7_wins_col, home_team_prev_7_otls_col, home_team_prev_7_losses_col], inplace=True)

    # calculate the number of wins in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team prev 7 wins ...')
    away_team_prev_7_wins_col = cons.away_team_prev_n_wins_col + '7'
    feature_df = prevN_result(feature_df, backfill, away_team_prev_7_wins_col, cons.away_team_name_col, 7)

    # calculate the number of losses in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team prev 7 losses ...')
    away_team_prev_7_losses_col = cons.away_team_prev_n_losses_col + '7'
    feature_df = prevN_result(feature_df, backfill, away_team_prev_7_losses_col, cons.away_team_name_col, 7)

    # calculate the number of OTLs in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team prev 7 OTLs ...')
    away_team_prev_7_otls_col = cons.away_team_prev_n_otls_col + '7'
    feature_df = prevN_result(feature_df, backfill, away_team_prev_7_otls_col, cons.away_team_name_col, 7)

    # calculate the points percentage in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 7 points percentage ...')
    away_team_prev_7_points_percentage_col = cons.away_team_prev_n_points_percentage_col + '7'
    if away_team_prev_7_points_percentage_col not in feature_df.columns:
        feature_df[away_team_prev_7_points_percentage_col] = (feature_df[away_team_prev_7_wins_col] * 2 + feature_df[away_team_prev_7_otls_col]) / ((feature_df[away_team_prev_7_wins_col] + feature_df[away_team_prev_7_otls_col] + feature_df[away_team_prev_7_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[away_team_prev_7_points_percentage_col].isna()]
        feature_df_new[away_team_prev_7_points_percentage_col] = (feature_df_new[away_team_prev_7_wins_col] * 2 + feature_df_new[away_team_prev_7_otls_col]) / ((feature_df_new[away_team_prev_7_wins_col] + feature_df_new[away_team_prev_7_otls_col] + feature_df_new[away_team_prev_7_losses_col]) * 2)
        feature_df.update(feature_df_new[away_team_prev_7_points_percentage_col])
    feature_df.drop(columns=[away_team_prev_7_wins_col, away_team_prev_7_otls_col, away_team_prev_7_losses_col], inplace=True)

    # calculate the number of wins in the previous 3 games for the home team in all matchups
    home_team_prev_3_wins_col = cons.home_team_prev_n_wins_col + '3'
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 wins ...')
    feature_df = prevN_result(feature_df, backfill, home_team_prev_3_wins_col, cons.home_team_name_col, 3)

    # calculate the number of losses in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 losses ...')
    home_team_prev_3_losses_col = cons.home_team_prev_n_losses_col + '3'
    feature_df = prevN_result(feature_df, backfill, home_team_prev_3_losses_col, cons.home_team_name_col, 3)

    # calculate the number of OTLs in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [sub_feature_creation] home team prev 3 OTLs ...')
    home_team_prev_3_otls_col = cons.home_team_prev_n_otls_col + '3'
    feature_df = prevN_result(feature_df, backfill, home_team_prev_3_otls_col, cons.home_team_name_col, 3)

    # calculate the points percentage in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 3 points percentage ...')
    home_team_prev_3_points_percentage_col = cons.home_team_prev_n_points_percentage_col + '3'
    if home_team_prev_3_points_percentage_col not in feature_df.columns:
        feature_df[home_team_prev_3_points_percentage_col] = (feature_df[home_team_prev_3_wins_col] * 2 + feature_df[home_team_prev_3_otls_col]) / ((feature_df[home_team_prev_3_wins_col] + feature_df[home_team_prev_3_otls_col] + feature_df[home_team_prev_3_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[home_team_prev_3_points_percentage_col].isna()]
        feature_df_new[home_team_prev_3_points_percentage_col] = (feature_df_new[home_team_prev_3_wins_col] * 2 + feature_df_new[home_team_prev_3_otls_col]) / ((feature_df_new[home_team_prev_3_wins_col] + feature_df_new[home_team_prev_3_otls_col] + feature_df_new[home_team_prev_3_losses_col]) * 2)
        feature_df.update(feature_df_new[home_team_prev_3_points_percentage_col])
    feature_df.drop(columns=[home_team_prev_3_wins_col, home_team_prev_3_otls_col, home_team_prev_3_losses_col], inplace=True)

    # calculate the number of wins in the previous 3 games for the away team in all matchups
    away_team_prev_3_wins_col = cons.away_team_prev_n_wins_col + '3'
    if debug: print('\t\t... [sub_feature_creation] away team prev 3 wins ...')
    feature_df = prevN_result(feature_df, backfill, away_team_prev_3_wins_col, cons.away_team_name_col, 3)

    # calculate the number of losses in the previous 3 games for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team prev 3 losses ...')
    away_team_prev_3_losses_col = cons.away_team_prev_n_losses_col + '3'
    feature_df = prevN_result(feature_df, backfill, away_team_prev_3_losses_col, cons.away_team_name_col, 3)

    # calculate the number of OTLs in the previous 3 games for the away team in all matchups
    if debug: print('\t\t... [sub_feature_creation] away team prev 3 OTLs ...')
    away_team_prev_3_otls_col = cons.away_team_prev_n_otls_col + '3'
    feature_df = prevN_result(feature_df, backfill, away_team_prev_3_otls_col, cons.away_team_name_col, 3)

    # calculate the points percentage in the previous 3 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 3 points percentage ...')
    away_team_prev_3_points_percentage_col = cons.away_team_prev_n_points_percentage_col + '3'
    if away_team_prev_3_points_percentage_col not in feature_df.columns:
        feature_df[away_team_prev_3_points_percentage_col] = (feature_df[away_team_prev_3_wins_col] * 2 + feature_df[away_team_prev_3_otls_col]) / ((feature_df[away_team_prev_3_wins_col] + feature_df[away_team_prev_3_otls_col] + feature_df[away_team_prev_3_losses_col]) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[away_team_prev_3_points_percentage_col].isna()]
        feature_df_new[away_team_prev_3_points_percentage_col] = (feature_df_new[away_team_prev_3_wins_col] * 2 + feature_df_new[away_team_prev_3_otls_col]) / ((feature_df_new[away_team_prev_3_wins_col] + feature_df_new[away_team_prev_3_otls_col] + feature_df_new[away_team_prev_3_losses_col]) * 2)
        feature_df.update(feature_df_new[away_team_prev_3_points_percentage_col])
    feature_df.drop(columns=[away_team_prev_3_wins_col, away_team_prev_3_otls_col, away_team_prev_3_losses_col], inplace=True)

    # calculate goals for in the previous 3 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 3 goals for ...')
    feature_df = prevN_gfpg(3, feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col)

    # calculate goals for in the previous 3 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 3 goals for ...')
    feature_df = prevN_gfpg(3, feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col)

        # calculate goals for in the previous 7 games for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team prev 7 goals for ...')
    feature_df = prevN_gfpg(7, feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col)

    # calculate goals for in the previous 7 games for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team prev 7 goals for ...')
    feature_df = prevN_gfpg(7, feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col)

    # calculate days since last played game for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.home_team_days_since_last_game_col, cons.home_team_name_col)

    # calculate days since last played game for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.away_team_days_since_last_game_col, cons.away_team_name_col)

    geoloc_df = fileLoad(cons.util_data_folder, cons.venue_geolocations_filename)
    feature_df = feature_df.merge(geoloc_df, how='left', left_on=cons.venue_col, right_on=cons.venue_col)

    # calculate the haversine distance for the last 7 days for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team travel distance in last 7 days ...')
    feature_df = hav_dist_7days(feature_df, cons.home_team_travel_distance_7days_col, cons.home_team_name_col, backfill)

    # calculate the haversine distance for the last 7 days for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team travel distance in last 7 days ...')
    feature_df = hav_dist_7days(feature_df, cons.away_team_travel_distance_7days_col, cons.away_team_name_col, backfill)

    feature_df.drop(columns=[cons.venue_col+'_lat', cons.venue_col+'_long'], inplace=True)

    return feature_df


def datetime_feature_add(feature_df):

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    feature_df[cons.starttime_utc_col] = pd.to_datetime(feature_df[cons.starttime_utc_col]).dt.tz_convert('US/Eastern')
    feature_df[cons.game_date_col] = feature_df[cons.starttime_utc_col].dt.date
    feature_df[cons.game_time_col] = feature_df[cons.starttime_utc_col].dt.hour * 60 + feature_df[cons.starttime_utc_col].dt.minute
    feature_df.drop(columns=[cons.starttime_utc_col], inplace=True)

    return feature_df


def prevN_result(data_df, backfill, target_col, home_away_team_col, n):
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()]

    # initialize the target column with zeros
    data_df_target.loc[data_df_target[cons.last_period_col].isna(), target_col] = 0

    # iterate through each row in the DataFrame and calculate the number of results in the previous n games for the specified team
    for idx, row in data_df_target.iterrows():
        team = row[home_away_team_col]
        game_date = row[cons.game_date_col]
        season = row[cons.season_name_col]
        prev_games = data_df.loc[
            (pd.to_datetime(data_df[cons.game_date_col]).dt.date < game_date) &
            (data_df[cons.season_name_col] == season) &
            ((data_df[cons.home_team_name_col] == team) | (data_df[cons.away_team_name_col] == team))
        ].tail(n)
        if 'Wins' in target_col:
            target_df = prev_games.loc[
                ((prev_games[cons.home_team_name_col] == team) &
                 (prev_games[cons.home_team_score_col] > prev_games[cons.away_team_score_col])) |
                ((prev_games[cons.away_team_name_col] == team) &
                 (prev_games[cons.away_team_score_col] > prev_games[cons.home_team_score_col]))
            ].shape[0]
        elif 'Losses' in target_col:
            target_df = prev_games.loc[
                ((prev_games[cons.home_team_name_col] == team) &
                 (prev_games[cons.home_team_score_col] < prev_games[cons.away_team_score_col]) &
                 (prev_games[cons.last_period_col] == 'REG')) |
                ((prev_games[cons.away_team_name_col] == team) &
                 (prev_games[cons.away_team_score_col] < prev_games[cons.home_team_score_col]) &
                 (prev_games[cons.last_period_col] == 'REG'))
            ].shape[0]
        elif 'OTL' in target_col:
            target_df = prev_games.loc[
                ((prev_games[cons.home_team_name_col] == team) &
                 (prev_games[cons.home_team_score_col] < prev_games[cons.away_team_score_col]) &
                 (prev_games[cons.last_period_col] != 'REG')) |
                ((prev_games[cons.away_team_name_col] == team) &
                 (prev_games[cons.away_team_score_col] < prev_games[cons.home_team_score_col]) &
                 (prev_games[cons.last_period_col] != 'REG'))
            ].shape[0]
        data_df_target.at[idx, target_col] = target_df

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def season_result(data_df, backfill, target_col, home_away_team_col):
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()]

    # initialize the target column with zeros
    data_df_target.loc[data_df_target[cons.last_period_col].isna(), target_col] = 0

    # iterate through each row in the DataFrame and calculate the number of results in the previous 10 games for the specified team
    for idx, row in data_df_target.iterrows():
        team = row[home_away_team_col]
        game_date = row[cons.game_date_col]
        season = row[cons.season_name_col]
        season_games = data_df.loc[
            (pd.to_datetime(data_df[cons.game_date_col]).dt.date < game_date) &
            (data_df[cons.season_name_col] == season) &
            ((data_df[cons.home_team_name_col] == team) | (data_df[cons.away_team_name_col] == team))
        ]
        if 'Wins' in target_col:
            target_df = season_games.loc[
                ((season_games[cons.home_team_name_col] == team) &
                 (season_games[cons.home_team_score_col] > season_games[cons.away_team_score_col])) |
                ((season_games[cons.away_team_name_col] == team) &
                 (season_games[cons.away_team_score_col] > season_games[cons.home_team_score_col]))
            ].shape[0]
        elif 'Losses' in target_col:
            target_df = season_games.loc[
                ((season_games[cons.home_team_name_col] == team) &
                 (season_games[cons.home_team_score_col] < season_games[cons.away_team_score_col]) &
                 (season_games[cons.last_period_col] == 'REG')) |
                ((season_games[cons.away_team_name_col] == team) &
                 (season_games[cons.away_team_score_col] < season_games[cons.home_team_score_col]) &
                 (season_games[cons.last_period_col] == 'REG'))
            ].shape[0]
        elif 'OTL' in target_col:
            target_df = season_games.loc[
                ((season_games[cons.home_team_name_col] == team) &
                 (season_games[cons.home_team_score_col] < season_games[cons.away_team_score_col]) &
                 (season_games[cons.last_period_col] != 'REG')) |
                ((season_games[cons.away_team_name_col] == team) &
                 (season_games[cons.away_team_score_col] < season_games[cons.home_team_score_col]) &
                 (season_games[cons.last_period_col] != 'REG'))
            ].shape[0]
        data_df_target.at[idx, target_col] = target_df

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def prevN_gfpg(n, data_df, backfill, target_col, home_away_team_col):
    
    # initialize the target column with zeros
    target_col = target_col + str(n)
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()]

    # initialize the target column with zeros
    data_df_target.loc[data_df_target[cons.last_period_col].isna(), target_col] = 0

    # iterate through each row in the DataFrame and calculate the number of results in the previous 10 games for the specified team
    for idx, row in data_df_target.iterrows():
        team = row[home_away_team_col]
        game_date = row[cons.game_date_col]
        season = row[cons.season_name_col]
        prev_games = data_df.loc[
            (pd.to_datetime(data_df[cons.game_date_col]).dt.date < game_date) &
            (data_df[cons.season_name_col] == season) &
            ((data_df[cons.home_team_name_col] == team) | (data_df[cons.away_team_name_col] == team))
        ].tail(n)
        target_df = prev_games.loc[
            (prev_games[cons.home_team_name_col] == team)
        ][cons.home_team_score_col].sum() + prev_games.loc[
            (prev_games[cons.away_team_name_col] == team)
        ][cons.away_team_score_col].sum()
        data_df_target.at[idx, target_col] = target_df

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def season_gfpg(data_df, backfill, target_col, home_away_team_col):
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()]

    # initialize the target column with zeros
    data_df_target.loc[data_df_target[cons.last_period_col].isna(), target_col] = 0

    # iterate through each row in the DataFrame and calculate the number of results in the previous 10 games for the specified team
    for idx, row in data_df_target.iterrows():
        team = row[home_away_team_col]
        game_date = row[cons.game_date_col]
        season = row[cons.season_name_col]
        season_games = data_df.loc[
            (pd.to_datetime(data_df[cons.game_date_col]).dt.date < game_date) &
            (data_df[cons.season_name_col] == season) &
            ((data_df[cons.home_team_name_col] == team) | (data_df[cons.away_team_name_col] == team))
        ]
        target_df = season_games.loc[
            (season_games[cons.home_team_name_col] == team)
        ][cons.home_team_score_col].sum() + season_games.loc[
            (season_games[cons.away_team_name_col] == team)
        ][cons.away_team_score_col].sum()
        data_df_target.at[idx, target_col] = target_df

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def days_since_last_played(data_df, target_col, team_col):

    # calculate days since last game regardless of home/away team
    data_df[cons.game_date_col+'_dt'] = pd.to_datetime(data_df[cons.game_date_col], errors='coerce')

    row_id_col = '_row_id'
    team_col_name = 'team'
    source_col_name = 'source_col'
    days_col_name = 'days_since_last_game'
    season_col_name = 'season'

    base_df = data_df[[cons.game_date_col+'_dt', cons.home_team_name_col, cons.away_team_name_col, cons.season_name_col]].reset_index()
    base_df.rename(columns={'index': row_id_col}, inplace=True)

    home_games = base_df[[row_id_col, cons.game_date_col+'_dt', cons.home_team_name_col, cons.season_name_col]].rename(
        columns={cons.home_team_name_col: team_col_name, cons.season_name_col: season_col_name}
    )
    home_games[source_col_name] = cons.home_team_name_col

    away_games = base_df[[row_id_col, cons.game_date_col+'_dt', cons.away_team_name_col, cons.season_name_col]].rename(
        columns={cons.away_team_name_col: team_col_name, cons.season_name_col: season_col_name}
    )
    away_games[source_col_name] = cons.away_team_name_col

    all_team_games = pd.concat([home_games, away_games], ignore_index=True)
    all_team_games.sort_values(by=[team_col_name, cons.game_date_col+'_dt', row_id_col], inplace=True)
    all_team_games[days_col_name] = all_team_games.groupby([team_col_name, season_col_name])[cons.game_date_col+'_dt'].diff().dt.days

    base_df[target_col] = all_team_games.loc[
        all_team_games[source_col_name] == team_col
    ].set_index(row_id_col)[days_col_name].reindex(base_df.index)

    base_df.drop(columns=[cons.game_date_col+'_dt', cons.home_team_name_col, cons.away_team_name_col, '_row_id', cons.season_name_col], inplace=True)
    data_df.drop(columns=[cons.game_date_col+'_dt'], inplace=True)
    if target_col in data_df.columns:
        data_df.drop(columns=[target_col], inplace=True)

    data_df = data_df.merge(base_df, how='left', left_on=data_df.index, right_on=base_df.index)
    data_df.drop(columns=['key_0'], inplace=True)

    return data_df


def hav_dist_7days(data_df, target_col, team_col, backfill):

    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()] 

    # calculate the haversine distance for the last 7 days for the specified team in all matchups
    data_df_target[target_col] = None
    for idx, row in data_df_target.iterrows():
        team = row[team_col]
        game_date = row[cons.game_date_col]
        season = row[cons.season_name_col]
        team_games = data_df.loc[
            (pd.to_datetime(data_df[cons.game_date_col]).dt.date < game_date) &
            (data_df[cons.season_name_col] == season) &
            ((data_df[cons.home_team_name_col] == team) | (data_df[cons.away_team_name_col] == team))
        ].tail(7)
        distances = []
        for _, game in team_games.iterrows():
            if pd.notna(game[cons.venue_col+'_lat']) and pd.notna(game[cons.venue_col+'_long']):
                distance = hs.haversine((row[cons.venue_col+'_lat'], row[cons.venue_col+'_long']), (game[cons.venue_col+'_lat'], game[cons.venue_col+'_long']))
                distances.append(distance)
        if distances:
            data_df_target.at[idx, target_col] = sum(distances) / len(distances)
        else:
            data_df_target.at[idx, target_col] = None

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)
    
    return data_df