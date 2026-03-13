import pandas as pd
import constants as cons
from sklearn.inspection import permutation_importance


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
        prev_games = data_df.loc[
            (data_df[cons.game_date_col] < game_date) &
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
        season_games = data_df.loc[
            (data_df[cons.game_date_col] < game_date) &
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
        prev_games = data_df.loc[
            (data_df[cons.game_date_col] < game_date) &
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
        season_games = data_df.loc[
            (data_df[cons.game_date_col] < game_date) &
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


def days_since_last_played(feature_df, target_col, team_col):

    # calculate days since last game regardless of home/away team
    feature_df[cons.game_date_col+'_dt'] = pd.to_datetime(feature_df[cons.game_date_col], errors='coerce')

    row_id_col = '_row_id'
    team_col_name = 'team'
    source_col_name = 'source_col'
    days_col_name = 'days_since_last_game'

    base_df = feature_df[[cons.game_date_col+'_dt', cons.home_team_name_col, cons.away_team_name_col]].reset_index()
    base_df.rename(columns={'index': row_id_col}, inplace=True)

    home_games = base_df[[row_id_col, cons.game_date_col+'_dt', cons.home_team_name_col]].rename(
        columns={cons.home_team_name_col: team_col_name}
    )
    home_games[source_col_name] = cons.home_team_name_col

    away_games = base_df[[row_id_col, cons.game_date_col+'_dt', cons.away_team_name_col]].rename(
        columns={cons.away_team_name_col: team_col_name}
    )
    away_games[source_col_name] = cons.away_team_name_col

    all_team_games = pd.concat([home_games, away_games], ignore_index=True)
    all_team_games.sort_values(by=[team_col_name, cons.game_date_col+'_dt', row_id_col], inplace=True)
    all_team_games[days_col_name] = all_team_games.groupby(team_col_name)[cons.game_date_col+'_dt'].diff().dt.days

    feature_df[target_col] = all_team_games.loc[
        all_team_games[source_col_name] == team_col
    ].set_index(row_id_col)[days_col_name].reindex(feature_df.index)

    feature_df.drop(columns=[cons.game_date_col+'_dt'], inplace=True)

    return feature_df