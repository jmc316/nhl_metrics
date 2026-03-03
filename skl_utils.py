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


def model_analysis(model, df_data):

    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': cons.feature_cols, 'importance': importances})
    print(importance_df.sort_values(by='importance', ascending=False))

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()