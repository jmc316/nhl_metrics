import bisect

import numpy as np
import pandas as pd
import constants as cons

from utils.file_utils import csvSave
from schedule import load_sched_df_features


def team_features_update(data_df_in=pd.DataFrame, verbose=False):

    if data_df_in.empty:
        data_df = load_sched_df_features(feat_set_label='team')
    else:
        data_df = data_df_in[cons.team_feature_cols]

    # add team-based features
    team_features = [cons.point_per_n_col, cons.goal_diff_n_col, cons.corsi_per_n_col, cons.elo_rat_col,
                     cons.pp_per_n_col, cons.pk_per_n_col]
    skipped_features = [cons.fenwick_per_n_col]

    # features to add in the future
    future_features = ['ex_gf_per', 'ex_ga_per', 'shot_per']

    for feature in team_features:

        if verbose: print(f'\tAdding {feature}...')

        # add a feature that calculates the win percentage for the last n games played for each team
        if feature == cons.point_per_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')
                data_df = points_percentage_feature_add(data_df, debug=verbose, backfill=True, team_col=cons.home_team_name_col, n=window)
                data_df = points_percentage_feature_add(data_df, debug=verbose, backfill=True, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_games_played_col = cons.point_per_n_col.format(pre='home', n=window)
                away_games_played_col = cons.point_per_n_col.format(pre='away', n=window)
                data_df[cons.point_per_n_col.format(pre='rel', n=window)] = data_df[home_games_played_col] - data_df[away_games_played_col]
                data_df.drop(columns=[home_games_played_col, away_games_played_col], inplace=True)
            continue

        # add a feature that calculates the goal differential for the last n games played for each team
        if feature == cons.goal_diff_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')

                # calculate the goals for and against for the home team and away team for the last n games played
                data_df = prevN_gpg(data_df, target_col=cons.goal_diff_n_col.format(pre='home', n=window)+'for', backfill=True, team_col=cons.home_team_name_col, n=window, for_against='for')
                data_df = prevN_gpg(data_df, target_col=cons.goal_diff_n_col.format(pre='home', n=window)+'against', backfill=True, team_col=cons.home_team_name_col, n=window, for_against='against')
                data_df = prevN_gpg(data_df, target_col=cons.goal_diff_n_col.format(pre='away', n=window)+'for', backfill=True, team_col=cons.away_team_name_col, n=window, for_against='for')
                data_df = prevN_gpg(data_df, target_col=cons.goal_diff_n_col.format(pre='away', n=window)+'against', backfill=True, team_col=cons.away_team_name_col, n=window, for_against='against')

                # only calculate the goal differential if both the goals for and against are not null
                data_df.loc[~data_df[cons.goal_diff_n_col.format(pre='home', n=window)+'for'].isna() & ~data_df[cons.goal_diff_n_col.format(pre='home', n=window)+'against'].isna(), cons.goal_diff_n_col.format(pre='home', n=window)] = data_df[cons.goal_diff_n_col.format(pre='home', n=window)+'for'] - data_df[cons.goal_diff_n_col.format(pre='home', n=window)+'against']
                data_df.loc[~data_df[cons.goal_diff_n_col.format(pre='away', n=window)+'for'].isna() & ~data_df[cons.goal_diff_n_col.format(pre='away', n=window)+'against'].isna(), cons.goal_diff_n_col.format(pre='away', n=window)] = data_df[cons.goal_diff_n_col.format(pre='away', n=window)+'for'] - data_df[cons.goal_diff_n_col.format(pre='away', n=window)+'against']
                data_df.drop(columns=[cons.goal_diff_n_col.format(pre='home', n=window)+'for', cons.goal_diff_n_col.format(pre='home', n=window)+'against', cons.goal_diff_n_col.format(pre='away', n=window)+'for', cons.goal_diff_n_col.format(pre='away', n=window)+'against'], inplace=True)

                # create relational feature, drop individual features
                home_goal_diff_col = cons.goal_diff_n_col.format(pre='home', n=window)
                away_goal_diff_col = cons.goal_diff_n_col.format(pre='away', n=window)

                # only calculate relational feature if both the home and away goal differentials are not null
                data_df.loc[~data_df[home_goal_diff_col].isna() & ~data_df[away_goal_diff_col].isna(), cons.goal_diff_n_col.format(pre='rel', n=window)] = data_df[home_goal_diff_col] - data_df[away_goal_diff_col]
                data_df.drop(columns=[home_goal_diff_col, away_goal_diff_col], inplace=True)
            continue

        # add a feature that calculates the corsi percentage for the last n games played for each team
        if feature == cons.corsi_per_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')

                data_df = prevN_corsi(data_df, target_col=cons.corsi_per_n_col.format(pre='home', n=window), backfill=True, team_col=cons.home_team_name_col, n=window)
                data_df = prevN_corsi(data_df, target_col=cons.corsi_per_n_col.format(pre='away', n=window), backfill=True, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_corsi_col = cons.corsi_per_n_col.format(pre='home', n=window)
                away_corsi_col = cons.corsi_per_n_col.format(pre='away', n=window)

                # only calculate relational feature if both the home and away corsi percentages are not null
                data_df.loc[~data_df[home_corsi_col].isna() & ~data_df[away_corsi_col].isna(), cons.corsi_per_n_col.format(pre='rel', n=window)] = data_df[home_corsi_col] - data_df[away_corsi_col]
                data_df.drop(columns=[home_corsi_col, away_corsi_col], inplace=True)
            continue

        # add a feature that calculates the fenwick percentage for the last n games played for each team
        if feature == cons.fenwick_per_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')

                data_df = prevN_fenwick(data_df, target_col=cons.fenwick_per_n_col.format(pre='home', n=window), backfill=True, team_col=cons.home_team_name_col, n=window)
                data_df = prevN_fenwick(data_df, target_col=cons.fenwick_per_n_col.format(pre='away', n=window), backfill=True, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_fenwick_col = cons.fenwick_per_n_col.format(pre='home', n=window)
                away_fenwick_col = cons.fenwick_per_n_col.format(pre='away', n=window)
                
                # only calculate relational feature if both the home and away fenwick percentages are not null
                data_df.loc[~data_df[home_fenwick_col].isna() & ~data_df[away_fenwick_col].isna(), cons.fenwick_per_n_col.format(pre='rel', n=window)] = data_df[home_fenwick_col] - data_df[away_fenwick_col]
                data_df.drop(columns=[home_fenwick_col, away_fenwick_col], inplace=True)
            continue

        # add a feature that calculates the power play percentage for the last n games played for each team
        if feature == cons.pp_per_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')

                data_df = add_ppper(data_df, team_col=cons.home_team_name_col, n=window)
                data_df = add_ppper(data_df, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_pp_col = cons.pp_per_n_col.format(pre='home', n=window)
                away_pp_col = cons.pp_per_n_col.format(pre='away', n=window)

                # only calculate relational feature if both the home and away power play percentages are not null
                data_df.loc[~data_df[home_pp_col].isna() & ~data_df[away_pp_col].isna(), cons.pp_per_n_col.format(pre='rel', n=window)] = data_df[home_pp_col] - data_df[away_pp_col]
                data_df.drop(columns=[home_pp_col, away_pp_col], inplace=True)

        # add a feature that calculates the penalty kill percentage for the last n games played for each team
        if feature == cons.pk_per_n_col:
            for window in cons.team_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')

                data_df = add_pkper(data_df, team_col=cons.home_team_name_col, n=window)
                data_df = add_pkper(data_df, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_pk_col = cons.pk_per_n_col.format(pre='home', n=window)
                away_pk_col = cons.pk_per_n_col.format(pre='away', n=window)

                # only calculate relational feature if both the home and away penalty kill percentages are not null
                data_df.loc[~data_df[home_pk_col].isna() & ~data_df[away_pk_col].isna(), cons.pk_per_n_col.format(pre='rel', n=window)] = data_df[home_pk_col] - data_df[away_pk_col]
                data_df.drop(columns=[home_pk_col, away_pk_col], inplace=True)

        # add a feature that is a team's elo rating, which is a measure of team strength based on game results
        if feature == cons.elo_rat_col:
            data_df = compute_elo_ratings(data_df)
            data_df.drop(columns=[cons.elo_rat_col.format(pre='home'), cons.elo_rat_col.format(pre='away')], inplace=True)
            continue

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            print(f'Writing team features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.team_features_folder, cons.team_features_filename.format(season=season))

    return data_df, team_features


def load_team_df():

    pass


def points_percentage_feature_add(feature_df, debug, backfill, team_col, n=None):

    # assign n to a large number to capture all games in the current season for the specified team
    if not n:
        n = cons.max_single_season_games
        n_str = 'season'
    else:
        n_str = 'previous ' + str(n) + ' games'

    # calculate the number of wins for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} win total ...')
    feature_df = prevN_result(feature_df, backfill, team_col[:4]+'Wins', team_col, n)

    # calculate the number of losses for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} loss total ...')
    feature_df = prevN_result(feature_df, backfill, team_col[:4]+'Losses', team_col, n)

    # calculate the number of OTLs for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} OTL total ...')
    feature_df = prevN_result(feature_df, backfill, team_col[:4]+'OTLs', team_col, n)

    # calculate the points percentage of the home team in all matchups
    # if debug: print(f'\t\t... [feature_creation] {team_col[:4]} team {n_str} points percentage ...')
    if cons.point_per_n_col.format(pre=team_col[:4], n=n) not in feature_df.columns:
        feature_df[cons.point_per_n_col.format(pre=team_col[:4], n=n)] = (feature_df[team_col[:4]+'Wins'] * 2 + feature_df[team_col[:4]+'OTLs']) / ((feature_df[team_col[:4]+'Wins'] + feature_df[team_col[:4]+'OTLs'] + feature_df[team_col[:4]+'Losses']) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[cons.point_per_n_col.format(pre=team_col[:4], n=n)].isna()]
        feature_df_new[cons.point_per_n_col.format(pre=team_col[:4], n=n)] = (feature_df_new[team_col[:4]+'Wins'] * 2 + feature_df_new[team_col[:4]+'OTLs']) / ((feature_df_new[team_col[:4]+'Wins'] + feature_df_new[team_col[:4]+'OTLs'] + feature_df_new[team_col[:4]+'Losses']) * 2)
        feature_df.update(feature_df_new[cons.point_per_n_col.format(pre=team_col[:4], n=n)])
    feature_df.drop(columns=[team_col[:4]+'Wins', team_col[:4]+'OTLs', team_col[:4]+'Losses'], inplace=True)

    return feature_df


def prevN_result(data_df, backfill, target_col, team_col, n):
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    # local row column names
    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    result_col = '_result'

    # individual game dates, scores, regulation game mask
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    home_scores = pd.to_numeric(data_df[cons.home_team_score_col], errors='coerce')
    away_scores = pd.to_numeric(data_df[cons.away_team_score_col], errors='coerce')
    reg_mask = data_df[cons.last_period_col] == 'REG'

    # based on game result, create a dataframe mask
    if 'Wins' in target_col:
        home_result = (home_scores > away_scores)
        away_result = (away_scores > home_scores)
    elif 'Losses' in target_col:
        home_result = (home_scores < away_scores) & reg_mask
        away_result = (away_scores < home_scores) & reg_mask
    elif 'OTL' in target_col:
        home_result = (home_scores < away_scores) & (~reg_mask)
        away_result = (away_scores < home_scores) & (~reg_mask)
    else:
        raise ValueError("target_col must contain one of: Wins, Losses, OTL")

    # Normalize to a team-centric table (one row per team per game) so home/away can be handled identically.
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        result_col: home_result.astype(np.int8)
    })
    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        result_col: away_result.astype(np.int8)
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])
    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # Store per-team cumulative results so range totals can be answered in O(1).
    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        results = group[result_col].to_numpy(dtype=np.int16)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0], np.cumsum(results, dtype=np.int32)))
        )

    # Prepare target rows as numpy arrays to keep the main loop lightweight.
    target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df_target), np.nan, dtype=np.float64)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, result_prefix = history
        # First prior game index on this date boundary; excludes the current game and future games.
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # Sliding window total: prefix[end] - prefix[start] gives results in the previous n games.
        start_idx = max(0, end_idx - n)
        target_vals[i] = result_prefix[end_idx] - result_prefix[start_idx]

    data_df_target[target_col] = target_vals

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def prevN_gpg(data_df, backfill, target_col, team_col, n, for_against):

    if for_against == 'for':
        score_col = cons.home_team_score_col if team_col == cons.home_team_name_col else cons.away_team_score_col
    elif for_against == 'against':
        score_col = cons.away_team_score_col if team_col == cons.home_team_name_col else cons.home_team_score_col
    else:
        raise ValueError("for_against must be either 'for' or 'against'")
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    score_key_col = '_score'

    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    score_vals = pd.to_numeric(data_df[score_col], errors='coerce')

    # Normalize to one team-game row per side so home/away can share the same lookup path.
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        score_key_col: score_vals
    })
    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        score_key_col: score_vals
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])
    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # Build per-team cumulative score history so each n-game sum is O(1) after bisect.
    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        scores = group[score_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(scores)))
        )

    # Pull target columns into arrays to reduce repeated dataframe indexing in the loop.
    target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df_target), np.nan, dtype=float)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, score_prefix = history
        # Locate boundary of games strictly before current game date.
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # Window total from prefix sums for the previous n games.
        start_idx = max(0, end_idx - n)
        target_vals[i] = score_prefix[end_idx] - score_prefix[start_idx]

    data_df_target[target_col] = target_vals

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def prevN_corsi(data_df, backfill, target_col, team_col, n):

    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    corsi_key_col = '_corsi'

    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    home_corsi = pd.to_numeric(data_df[cons.home_shot_og_col], errors='coerce') +\
        pd.to_numeric(data_df[cons.home_shot_miss_col], errors='coerce') +\
            pd.to_numeric(data_df[cons.home_shot_blk_col], errors='coerce')
    away_corsi = pd.to_numeric(data_df[cons.away_shot_og_col], errors='coerce') +\
        pd.to_numeric(data_df[cons.away_shot_miss_col], errors='coerce') +\
            pd.to_numeric(data_df[cons.away_shot_blk_col], errors='coerce')
    data_df['_home_corsi_per'] = home_corsi / (home_corsi + away_corsi)
    data_df['_away_corsi_per'] = away_corsi / (home_corsi + away_corsi)

    # calculate rolling corsi for home and away teams over the last n games played
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        corsi_key_col: data_df['_home_corsi_per']
    })

    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        corsi_key_col: data_df['_away_corsi_per']
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])

    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # Build per-team cumulative corsi history so each n-game average is O(1) after bisect.
    team_history = {}

    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        corsi_vals = group[corsi_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(corsi_vals)))
        )

    # Pull target columns into arrays to reduce repeated dataframe indexing in the loop.
    target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df_target), np.nan, dtype=float)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, corsi_prefix = history
        # Locate boundary of games strictly before current game date.
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # Window average from prefix sums for the previous n games.
        start_idx = max(0, end_idx - n)
        games_played = end_idx - start_idx
        target_vals[i] = (corsi_prefix[end_idx] - corsi_prefix[start_idx]) / games_played

    data_df_target[target_col] = target_vals

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def prevN_fenwick(data_df, target_col, backfill, team_col, n, data_df_target=None):
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    fenwick_key_col = '_fenwick'

    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    home_fenwick = pd.to_numeric(data_df[cons.home_shot_og_col], errors='coerce') +\
        pd.to_numeric(data_df[cons.home_shot_miss_col], errors='coerce')
    away_fenwick = pd.to_numeric(data_df[cons.away_shot_og_col], errors='coerce') +\
        pd.to_numeric(data_df[cons.away_shot_miss_col], errors='coerce')
    data_df['_home_fenwick_per'] = home_fenwick / (home_fenwick + away_fenwick)
    data_df['_away_fenwick_per'] = away_fenwick / (home_fenwick + away_fenwick)

    # calculate rolling fenwick for home and away teams over the last n games played
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        fenwick_key_col: data_df['_home_fenwick_per']
    })

    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        fenwick_key_col: data_df['_away_fenwick_per']
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])

    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # Build per-team cumulative fenwick history so each n-game average is O(1) after bisect.
    team_history = {}

    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        fenwick_vals = group[fenwick_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(fenwick_vals)))
        )

    # Pull target columns into arrays to reduce repeated dataframe indexing in the loop.
    target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df_target), np.nan, dtype=float)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, fenwick_prefix = history
        # Locate boundary of games strictly before current game date.
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # Window average from prefix sums for the previous n games.
        start_idx = max(0, end_idx - n)
        games_played = end_idx - start_idx
        target_vals[i] = (fenwick_prefix[end_idx] - fenwick_prefix[start_idx]) / games_played

    data_df_target[target_col] = target_vals

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def compute_elo_ratings(data_df, k=6, home_advantage=25, season_regress_factor=0.75, starting_elo=1500):

    data_df = data_df.sort_values(cons.starttime_est_col).reset_index(drop=True)
    
    teams = pd.unique(data_df[[cons.home_team_name_col, cons.away_team_name_col]].values.ravel())
    elo_ratings = {team: starting_elo for team in teams}
    
    current_season = None
    
    home_elo_pre = []
    away_elo_pre = []
    
    for idx, row in data_df.iterrows():

        season = row[cons.season_name_col]
        home_team = row[cons.home_team_name_col]
        away_team = row[cons.away_team_name_col]
        
        # Regress toward mean at the start of a new season
        if season != current_season:
            if current_season is not None:  # skip regression before the very first season
                for team in elo_ratings:
                    elo_ratings[team] = (
                        elo_ratings[team] * (1 - season_regress_factor)
                        + starting_elo * season_regress_factor
                    )
            current_season = season

        # Skip rows where the game outcome is not yet known (e.g., future games)
        if pd.isna(row[cons.home_team_win_col]):
            break
        
        home_rating = elo_ratings[home_team]
        away_rating = elo_ratings[away_team]
        
        # Record PRE-GAME ratings as features (critical: before this game's outcome updates them)
        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)
        
        # Expected outcome (with home-ice adjustment baked into the rating diff)
        expected_home = 1 / (1 + 10 ** (-(home_rating + home_advantage - away_rating) / 400))
        
        actual_home = row[cons.home_team_win_col]  # 1 if home won, 0 if away won
        
        # Update ratings AFTER recording pre-game values
        elo_ratings[home_team] = home_rating + k * (actual_home - expected_home)
        elo_ratings[away_team] = away_rating + k * ((1 - actual_home) - (1 - expected_home))
    
    data_df.loc[data_df[cons.home_team_win_col].notna(), cons.elo_rat_col.format(pre='home')] = home_elo_pre
    data_df.loc[data_df[cons.home_team_win_col].notna(), cons.elo_rat_col.format(pre='away')] = away_elo_pre
    data_df.loc[data_df[cons.home_team_win_col].notna(), cons.elo_rat_col.format(pre='rel')] = data_df[cons.elo_rat_col.format(pre='home')] - data_df[cons.elo_rat_col.format(pre='away')]

    # Get final Elo per team after all completed games
    final_elo = {team: elo_ratings[team] for team in elo_ratings}  # from the function's internal state — you may want to return this separately

    data_df.loc[data_df[cons.home_team_win_col].isna(), cons.elo_rat_col.format(pre='home')] = data_df[cons.home_team_name_col].map(final_elo)
    data_df.loc[data_df[cons.home_team_win_col].isna(), cons.elo_rat_col.format(pre='away')] = data_df[cons.away_team_name_col].map(final_elo)
    data_df.loc[data_df[cons.home_team_win_col].isna(), cons.elo_rat_col.format(pre='rel')] = data_df[cons.elo_rat_col.format(pre='home')] - data_df[cons.elo_rat_col.format(pre='away')]

    return data_df


def add_ppper(data_df, team_col, n):

    target_col = cons.pp_per_n_col.format(pre=team_col[:4], n=n)

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    goal_key_col = '_pp_goal'
    opp_key_col = '_pp_opp'

    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    home_pp_goals = pd.to_numeric(data_df[cons.home_pp_goal_col], errors='coerce')
    away_pp_goals = pd.to_numeric(data_df[cons.away_pp_goal_col], errors='coerce')
    home_penalties = pd.to_numeric(data_df[cons.home_penalty_col], errors='coerce')
    away_penalties = pd.to_numeric(data_df[cons.away_penalty_col], errors='coerce')

    # a team's power play opportunities come from penalties committed by their opponent
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        goal_key_col: home_pp_goals,
        opp_key_col: away_penalties
    })

    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        goal_key_col: away_pp_goals,
        opp_key_col: home_penalties
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])
    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # build per-team cumulative pp goal/opportunity history so each n-game percentage is O(1) after bisect
    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        goal_vals = group[goal_key_col].to_numpy(dtype=float)
        opp_vals = group[opp_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(goal_vals))),
            np.concatenate(([0.0], np.cumsum(opp_vals)))
        )

    target_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df[team_col].to_numpy()
    target_seasons = data_df[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df), np.nan, dtype=float)
    for i in range(len(data_df)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, goal_prefix, opp_prefix = history
        # locate boundary of games strictly before current game date
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # window totals from prefix sums for the previous n games
        start_idx = max(0, end_idx - n)
        opportunities = opp_prefix[end_idx] - opp_prefix[start_idx]
        if opportunities > 0:
            target_vals[i] = (goal_prefix[end_idx] - goal_prefix[start_idx]) / opportunities

    data_df[target_col] = target_vals

    return data_df


def add_pkper(data_df, team_col, n):

    target_col = cons.pk_per_n_col.format(pre=team_col[:4], n=n)

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    goal_against_key_col = '_pk_goal_against'
    opp_key_col = '_pk_opp'

    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    home_pp_goals = pd.to_numeric(data_df[cons.home_pp_goal_col], errors='coerce')
    away_pp_goals = pd.to_numeric(data_df[cons.away_pp_goal_col], errors='coerce')
    home_penalties = pd.to_numeric(data_df[cons.home_penalty_col], errors='coerce')
    away_penalties = pd.to_numeric(data_df[cons.away_penalty_col], errors='coerce')

    # a team's penalty kill opportunities come from penalties committed by themselves, allowing the opponent's PP goals against them
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        goal_against_key_col: away_pp_goals,
        opp_key_col: home_penalties
    })

    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: game_dates,
        goal_against_key_col: home_pp_goals,
        opp_key_col: away_penalties
    })

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])
    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    # build per-team cumulative pk goal-against/opportunity history so each n-game percentage is O(1) after bisect
    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        goal_against_vals = group[goal_against_key_col].to_numpy(dtype=float)
        opp_vals = group[opp_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(goal_against_vals))),
            np.concatenate(([0.0], np.cumsum(opp_vals)))
        )

    target_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df[team_col].to_numpy()
    target_seasons = data_df[cons.season_name_col].to_numpy()

    target_vals = np.full(len(data_df), np.nan, dtype=float)
    for i in range(len(data_df)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, goal_against_prefix, opp_prefix = history
        # locate boundary of games strictly before current game date
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # window totals from prefix sums for the previous n games
        start_idx = max(0, end_idx - n)
        opportunities = opp_prefix[end_idx] - opp_prefix[start_idx]
        if opportunities > 0:
            target_vals[i] = 1 - ((goal_against_prefix[end_idx] - goal_against_prefix[start_idx]) / opportunities)

    data_df[target_col] = target_vals

    return data_df