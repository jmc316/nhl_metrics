import bisect

import numpy as np
import pandas as pd
import constants as cons

from file_utils import csvLoad


def dependent_feature_add(feature_df, backfill=True, debug=True):

    # add the home team's points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.home_team_name_col)

    # add the away team's points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.away_team_name_col)

    # add the home team's previous 3 games points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.home_team_name_col, 3)

    # add the away team's previous 3 games points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.away_team_name_col, 3)

    # add the home team's previous 7 games points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.home_team_name_col, 7)

    # add the away team's previous 7 games points percentage
    feature_df = points_percentage_feature_add(feature_df, debug, backfill, cons.away_team_name_col, 7)

    # calculate goals for in the previous 3 games for the home team in all matchups
    if debug: print('\t... [feature_creation] home team previous 3 goals for ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col, 3, 'for')

    # calculate goals for in the previous 3 games for the away team in all matchups
    if debug: print('\t... [feature_creation] away team previous 3 goals for ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col, 3, 'for')

    # calculate goals for in the previous 7 games for the home team in all matchups
    if debug: print('\t... [feature_creation] home team previous 7 goals for ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.home_team_prev_n_goals_for_col, cons.home_team_name_col, 7, 'for')

    # calculate goals for in the previous 7 games for the away team in all matchups
    if debug: print('\t... [feature_creation] away team previous 7 goals for ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.away_team_prev_n_goals_for_col, cons.away_team_name_col, 7, 'for')

    # calculate goals against in the previous 3 games for the home team in all matchups
    if debug: print('\t... [feature_creation] home team previous 3 goals against ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.home_team_prev_n_goals_against_col, cons.home_team_name_col, 3, 'against')

    # calculate goals against in the previous 3 games for the away team in all matchups
    if debug: print('\t... [feature_creation] away team previous 3 goals against ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.away_team_prev_n_goals_against_col, cons.away_team_name_col, 3, 'against')

    # calculate goals against in the previous 7 games for the home team in all matchups
    if debug: print('\t... [feature_creation] home team previous 7 goals against ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.home_team_prev_n_goals_against_col, cons.home_team_name_col, 7, 'against')

    # calculate goals against in the previous 7 games for the away team in all matchups
    if debug: print('\t... [feature_creation] away team previous 7 goals against ...')
    feature_df = prevN_gpg(feature_df, backfill, cons.away_team_prev_n_goals_against_col, cons.away_team_name_col, 7, 'against')

    # calculate days since last played game for the home team in all matchups
    if debug: print('\t... [feature_creation] home team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.home_team_days_since_last_game_col, cons.home_team_name_col)

    # calculate days since last played game for the away team in all matchups
    if debug: print('\t... [feature_creation] away team days since last game ...')
    feature_df = days_since_last_played(feature_df, cons.away_team_days_since_last_game_col, cons.away_team_name_col)

    # load the geolocation file for venues and merge with the feature dataframe
    geoloc_df = csvLoad(cons.util_data_folder, cons.venue_geoloc_filename)
    feature_df = feature_df.merge(geoloc_df, how='left', left_on=cons.venue_col, right_on=cons.venue_col)

    # calculate the haversine distance for the previous 7 days for the home team in all matchups
    if debug: print('\t\t... [feature_creation] home team travel distance previous 7 days ...')
    feature_df = hav_dist_Ndays(feature_df, cons.home_team_travel_distance_7days_col, cons.home_team_name_col, backfill, 7)

    # calculate the haversine distance for the previous 7 days for the away team in all matchups
    if debug: print('\t\t... [feature_creation] away team travel distance previous 7 days ...')
    feature_df = hav_dist_Ndays(feature_df, cons.away_team_travel_distance_7days_col, cons.away_team_name_col, backfill, 7)

    # calculate the series score for all playoff games
    if debug: print(f'\t\t... [feature_creation] playoff series scores ...')
    feature_df = playoff_series_score(feature_df, backfill)

    feature_df.drop(columns=[cons.venue_col+'_lat', cons.venue_col+'_long'], inplace=True)

    return feature_df


def datetime_feature_add(feature_df):

    # convert the 'startTimeUTC' column to datetime and extract the relevant features
    feature_df[cons.starttime_utc_col] = pd.to_datetime(feature_df[cons.starttime_utc_col]).dt.tz_convert('US/Eastern')
    feature_df[cons.game_date_col] = feature_df[cons.starttime_utc_col].dt.date
    feature_df[cons.game_time_col] = feature_df[cons.starttime_utc_col].dt.hour * 60 + feature_df[cons.starttime_utc_col].dt.minute
    feature_df.drop(columns=[cons.starttime_utc_col], inplace=True)

    return feature_df


def points_percentage_feature_add(feature_df, debug, backfill, team_col, n=None):

    # assign n to a large number to capture all games in the current season for the specified team
    if not n:
        n = cons.max_single_season_games
        n_str = 'season'
    else:
        n_str = 'previous ' + str(n) + ' games'

    # calculate the number of wins for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} win total ...')
    feature_df = prevN_result(feature_df, backfill, team_col+'Wins', team_col, n)

    # calculate the number of losses for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} loss total ...')
    feature_df = prevN_result(feature_df, backfill, team_col+'Losses', team_col, n)

    # calculate the number of OTLs for the home team in all matchups
    # if debug: print(f'\t\t... [sub_feature_creation] {team_col[:4]} team {n_str} OTL total ...')
    feature_df = prevN_result(feature_df, backfill, team_col+'OTLs', team_col, n)

    # calculate the points percentage of the home team in all matchups
    if debug: print(f'\t\t... [feature_creation] {team_col[:4]} team {n_str} points percentage ...')
    if team_col+'PointsPercentage' not in feature_df.columns:
        feature_df[team_col+'PointsPercentage'] = (feature_df[team_col+'Wins'] * 2 + feature_df[team_col+'OTLs']) / ((feature_df[team_col+'Wins'] + feature_df[team_col+'OTLs'] + feature_df[team_col+'Losses']) * 2)
    else:
        feature_df_new = feature_df.loc[feature_df[team_col+'PointsPercentage'].isna()]
        feature_df_new[team_col+'PointsPercentage'] = (feature_df_new[team_col+'Wins'] * 2 + feature_df_new[team_col+'OTLs']) / ((feature_df_new[team_col+'Wins'] + feature_df_new[team_col+'OTLs'] + feature_df_new[team_col+'Losses']) * 2)
        feature_df.update(feature_df_new[team_col+'PointsPercentage'])
    feature_df.drop(columns=[team_col+'Wins', team_col+'OTLs', team_col+'Losses'], inplace=True)

    return feature_df


def prevN_result(data_df, backfill, target_col, team_col, n):
    
    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    date_col = '_game_date'
    result_col = '_result'

    game_dates = pd.to_datetime(data_df[cons.game_date_col], errors='coerce').dt.date
    home_scores = pd.to_numeric(data_df[cons.home_team_score_col], errors='coerce')
    away_scores = pd.to_numeric(data_df[cons.away_team_score_col], errors='coerce')
    reg_mask = data_df[cons.last_period_col] == 'REG'

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

    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        results = group[result_col].to_numpy(dtype=np.int16)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0], np.cumsum(results, dtype=np.int32)))
        )

    target_dates = pd.to_datetime(data_df_target[cons.game_date_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.zeros(len(data_df_target), dtype=np.int16)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, result_prefix = history
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

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
    
    # initialize the target column with zeros
    target_col = target_col + str(n)
    
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

    game_dates = pd.to_datetime(data_df[cons.game_date_col], errors='coerce').dt.date
    score_vals = pd.to_numeric(data_df[score_col], errors='coerce').fillna(0.0)

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

    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        scores = group[score_key_col].to_numpy(dtype=float)
        team_history[key] = (
            group[date_col].tolist(),
            np.concatenate(([0.0], np.cumsum(scores)))
        )

    target_dates = pd.to_datetime(data_df_target[cons.game_date_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()

    target_vals = np.zeros(len(data_df_target), dtype=float)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, score_prefix = history
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        start_idx = max(0, end_idx - n)
        target_vals[i] = score_prefix[end_idx] - score_prefix[start_idx]

    data_df_target[target_col] = target_vals

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df


def days_since_last_played(data_df, target_col, team_col):

    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    days_col_name = '_days_since_last_game'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'

    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.game_date_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    all_team_games = pd.DataFrame({
        row_id_col: np.concatenate([row_ids, row_ids]),
        team_col_name: np.concatenate([home_teams, away_teams]),
        season_col_name: np.concatenate([seasons, seasons]),
        game_date_col_name: np.concatenate([game_dates, game_dates]),
        source_col_name: np.concatenate([
            np.full(n_rows, cons.home_team_name_col, dtype=object),
            np.full(n_rows, cons.away_team_name_col, dtype=object)
        ])
    })

    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    all_team_games[days_col_name] = all_team_games.groupby([team_col_name, season_col_name], sort=False)[game_date_col_name].diff().dt.days

    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, days_col_name]
    ].set_index(row_id_col)[days_col_name]

    data_df[target_col] = target_series.reindex(row_ids).to_numpy()

    return data_df


def hav_dist_Ndays(data_df, target_col, team_col, backfill, n):

    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    # Precompute date/venue history per team-season once and query it with binary search.
    date_col = '_game_date'
    row_id_col = '_row_id'
    team_key_col = '_team'
    season_key_col = '_season'
    venue_lat_col = cons.venue_col + '_lat'
    venue_long_col = cons.venue_col + '_long'

    all_game_dates = pd.to_datetime(data_df[cons.game_date_col], errors='coerce').dt.date
    home_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.home_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: all_game_dates,
        venue_lat_col: pd.to_numeric(data_df[venue_lat_col], errors='coerce'),
        venue_long_col: pd.to_numeric(data_df[venue_long_col], errors='coerce')
    })
    away_games = pd.DataFrame({
        row_id_col: data_df.index,
        team_key_col: data_df[cons.away_team_name_col],
        season_key_col: data_df[cons.season_name_col],
        date_col: all_game_dates,
        venue_lat_col: pd.to_numeric(data_df[venue_lat_col], errors='coerce'),
        venue_long_col: pd.to_numeric(data_df[venue_long_col], errors='coerce')
    })
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.dropna(subset=[date_col])
    team_games.sort_values(by=[team_key_col, season_key_col, date_col, row_id_col], inplace=True)

    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        team_history[key] = (
            group[date_col].tolist(),
            group[venue_lat_col].to_numpy(dtype=float),
            group[venue_long_col].to_numpy(dtype=float)
        )

    def _haversine_vector_km(cur_lat, cur_long, prev_lats, prev_longs):
        cur_lat_rad = np.radians(cur_lat)
        cur_long_rad = np.radians(cur_long)
        prev_lats_rad = np.radians(prev_lats)
        prev_longs_rad = np.radians(prev_longs)

        dlat = prev_lats_rad - cur_lat_rad
        dlong = prev_longs_rad - cur_long_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(cur_lat_rad) * np.cos(prev_lats_rad) * np.sin(dlong / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    target_dates = pd.to_datetime(data_df_target[cons.game_date_col], errors='coerce').dt.date.to_numpy()
    target_teams = data_df_target[team_col].to_numpy()
    target_seasons = data_df_target[cons.season_name_col].to_numpy()
    target_lats = pd.to_numeric(data_df_target[venue_lat_col], errors='coerce').to_numpy(dtype=float)
    target_longs = pd.to_numeric(data_df_target[venue_long_col], errors='coerce').to_numpy(dtype=float)

    target_vals = np.full(len(data_df_target), np.nan)
    for i in range(len(data_df_target)):
        game_date = target_dates[i]
        if pd.isna(game_date) or np.isnan(target_lats[i]) or np.isnan(target_longs[i]):
            continue

        history = team_history.get((target_teams[i], target_seasons[i]))
        if not history:
            continue

        hist_dates, hist_lats, hist_longs = history
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        start_idx = max(0, end_idx - n)
        prev_lats = hist_lats[start_idx:end_idx]
        prev_longs = hist_longs[start_idx:end_idx]
        valid_prev = (~np.isnan(prev_lats)) & (~np.isnan(prev_longs))
        if not valid_prev.any():
            continue

        dists = _haversine_vector_km(target_lats[i], target_longs[i], prev_lats[valid_prev], prev_longs[valid_prev])
        if dists.size:
            target_vals[i] = float(dists.mean())

    data_df_target[target_col] = target_vals
    data_df_target[target_col] = data_df_target[target_col].where(pd.notna(data_df_target[target_col]), None)

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)
    
    return data_df


def playoff_series_score(data_df, backfill):

    # create dataframe to loop through
    if backfill:
        data_df_target = data_df.loc[data_df[cons.game_type_col]==3].copy()
    else:
        data_df_target = data_df.loc[data_df[cons.last_period_col].isna()].copy()

    series_score_cols = [cons.home_team_series_score_col, cons.away_team_series_score_col]
    data_df_target[series_score_cols] = 0

    # Build playoff-series history once, then query prior-game counts with binary search.
    row_id_col = '_row_id'
    season_key_col = '_season'
    team_low_col = '_team_low'
    team_high_col = '_team_high'
    date_col = '_game_date'
    winner_col = '_winner'

    all_dates = pd.to_datetime(data_df[cons.game_date_col], errors='coerce').dt.date
    playoff_mask = data_df[cons.game_type_col] == 3

    playoff_games = data_df.loc[playoff_mask, [
        cons.home_team_name_col,
        cons.away_team_name_col,
        cons.season_name_col,
        cons.home_team_score_col,
        cons.away_team_score_col
    ]].copy()

    if not playoff_games.empty:
        playoff_games[row_id_col] = data_df.loc[playoff_mask].index.to_numpy()
        playoff_games[date_col] = all_dates.loc[playoff_mask].to_numpy()
        playoff_games = playoff_games.dropna(subset=[date_col])

        home_arr = playoff_games[cons.home_team_name_col].to_numpy()
        away_arr = playoff_games[cons.away_team_name_col].to_numpy()
        playoff_games[season_key_col] = playoff_games[cons.season_name_col].to_numpy()
        playoff_games[team_low_col] = np.where(home_arr <= away_arr, home_arr, away_arr)
        playoff_games[team_high_col] = np.where(home_arr <= away_arr, away_arr, home_arr)

        home_scores = pd.to_numeric(playoff_games[cons.home_team_score_col], errors='coerce').to_numpy()
        away_scores = pd.to_numeric(playoff_games[cons.away_team_score_col], errors='coerce').to_numpy()
        playoff_games[winner_col] = np.where(
            home_scores > away_scores,
            home_arr,
            np.where(away_scores > home_scores, away_arr, None)
        )

        playoff_games.sort_values(
            by=[season_key_col, team_low_col, team_high_col, date_col, row_id_col],
            inplace=True
        )

        series_history = {}
        for key, group in playoff_games.groupby([season_key_col, team_low_col, team_high_col], sort=False):
            team_low = key[1]
            team_high = key[2]
            winners = group[winner_col].to_numpy(dtype=object)
            low_prefix = np.concatenate(([0], np.cumsum((winners == team_low).astype(np.int16), dtype=np.int32)))
            high_prefix = np.concatenate(([0], np.cumsum((winners == team_high).astype(np.int16), dtype=np.int32)))
            series_history[key] = (group[date_col].tolist(), team_low, low_prefix, high_prefix)

        target_dates = pd.to_datetime(data_df_target[cons.game_date_col], errors='coerce').dt.date.to_numpy()
        target_seasons = data_df_target[cons.season_name_col].to_numpy()
        target_home = data_df_target[cons.home_team_name_col].to_numpy()
        target_away = data_df_target[cons.away_team_name_col].to_numpy()

        home_series_vals = np.zeros(len(data_df_target), dtype=np.int16)
        away_series_vals = np.zeros(len(data_df_target), dtype=np.int16)

        for i in range(len(data_df_target)):
            game_date = target_dates[i]
            if pd.isna(game_date):
                continue

            home_team = target_home[i]
            away_team = target_away[i]
            low_team = home_team if home_team <= away_team else away_team
            high_team = away_team if home_team <= away_team else home_team

            history = series_history.get((target_seasons[i], low_team, high_team))
            if not history:
                continue

            hist_dates, series_low_team, low_prefix, high_prefix = history
            end_idx = bisect.bisect_left(hist_dates, game_date)
            if end_idx == 0:
                continue

            if home_team == series_low_team:
                home_series_vals[i] = low_prefix[end_idx]
                away_series_vals[i] = high_prefix[end_idx]
            else:
                home_series_vals[i] = high_prefix[end_idx]
                away_series_vals[i] = low_prefix[end_idx]

        data_df_target[cons.home_team_series_score_col] = home_series_vals
        data_df_target[cons.away_team_series_score_col] = away_series_vals

    if backfill:
        data_df = pd.concat([data_df.loc[data_df[cons.game_type_col]!=3], data_df_target], ignore_index=True)
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    return data_df