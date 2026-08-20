import bisect

import numpy as np
import pandas as pd
import constants as cons

from utils.file_utils import csvLoad, csvSave
from schedule import load_sched_df_features


def sched_features_update(data_df_in=pd.DataFrame, verbose=False):

    if data_df_in.empty:
        data_df = load_sched_df_features()
    else:
        data_df = data_df_in[cons.base_feature_cols]

    # add schedule-based features
    sched_features = [cons.game_time_secs_est_col, cons.day_of_week_col, cons.reg_game_num_perc_col, cons.playoff_game_num_col,
                      cons.days_rest_col, cons.is_outdoor_venue_col, cons.road_trip_seq_col, cons.travel_dist_n_days_col,
                      cons.games_played_n_days_col, cons.crossed_tz_n_days_col, cons.is_home_opener_col, cons.rival_match_col,
                      cons.market_intensity_col, cons.is_ret_home_trap_col, cons.is_venue_alt_shock_col, cons.playoff_series_score_col]

    # features to add in the future
    future_features = ['isMajorCeremonyNight']

    for feature in sched_features:

        if verbose: print(f'\tAdding {feature}...')

        # add the game time in EST timezone for each game
        if feature == cons.game_time_secs_est_col:
            data_df = game_time_seconds_est(data_df)
            continue

        # add the day of the week the game is played for each game
        if feature == cons.day_of_week_col:
            data_df = day_of_week(data_df)
            continue
    
        # add the game number percentage for each team in each regular season
        if feature == cons.reg_game_num_perc_col:
            data_df = reg_game_num_perc(data_df, team_col=cons.home_team_name_col)
            data_df = reg_game_num_perc(data_df, team_col=cons.away_team_name_col)
            continue

        # add the game number for each team in each playoff season
        if feature == cons.playoff_game_num_col:
            data_df = playoff_game_num(data_df, team_col=cons.home_team_name_col)
            data_df = playoff_game_num(data_df, team_col=cons.away_team_name_col)
            continue

        # add the number of rest days for each team prior to each game in each season
        if feature == cons.days_rest_col:
            data_df = days_rest(data_df, team_col=cons.home_team_name_col)
            data_df = days_rest(data_df, team_col=cons.away_team_name_col)

            # create relational feature, drop individual features
            home_days_rest = cons.days_rest_col.format(pre='home')
            away_days_rest = cons.days_rest_col.format(pre='away')
            data_df[cons.days_rest_col.format(pre='rel')] = data_df[home_days_rest] - data_df[away_days_rest]
            data_df.drop(columns=[home_days_rest, away_days_rest], inplace=True)
            continue

        # add a binary feature indicating whether the game is played in an outdoor venue
        if feature == cons.is_outdoor_venue_col:
            data_df = is_outdoor_venue(data_df)
            continue

        # add a feature indicating the sequence of games played on the road for the away team
        if feature == cons.road_trip_seq_col:
            data_df = road_trip_sequence(data_df)
            continue

        # add a feature that counts the number of games played in the last N days for each team in each season
        if feature == cons.games_played_n_days_col:
            for window in cons.sched_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')
                data_df = games_played_last_n_days(data_df, team_col=cons.home_team_name_col, n=window)
                data_df = games_played_last_n_days(data_df, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_games_played_col = cons.games_played_n_days_col.format(pre='home', n=window)
                away_games_played_col = cons.games_played_n_days_col.format(pre='away', n=window)
                data_df[cons.games_played_n_days_col.format(pre='rel', n=window)] = data_df[home_games_played_col] - data_df[away_games_played_col]
                data_df.drop(columns=[home_games_played_col, away_games_played_col], inplace=True)
            continue

        # add a feature that calculates the travel distance for each team in the last N days for each season
        if feature == cons.travel_dist_n_days_col:
            for window in cons.sched_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')
                data_df = travel_distance_last_n_days(data_df, team_col=cons.home_team_name_col, n=window)
                data_df = travel_distance_last_n_days(data_df, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_travel_dist_col = cons.travel_dist_n_days_col.format(pre='home', n=window)
                away_travel_dist_col = cons.travel_dist_n_days_col.format(pre='away', n=window)
                data_df[cons.travel_dist_n_days_col.format(pre='rel', n=window)] = data_df[home_travel_dist_col] - data_df[away_travel_dist_col]
                data_df.drop(columns=[home_travel_dist_col, away_travel_dist_col], inplace=True)
            continue

        # add a feature that counts the number of time zones crossed for each team in the last N days for each season
        if feature == cons.crossed_tz_n_days_col:
            for window in cons.sched_feat_windows:
                if verbose: print(f'\t\tProcessing window: {window}')
                data_df = time_zones_crossed_last_n_days(data_df, team_col=cons.home_team_name_col, n=window)
                data_df = time_zones_crossed_last_n_days(data_df, team_col=cons.away_team_name_col, n=window)

                # create relational feature, drop individual features
                home_tz_crossed_col = cons.crossed_tz_n_days_col.format(pre='home', n=window)
                away_tz_crossed_col = cons.crossed_tz_n_days_col.format(pre='away', n=window)
                data_df[cons.crossed_tz_n_days_col.format(pre='rel', n=window)] = data_df[home_tz_crossed_col] - data_df[away_tz_crossed_col]
                data_df.drop(columns=[home_tz_crossed_col, away_tz_crossed_col], inplace=True)
            continue

        # add a feature that indicates the timezone of the venue for each game
        if feature == cons.is_home_opener_col:
            data_df = is_home_opener(data_df)
            continue

        # add a feature that indicates if the matchup is inter-divisional or inter-conference
        if feature == cons.rival_match_col:
            data_df = rival_match(data_df)
            continue

        # add a feature that indicates the market intensity of the home team
        if feature == cons.market_intensity_col:
            data_df = market_intensity(data_df)
            continue

        # add a feature that indicates if the home team is a returning home team after a road trip
        if feature == cons.is_ret_home_trap_col:
            data_df = is_return_home_after_road_trip(data_df)
            continue

        # add a feature that indicates if the venue is above 4,000ft elevation
        if feature == cons.is_venue_alt_shock_col:
            data_df = is_venue_altitude_shock(data_df)
            continue

        # add a feature that indicates the playoff series score for each team in each playoff season
        if feature == cons.playoff_series_score_col:
            data_df = playoff_series_score(data_df)
            continue

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            if verbose: print(f'Writing schedule features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.sched_features_folder, cons.sched_features_filename.format(season=season))

    return data_df, sched_features


def game_time_seconds_est(data_df):
    """
    Adds a new column to the input DataFrame that represents the game start time in EST timezone in
    seconds since midnight.
    """

    # the start time of the game in EST timezone
    data_df[cons.game_time_secs_est_col] = (data_df[cons.starttime_est_col] - data_df[cons.starttime_est_col].dt.normalize()).dt.total_seconds().astype(int)
    
    return data_df


def day_of_week(data_df):
    """
    Adds a new column to the input DataFrame that represents the day of the week the game is played.
    The day of the week is represented as an integer (0=Monday, 1=Tuesday, ..., 6=Sunday).
    """

    # the day of the week the game is played
    data_df[cons.day_of_week_col] = data_df[cons.starttime_est_col].dt.dayofweek.astype(int)
    
    return data_df


def reg_game_num_perc(data_df, team_col):
    """
    Adds a new column to the input DataFrame that represents how far into the regular season (as a
    percentage, 0 to 1) each game falls for the team, counting both home and away games in chronological
    order. Only gameType==2 (regular season) games count toward the total; gameType==3 (playoff) games
    are always given a value of 1.0.
    """

    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'
    game_type_col_name = '_game_type'
    game_num_col_name = '_game_num'
    game_num_perc_col_name = '_game_num_perc'

    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    game_types = data_df[cons.game_type_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so each team's games can be counted together.
    all_team_games = pd.DataFrame({
        row_id_col: np.concatenate([row_ids, row_ids]),
        team_col_name: np.concatenate([home_teams, away_teams]),
        season_col_name: np.concatenate([seasons, seasons]),
        game_date_col_name: np.concatenate([game_dates, game_dates]),
        game_type_col_name: np.concatenate([game_types, game_types]),
        source_col_name: np.concatenate([
            np.full(n_rows, cons.home_team_name_col, dtype=object),
            np.full(n_rows, cons.away_team_name_col, dtype=object)
        ])
    })

    # Within each team-season timeline, number the regular season games in chronological order.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    is_regular_season = all_team_games[game_type_col_name] == 2
    regular_season_games = all_team_games.loc[is_regular_season].copy()
    regular_season_group = regular_season_games.groupby([team_col_name, season_col_name], sort=False)
    regular_season_games[game_num_col_name] = regular_season_group.cumcount() + 1
    all_team_games.loc[is_regular_season, game_num_col_name] = regular_season_games[game_num_col_name].to_numpy()

    # Express the game number as a percentage of the team's total regular season games; playoff games get 1.0.
    all_team_games[game_num_perc_col_name] = 1.0
    all_team_games.loc[is_regular_season, game_num_perc_col_name] = (
        regular_season_games[game_num_col_name]
        / regular_season_group[game_num_col_name].transform('size')
    ).to_numpy()

    # Select either home-side or away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, game_num_perc_col_name]
    ].set_index(row_id_col)[game_num_perc_col_name]

    target_col = cons.reg_game_num_perc_col.format(team=team_col[:4])
    data_df[target_col] = target_series.reindex(row_ids).to_numpy()

    return data_df


def playoff_game_num(data_df, team_col):
    """
    Adds a new column to the input DataFrame that represents the running count of playoff games played by
    the team that season, counting both home and away games in chronological order. The calculation is
    done separately for each team within each season. Set to 0 for all non-playoff games (gameType != 3).
    """

    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'
    game_type_col_name = '_game_type'
    game_num_col_name = '_game_num'

    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    game_types = data_df[cons.game_type_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so each team's games can be counted together.
    all_team_games = pd.DataFrame({
        row_id_col: np.concatenate([row_ids, row_ids]),
        team_col_name: np.concatenate([home_teams, away_teams]),
        season_col_name: np.concatenate([seasons, seasons]),
        game_date_col_name: np.concatenate([game_dates, game_dates]),
        game_type_col_name: np.concatenate([game_types, game_types]),
        source_col_name: np.concatenate([
            np.full(n_rows, cons.home_team_name_col, dtype=object),
            np.full(n_rows, cons.away_team_name_col, dtype=object)
        ])
    })

    # Within each team-season timeline, number the playoff games in chronological order.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    is_playoffs = all_team_games[game_type_col_name] == 3
    playoff_group = all_team_games.loc[is_playoffs].groupby([team_col_name, season_col_name], sort=False)

    # Regular season games get 0; playoff games get the running count of playoff games played so far.
    all_team_games[game_num_col_name] = 0
    all_team_games.loc[is_playoffs, game_num_col_name] = playoff_group.cumcount().to_numpy() + 1

    # Select either home-side or away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, game_num_col_name]
    ].set_index(row_id_col)[game_num_col_name]

    target_col = cons.playoff_game_num_col.format(team=team_col[:4])
    data_df[target_col] = target_series.reindex(row_ids).to_numpy()

    return data_df
    

def days_rest(data_df, team_col):
    """
    Adds a new column to the input DataFrame that represents the number of days since the last game for each team.
    The calculation is done separately for each team within each season.
    """

    # Local helper column names used for temporary reshaping.
    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    days_col_name = '_days_since_last_game'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'

    # Materialize source columns once as numpy arrays for faster construction.
    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so diff() can be computed uniformly.
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

    # Within each team-season timeline, compute days since that team's previous game.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    all_team_games[days_col_name] = all_team_games.groupby([team_col_name, season_col_name], sort=False)[game_date_col_name].diff().dt.days

    # Select either home-side or away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, days_col_name]
    ].set_index(row_id_col)[days_col_name]

    target_col = cons.days_rest_col.format(pre=team_col[:4])
    data_df[target_col] = target_series.reindex(row_ids).fillna(5)

    return data_df


def is_outdoor_venue(data_df):
    """
    Adds a new column to the input DataFrame that indicates whether the game is played in an outdoor venue.
    The value is 1 if the game is outdoors, and 0 otherwise.
    """

    # Determine if the venue is outdoors based on the venue name or other criteria.
    data_df[cons.is_outdoor_venue_col] = data_df[cons.venue_col].apply(lambda x: 1 if x in cons.outdoor_venues else 0)

    return data_df


def road_trip_sequence(data_df):
    """
    Adds a new column to the input DataFrame that indicates the sequence of games played on the road for each team.
    The value is an integer representing the number of consecutive road games played by the team.
    """

    # Local helper column names used for temporary reshaping.
    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'

    # Materialize source columns once as numpy arrays for faster construction.
    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so diff() can be computed uniformly.
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

    # Within each team-season timeline, compute the sequence of road games.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)

    # Consecutive away appearances reset to 0 whenever a home game occurs; group by the cumulative count of
    # home games so each cumcount() within a group directly gives the road trip sequence number.
    is_away = all_team_games[source_col_name] == cons.away_team_name_col
    reset_group = (~is_away).groupby(
        [all_team_games[team_col_name], all_team_games[season_col_name]]
    ).cumsum()
    seq = all_team_games.groupby([team_col_name, season_col_name, reset_group]).cumcount()
    all_team_games[cons.road_trip_seq_col] = np.where(is_away, seq, 0)

    # Select away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == cons.away_team_name_col,
        [row_id_col, cons.road_trip_seq_col]
    ].set_index(row_id_col)[cons.road_trip_seq_col]

    data_df[cons.road_trip_seq_col] = target_series.reindex(row_ids).to_numpy()

    return data_df


def games_played_last_n_days(data_df, team_col, n):
    """
    Adds a new column to the input DataFrame that counts the number of games played in the last N days for each team.
    The calculation is done separately for each team within each season.
    """

    # Local helper column names used for temporary reshaping.
    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    games_played_col_name = f'_games_played_last_{n}_days'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'

    # Materialize source columns once as numpy arrays for faster construction.
    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so rolling count can be computed uniformly.
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

    # Within each team-season timeline, compute the rolling count of games played in the last N days.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)

    # Time-based rolling window with closed='left' counts prior games in [current - n days, current), excluding
    # the current game itself, matching the original >= start_date and < current_date semantics.
    all_team_games.set_index(game_date_col_name, inplace=True)
    all_team_games[games_played_col_name] = (
        all_team_games
        .groupby([team_col_name, season_col_name], sort=False)[row_id_col]
        .rolling(f'{n}D', closed='left')
        .count()
        .to_numpy()
    )
    all_team_games.reset_index(inplace=True)

    # Select either home-side or away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, games_played_col_name]
    ].set_index(row_id_col)[games_played_col_name]

    target_col = cons.games_played_n_days_col.format(pre=team_col[:4], n=n)
    data_df[target_col] = target_series.reindex(row_ids).fillna(0).to_numpy().astype(int)

    return data_df


def travel_distance_last_n_days(data_df, team_col, n, backfill=True):

    # load the geolocation file for venues and merge with the feature dataframe
    geoloc_df = csvLoad(cons.util_data_folder, cons.venue_geoloc_filename)
    data_df = data_df.merge(geoloc_df, how='left', left_on=cons.venue_col, right_on=cons.venue_col)

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

    # Expand to team-centric rows so each team-season has one chronological venue timeline.
    all_game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
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

    # Cache date and venue arrays per team-season for fast repeated lookups.
    team_history = {}
    for key, group in team_games.groupby([team_key_col, season_key_col], sort=False):
        team_history[key] = (
            group[date_col].tolist(),
            group[venue_lat_col].to_numpy(dtype=float),
            group[venue_long_col].to_numpy(dtype=float)
        )

    # Vectorized great-circle distance for one current venue against multiple prior venues.
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

    # Pull target fields once so per-row logic avoids repeated dataframe access.
    target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
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
        # Use prior games only (strictly earlier dates) via lower-bound binary search.
        end_idx = bisect.bisect_left(hist_dates, game_date)
        if end_idx == 0:
            continue

        # Restrict to games within the last n days and ignore historical rows without coordinates.
        start_date = game_date - pd.Timedelta(days=n)
        start_idx = bisect.bisect_left(hist_dates, start_date)
        prev_lats = hist_lats[start_idx:end_idx]
        prev_longs = hist_longs[start_idx:end_idx]
        valid_prev = (~np.isnan(prev_lats)) & (~np.isnan(prev_longs))
        if not valid_prev.any():
            continue

        # Mean distance summarizes recent travel burden for this team before the current game.
        dists = _haversine_vector_km(target_lats[i], target_longs[i], prev_lats[valid_prev], prev_longs[valid_prev])
        if dists.size:
            target_vals[i] = float(dists.mean())

    target_col = cons.travel_dist_n_days_col.format(pre=team_col[:4], n=n)
    data_df_target[target_col] = target_vals
    data_df_target[target_col] = data_df_target[target_col].where(pd.notna(data_df_target[target_col]), 0)

    if backfill:
        data_df = data_df_target
    else:
        data_df = pd.concat([data_df.loc[data_df[cons.last_period_col].notna()], data_df_target], ignore_index=True)

    data_df.drop(columns=[venue_lat_col, venue_long_col], inplace=True)
    
    return data_df


def _venue_utc_offset_hours(data_df):
    """
    Returns a Series of the UTC offset (in hours) of each game's venue timezone at the game's start time.
    """

    game_est = data_df[cons.starttime_est_col].dt.tz_localize(cons.est_tz)
    tz_names = data_df[cons.venue_timezone_col]

    offsets = pd.Series(np.nan, index=data_df.index, dtype=float)
    for tz_name, idx in tz_names.groupby(tz_names).groups.items():
        if pd.isna(tz_name):
            continue
        localized = game_est.loc[idx].dt.tz_convert(tz_name)
        offsets.loc[idx] = localized.apply(lambda ts: ts.utcoffset().total_seconds() / 3600.0)

    return offsets


def time_zones_crossed_last_n_days(data_df, team_col, n):
    """
    Adds a new column to the input DataFrame that represents the total number of time zones crossed by each
    team over its last N games (including the travel into the current game). The calculation is done
    separately for each team within each season.
    """

    # Local helper column names used for temporary reshaping.
    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    tz_offset_col_name = '_tz_offset_hours'
    tz_crossed_col_name = '_tz_crossed'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'

    # Materialize source columns once as numpy arrays for faster construction.
    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()
    tz_offsets = _venue_utc_offset_hours(data_df).to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so diff() can be computed uniformly.
    all_team_games = pd.DataFrame({
        row_id_col: np.concatenate([row_ids, row_ids]),
        team_col_name: np.concatenate([home_teams, away_teams]),
        season_col_name: np.concatenate([seasons, seasons]),
        game_date_col_name: np.concatenate([game_dates, game_dates]),
        tz_offset_col_name: np.concatenate([tz_offsets, tz_offsets]),
        source_col_name: np.concatenate([
            np.full(n_rows, cons.home_team_name_col, dtype=object),
            np.full(n_rows, cons.away_team_name_col, dtype=object)
        ])
    })

    # Within each team-season timeline, compute the time zones crossed traveling into each game.
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    games_grouped = all_team_games.groupby([team_col_name, season_col_name], sort=False)
    tz_crossed_per_game = games_grouped[tz_offset_col_name].diff().abs()

    # Sum the zones crossed over the trailing N games (including the travel into the current game).
    all_team_games[tz_crossed_col_name] = tz_crossed_per_game.groupby(
        [all_team_games[team_col_name], all_team_games[season_col_name]], sort=False
    ).transform(lambda s: s.rolling(window=n, min_periods=1).sum())

    # Select either home-side or away-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == team_col,
        [row_id_col, tz_crossed_col_name]
    ].set_index(row_id_col)[tz_crossed_col_name]

    target_col = cons.crossed_tz_n_days_col.format(pre=team_col[:4], n=n)
    data_df[target_col] = target_series.reindex(row_ids).fillna(0).astype(int)

    return data_df


def is_home_opener(data_df):
    """
    Adds a new column to the input DataFrame that indicates whether the game is the home opener for the home team.
    The value is 1 if the game is the home opener, and 0 otherwise.
    """

    # Determine if the game is the home opener based on the game number for the home team.
    data_df[cons.is_home_opener_col] = (data_df[cons.reg_game_num_perc_col.format(team='home')] < 0.02).astype(int)

    return data_df


def rival_match(data_df):
    """
    Adds a new column to the input DataFrame that indicates if the matchup is between two teams in the
    same division (2), in the same conference (1), or neither (0).
    """

    home_divisions = data_df[cons.home_team_name_col].map(lambda team: (cons.team_info.get(team) or cons.defunct_team_info.get(team))['division'])
    away_divisions = data_df[cons.away_team_name_col].map(lambda team: (cons.team_info.get(team) or cons.defunct_team_info.get(team))['division'])
    home_conferences = data_df[cons.home_team_name_col].map(lambda team: (cons.team_info.get(team) or cons.defunct_team_info.get(team))['conference'])
    away_conferences = data_df[cons.away_team_name_col].map(lambda team: (cons.team_info.get(team) or cons.defunct_team_info.get(team))['conference'])

    data_df[cons.rival_match_col] = np.where(
        home_divisions == away_divisions, 2,
        np.where(home_conferences == away_conferences, 1, 0)
    )

    return data_df


def market_intensity(data_df):
    """
    Adds a new column to the input DataFrame that indicates the market intensity of the home team.
    The value is based on the market size and fan engagement, generated by Google Gemini
    """

    data_df[cons.market_intensity_col] = data_df[cons.home_team_name_col].map(lambda team: cons.market_intensity_map[team])

    return data_df


def is_return_home_after_road_trip(data_df):
    """
    Adds a new column to the input DataFrame that indicates if the home team is returning home after a road trip of 3 or more games.
    The value is 1 if the game is the first home game after a road trip of 3 or more games, and 0 otherwise.
    """

    # Local helper column names used for temporary reshaping.
    row_id_col = '_row_id'
    team_col_name = '_team'
    source_col_name = '_source_col'
    season_col_name = '_season'
    game_date_col_name = '_game_date_dt'
    road_trip_seq_col_name = '_road_trip_seq'
    is_ret_home_col_name = '_is_ret_home'

    # Materialize source columns once as numpy arrays for faster construction.
    row_ids = data_df.index.to_numpy()
    game_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').to_numpy()
    seasons = data_df[cons.season_name_col].to_numpy()
    home_teams = data_df[cons.home_team_name_col].to_numpy()
    away_teams = data_df[cons.away_team_name_col].to_numpy()

    n_rows = len(data_df)
    # Build one row per team appearance (home + away) so each team's road trip streak can be tracked.
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

    # Within each team-season timeline, recompute the road trip sequence (consecutive away games).
    all_team_games.sort_values(by=[team_col_name, season_col_name, game_date_col_name, row_id_col], inplace=True)
    is_away = all_team_games[source_col_name] == cons.away_team_name_col
    reset_group = (~is_away).groupby(
        [all_team_games[team_col_name], all_team_games[season_col_name]]
    ).cumsum()
    seq = all_team_games.groupby([team_col_name, season_col_name, reset_group]).cumcount()
    all_team_games[road_trip_seq_col_name] = np.where(is_away, seq, 0)

    # A home game qualifies if the team's immediately preceding game ended a road trip of 3+ games.
    prev_seq = all_team_games.groupby([team_col_name, season_col_name], sort=False)[road_trip_seq_col_name].shift(1)
    all_team_games[is_ret_home_col_name] = ((~is_away) & (prev_seq >= 3)).astype(int)

    # Select the home-side values and align back to the original dataframe index.
    target_series = all_team_games.loc[
        all_team_games[source_col_name] == cons.home_team_name_col,
        [row_id_col, is_ret_home_col_name]
    ].set_index(row_id_col)[is_ret_home_col_name]

    data_df[cons.is_ret_home_trap_col] = target_series.reindex(row_ids).fillna(0).to_numpy().astype(int)

    return data_df


def is_venue_altitude_shock(data_df):
    """
    Adds a new column to the input DataFrame that indicates if the venue is above 4,000ft elevation.
    The value is 1 if the venue is above 4,000ft, and 0 otherwise.
    """
    high_altitude_venues = cons.high_altitude_venues
    data_df[cons.is_venue_alt_shock_col] = data_df[cons.venue_col].isin(high_altitude_venues).astype(int)

    return data_df


def playoff_series_score(data_df, backfill=True):

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

    # Pull playoff games once; everything below works off this filtered subset.
    all_dates = pd.to_datetime(data_df[cons.starttime_est_col], errors='coerce').dt.date
    playoff_mask = data_df[cons.game_type_col] == 3

    playoff_games = data_df.loc[playoff_mask, [
        cons.home_team_name_col,
        cons.away_team_name_col,
        cons.season_name_col,
        cons.win_prob_col.format(team='home'),
        cons.win_prob_col.format(team='away')
    ]].copy()

    if not playoff_games.empty:
        playoff_games[row_id_col] = data_df.loc[playoff_mask].index.to_numpy()
        playoff_games[date_col] = all_dates.loc[playoff_mask].to_numpy()
        playoff_games = playoff_games.dropna(subset=[date_col])

        # Canonicalize matchup order so A vs B and B vs A map to the same series key.
        home_arr = playoff_games[cons.home_team_name_col].to_numpy()
        away_arr = playoff_games[cons.away_team_name_col].to_numpy()
        playoff_games[season_key_col] = playoff_games[cons.season_name_col].to_numpy()
        playoff_games[team_low_col] = np.where(home_arr <= away_arr, home_arr, away_arr)
        playoff_games[team_high_col] = np.where(home_arr <= away_arr, away_arr, home_arr)

        home_scores = pd.to_numeric(playoff_games[cons.win_prob_col.format(team='home')], errors='coerce').to_numpy()
        away_scores = pd.to_numeric(playoff_games[cons.win_prob_col.format(team='away')], errors='coerce').to_numpy()
        playoff_games[winner_col] = np.where(
            home_scores > away_scores,
            home_arr,
            np.where(away_scores > home_scores, away_arr, None)
        )

        playoff_games.sort_values(
            by=[season_key_col, team_low_col, team_high_col, date_col, row_id_col],
            inplace=True
        )

        # Build cumulative wins per series so "wins before game i" can be read in O(1).
        series_history = {}
        for key, group in playoff_games.groupby([season_key_col, team_low_col, team_high_col], sort=False):
            team_low = key[1]
            team_high = key[2]
            winners = group[winner_col].to_numpy(dtype=object)
            low_prefix = np.concatenate(([0], np.cumsum((winners == team_low).astype(np.int16), dtype=np.int32)))
            high_prefix = np.concatenate(([0], np.cumsum((winners == team_high).astype(np.int16), dtype=np.int32)))
            series_history[key] = (group[date_col].tolist(), team_low, low_prefix, high_prefix)

        # Extract target fields once to keep the lookup loop lightweight.
        target_dates = pd.to_datetime(data_df_target[cons.starttime_est_col], errors='coerce').dt.date.to_numpy()
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
            # Count only games strictly before the current one.
            end_idx = bisect.bisect_left(hist_dates, game_date)
            if end_idx == 0:
                continue

            # Map canonical low/high prefixes back to the home/away perspective for this row.
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

    data_df[cons.home_team_series_score_col] = data_df[cons.home_team_series_score_col].fillna(0).astype(int)
    data_df[cons.away_team_series_score_col] = data_df[cons.away_team_series_score_col].fillna(0).astype(int)

    return data_df