"""Builds goalie-based features (save %, GAA, rest, recent start counts) for the model dataset."""

import numpy as np
import pandas as pd
import constants as cons

from schedule import load_sched_df_features
from utils.file_utils import csvSave, csvLoad

# per-filter (None=all situations, ev/pp/sh) source column names in goalie_df
_SAVE_STAT_COLS_BY_FILT = {
    None: ('tot_saves', 'tot_shots_against'),
    'ev': ('ev_saves', 'ev_shots_against'),
    'pp': ('pp_saves', 'pp_shots_against'),
    'sh': ('sh_saves', 'sh_shots_against'),
}
_GOALS_AGAINST_COL_BY_FILT = {
    None: 'tot_goals_against',
    'ev': 'ev_goals_against',
    'pp': 'pp_goals_against',
    'sh': 'sh_goals_against',
}


def goalie_features_update(data_df_in=pd.DataFrame, verbose=False):
    """Add all goalie features to data_df_in (or a freshly loaded feature set if empty).

    For each feature, home/away values are computed and then collapsed into a single
    home-minus-away relational column, since only the relative difference is used by the model.
    """

    if data_df_in.empty:
        data_df = load_sched_df_features(feat_set_label='goalie')
    else:
        data_df = data_df_in[cons.goalie_feature_cols]

    # add goalie-based features
    goalie_features = [cons.save_per_n_col, cons.gaa_n_col, cons.goalie_days_rest_col, cons.num_starts_n_col,
                       cons.num_season_starts_col]

    # features to add in the future
    future_features = ['mins_played_last']

    goalie_df = load_goalie_df()

    for feature in goalie_features:

        if verbose: print(f'\tAdding {feature}...')

        # add a feature that tracks save percentage over the last n games for each starting goalie
        if feature == cons.save_per_n_col:
            for filt in cons.goalie_feat_prefixes:
                if verbose: print(f'\t\tProcessing filter: {filt}')
                for window in cons.goalie_feat_windows:
                    if verbose: print(f'\t\tProcessing window: {window}')
                    data_df = save_per_n(data_df, goalie_df, team='home', window=window, filt=filt, verbose=verbose)
                    data_df = save_per_n(data_df, goalie_df, team='away', window=window, filt=filt, verbose=verbose)

                    # create relational feature, drop individual features
                    if not filt:
                        home_save_per = cons.save_per_n_col.format(pre='home', n=window)
                        away_save_per = cons.save_per_n_col.format(pre='away', n=window)
                        rel_save_per = cons.save_per_n_col.format(pre='rel', n=window)
                    else:
                        home_save_per = f'{filt}_' + cons.save_per_n_col.format(pre='home', n=window)
                        away_save_per = f'{filt}_' + cons.save_per_n_col.format(pre='away', n=window)
                        rel_save_per = f'{filt}_' + cons.save_per_n_col.format(pre='rel', n=window)
                    data_df[rel_save_per] = data_df[home_save_per] - data_df[away_save_per]
                    data_df.drop(columns=[home_save_per, away_save_per], inplace=True)

        # add a feature that tracks goals against average over the last n games for each starting goalie
        if feature == cons.gaa_n_col:
            for filt in cons.goalie_feat_prefixes:
                if verbose: print(f'\t\tProcessing filter: {filt}')
                for window in cons.goalie_feat_windows:
                    if verbose: print(f'\t\tProcessing window: {window}')
                    data_df = gaa_n(data_df, goalie_df, team='home', window=window, filt=filt, verbose=verbose)
                    data_df = gaa_n(data_df, goalie_df, team='away', window=window, filt=filt, verbose=verbose)

                    # create relational feature, drop individual features
                    if not filt:
                        home_gaa = cons.gaa_n_col.format(pre='home', n=window)
                        away_gaa = cons.gaa_n_col.format(pre='away', n=window)
                        rel_gaa = cons.gaa_n_col.format(pre='rel', n=window)
                    else:
                        home_gaa = f'{filt}_' + cons.gaa_n_col.format(pre='home', n=window)
                        away_gaa = f'{filt}_' + cons.gaa_n_col.format(pre='away', n=window)
                        rel_gaa = f'{filt}_' + cons.gaa_n_col.format(pre='rel', n=window)
                    data_df[rel_gaa] = data_df[home_gaa] - data_df[away_gaa]
                    data_df.drop(columns=[home_gaa, away_gaa], inplace=True)

        # add a feature that tracks the number of days since each starting goalie last played
        if feature == cons.goalie_days_rest_col:
            data_df = goalie_rest(data_df, goalie_df, team='home', verbose=verbose)
            data_df = goalie_rest(data_df, goalie_df, team='away', verbose=verbose)

            # create relational feature, drop individual features
            home_goalie_rest = cons.goalie_days_rest_col.format(pre='home')
            away_goalie_rest = cons.goalie_days_rest_col.format(pre='away')
            rel_goalie_rest = cons.goalie_days_rest_col.format(pre='rel')
            data_df[rel_goalie_rest] = data_df[home_goalie_rest] - data_df[away_goalie_rest]
            data_df.drop(columns=[home_goalie_rest, away_goalie_rest], inplace=True)

            # cap the max relative goalie rest at 14 days, since a goalie who has not played in two weeks is likely injured or otherwise unavailable
            data_df[rel_goalie_rest] = data_df[rel_goalie_rest].clip(lower=-14, upper=14)

        # add a feature that tracks the number of starts for each starting goalie over the last n days
        if feature == cons.num_starts_n_col:
            for window in cons.goalie_feat_windows_2:
                if verbose: print(f'\t\tProcessing window: {window}')

                data_df = goalie_starts_n(data_df, goalie_df, team='home', window=window, verbose=verbose)
                data_df = goalie_starts_n(data_df, goalie_df, team='away', window=window, verbose=verbose)

                # create relational feature, drop individual features
                home_num_starts = cons.num_starts_n_col.format(pre='home', n=window)
                away_num_starts = cons.num_starts_n_col.format(pre='away', n=window)
                rel_num_starts = cons.num_starts_n_col.format(pre='rel', n=window)
                data_df[rel_num_starts] = data_df[home_num_starts] - data_df[away_num_starts]
                data_df.drop(columns=[home_num_starts, away_num_starts], inplace=True)

    if data_df_in.empty:
        for season in data_df[cons.season_name_col].unique():
            print(f'Writing goalie features for season: {season}...')
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.goalie_features_folder, cons.goalie_features_filename.format(season=season))

    return data_df, goalie_features


def load_goalie_df():
    """Load the per-game goalie stats dataframe used as the source for all goalie features."""

    # load the saved goalie features dataframe
    goalie_df = csvLoad(cons.goalie_features_folder, cons.goalie_data_filename)
    goalie_df[cons.starttime_est_col] = pd.to_datetime(goalie_df[cons.starttime_est_col])
    goalie_df[cons.goalie_id_col] = goalie_df[cons.goalie_id_col].astype('Float64')
    goalie_df[cons.season_name_col] = goalie_df[cons.season_name_col].astype('string')

    return goalie_df


def _asof_merge_goalie_stat(data_df, goalie_sorted, team_goalie_id_col, stat_cols):
    """As-of join each data_df row to the most recent prior game played by its starting goalie.

    Returns goalie_sorted's starttime and stat_cols, re-indexed to align with data_df.
    """

    left = data_df[[team_goalie_id_col, cons.starttime_est_col]].reset_index()
    left = left.sort_values(cons.starttime_est_col)
    left[team_goalie_id_col] = left[team_goalie_id_col].astype('Float64')

    merged = pd.merge_asof(left, goalie_sorted[[cons.goalie_id_col, cons.starttime_est_col] + stat_cols],
                            on=cons.starttime_est_col, left_by=team_goalie_id_col, right_by=cons.goalie_id_col,
                            direction='backward', allow_exact_matches=False)

    return merged.set_index('index')


def save_per_n(data_df, goalie_df, team, window, filt=None, verbose=False):
    """Add <team>'s starting goalie's save percentage over their prior `window` games (or 'Season' to date)."""

    # create a new column for the save percentage over the last n games for each starting goalie
    if not filt:
        save_per_n_col = cons.save_per_n_col.format(pre=team, n=window)
    else:
        save_per_n_col = f'{filt}_' + cons.save_per_n_col.format(pre=team, n=window)
    goalie_id_col = f'{team}_goalie_id'

    saves_col, shots_against_col = _SAVE_STAT_COLS_BY_FILT[filt]

    # precompute, per goalie, the rolling save/shot totals over their last `window` games so each
    # data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
    if window != 'Season':
        goalie_sorted['_roll_saves'] = goalie_sorted.groupby(cons.goalie_id_col)[saves_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted['_roll_shots'] = goalie_sorted.groupby(cons.goalie_id_col)[shots_against_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    # if the window is season, then we can just use the cumulative totals for each goalie up to that point in the season
    else:
        goalie_sorted['_roll_saves'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[saves_col] \
            .cumsum()
        goalie_sorted['_roll_shots'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[shots_against_col] \
            .cumsum()
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    merged = _asof_merge_goalie_stat(data_df, goalie_sorted, goalie_id_col, ['_roll_saves', '_roll_shots'])

    # treat goalies with no shots faced yet in the window as a 0% save rate rather than NaN/inf
    zero_shots = merged['_roll_shots'].notna() & (merged['_roll_shots'] == 0)
    save_percentage = (merged['_roll_saves'] / merged['_roll_shots']).mask(zero_shots, 0.0)

    data_df[save_per_n_col] = save_percentage

    return data_df


def gaa_n(data_df, goalie_df, team, window, filt=None, verbose=False):
    """Add <team>'s starting goalie's goals-against-per-60 over their prior `window` games (or 'Season' to date)."""

    # create a new column for the goals against average over the last n games for each starting goalie
    if not filt:
        gaa_n_col = cons.gaa_n_col.format(pre=team, n=window)
    else:
        gaa_n_col = f'{filt}_' + cons.gaa_n_col.format(pre=team, n=window)
    team_goalie_id_col = f'{team}_goalie_id'

    goals_against_col = _GOALS_AGAINST_COL_BY_FILT[filt]

    # precompute, per goalie, the rolling goals against/60 mins toi over their last `window` games so each
    # data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
    if window != 'Season':
        goalie_sorted['_roll_goals'] = goalie_sorted.groupby(cons.goalie_id_col)[goals_against_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted['_roll_toi'] = goalie_sorted.groupby(cons.goalie_id_col)['toi_secs'] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    # if the window is season, then we can just use the cumulative totals for each goalie up to that point in the season
    else:
        goalie_sorted['_roll_goals'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[goals_against_col] \
            .cumsum()
        goalie_sorted['_roll_toi'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])['toi_secs'] \
            .cumsum()
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    merged = _asof_merge_goalie_stat(data_df, goalie_sorted, team_goalie_id_col, ['_roll_goals', '_roll_toi'])

    # goals against per 60 minutes of ice time
    data_df[gaa_n_col] = (merged['_roll_goals'] * 60) / (merged['_roll_toi'] / 60)

    return data_df


def goalie_rest(data_df, goalie_df, team, verbose=False):
    """Add the number of days since <team>'s starting goalie last played."""

    team_goalie_id_col = f'{team}_goalie_id'

    # precompute, per goalie, the last game played so each data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
    goalie_sorted['_last_game_played'] = goalie_sorted.groupby(cons.goalie_id_col)[cons.starttime_est_col].shift(1)
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    merged = _asof_merge_goalie_stat(data_df, goalie_sorted, team_goalie_id_col, ['_last_game_played'])
    data_df[cons.goalie_days_rest_col.format(pre=team)] = (merged[cons.starttime_est_col] - merged['_last_game_played']).dt.days

    return data_df


def goalie_starts_n(data_df, goalie_df, team, window, verbose=False):
    """Add the number of games <team>'s starting goalie has started in the trailing `window` days (or so far this season)."""

    team_goalie_id_col = f'{team}_goalie_id'
    num_starts_n_col = cons.num_starts_n_col.format(pre=team, n=window)

    # count games within the trailing `window` days via searchsorted, since each goalie's games are
    # already sorted chronologically here (avoids groupby().rolling(on=...) reindex issues)
    def _rolling_start_count(times):
        times = times.to_numpy(dtype='datetime64[ns]')
        lower_bound = times - np.timedelta64(window, 'D')
        left_idx = np.searchsorted(times, lower_bound, side='right')
        right_idx = np.arange(1, len(times) + 1)
        return right_idx - left_idx

    # precompute, per goalie, the rolling number of starts over their last `window` days so each
    # data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
    if window != 'Season':
        goalie_sorted['_roll_starts'] = goalie_sorted.groupby(cons.goalie_id_col)[cons.starttime_est_col] \
            .transform(_rolling_start_count)
    # compute the number of games each goalie has started so far in the season up to that point
    else:
        goalie_sorted['_roll_starts'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col]).cumcount() + 1
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    merged = _asof_merge_goalie_stat(data_df, goalie_sorted, team_goalie_id_col, ['_roll_starts'])
    data_df[num_starts_n_col] = merged['_roll_starts']

    return data_df
