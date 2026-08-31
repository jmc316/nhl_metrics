import pandas as pd
import numpy as np
import constants as cons

from utils.file_utils import csvSave, csvLoad
from schedule import load_sched_df_features


def goalie_features_update(data_df_in=pd.DataFrame, verbose=False):

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

            # cap the max relative goalie rest at 7 days, since a goalie who has not played in a week is likely injured or otherwise unavailable
            data_df[rel_goalie_rest] = data_df[rel_goalie_rest].clip(lower=-7, upper=7)

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

    # load the saved goalie features dataframe
    goalie_df = csvLoad(cons.goalie_features_folder, cons.goalie_data_filename)
    goalie_df[cons.starttime_est_col] = pd.to_datetime(goalie_df[cons.starttime_est_col])
    goalie_df[cons.goalie_id_col] = goalie_df[cons.goalie_id_col].astype('Float64')
    goalie_df[cons.season_name_col] = goalie_df[cons.season_name_col].astype('string')

    return goalie_df


def save_per_n(data_df, goalie_df, team, window, filt=None, verbose=False):

    # create a new column for the save percentage over the last n games for each starting goalie
    if not filt:
        save_per_n_col = cons.save_per_n_col.format(pre=team, n=window)
    else:
        save_per_n_col = f'{filt}_' + cons.save_per_n_col.format(pre=team, n=window)
    goalie_id_col = f'{team}_goalie_id'

    if not filt:
        saves_col = 'tot_saves'
        shots_against_col = 'tot_shots_against'
    elif filt == 'ev':
        saves_col = 'ev_saves'
        shots_against_col = 'ev_shots_against'
    elif filt == 'pp':
        saves_col = 'pp_saves'
        shots_against_col = 'pp_shots_against'
    elif filt == 'sh':
        saves_col = 'sh_saves'
        shots_against_col = 'sh_shots_against'

    # precompute, per goalie, the rolling save/shot totals over their last `window` games so each
    # data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    if window != 'Season':
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_saves'] = goalie_sorted.groupby(cons.goalie_id_col)[saves_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted['_roll_shots'] = goalie_sorted.groupby(cons.goalie_id_col)[shots_against_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)
    # if the window is season, then we can just use the cumulative totals for each goalie up to that point in the season
    else:
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_saves'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[saves_col] \
            .cumsum()
        goalie_sorted['_roll_shots'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[shots_against_col] \
            .cumsum()
        goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    # for each data_df row, as-of match to the most recent prior game played by that goalie
    left = data_df[[goalie_id_col, cons.starttime_est_col]].reset_index()
    left = left.sort_values(cons.starttime_est_col)
    left[goalie_id_col] = left[goalie_id_col].astype('Float64')

    merged = pd.merge_asof(left, goalie_sorted[[cons.goalie_id_col, cons.starttime_est_col, '_roll_saves', '_roll_shots']],
                            on=cons.starttime_est_col, left_by=goalie_id_col, right_by=cons.goalie_id_col,
                            direction='backward', allow_exact_matches=False)

    has_history = merged['_roll_shots'].notna()
    zero_shots = has_history & (merged['_roll_shots'] == 0)
    save_percentage = merged['_roll_saves'] / merged['_roll_shots']
    save_percentage = save_percentage.mask(zero_shots, 0.0)

    merged = merged.set_index('index')
    data_df[save_per_n_col] = save_percentage.set_axis(merged.index)

    return data_df


def gaa_n(data_df, goalie_df, team, window, filt=None, verbose=False):
    # create a new column for the goals against average over the last n games for each starting goalie
    if not filt:
        gaa_n_col = cons.gaa_n_col.format(pre=team, n=window)
    else:
        gaa_n_col = f'{filt}_' + cons.gaa_n_col.format(pre=team, n=window)
    team_goalie_id_col = f'{team}_goalie_id'

    if not filt:
        goals_against_col = 'tot_goals_against'
    elif filt == 'ev':
        goals_against_col = 'ev_goals_against'
    elif filt == 'pp':
        goals_against_col = 'pp_goals_against'
    elif filt == 'sh':
        goals_against_col = 'sh_goals_against'

    # precompute, per goalie, the rolling goals against/60 mins toi over their last `window` games so each
    # data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    if window != 'Season':
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_goals'] = goalie_sorted.groupby(cons.goalie_id_col)[goals_against_col] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted['_roll_toi'] = goalie_sorted.groupby(cons.goalie_id_col)['toi_secs'] \
            .rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)
    # if the window is season, then we can just use the cumulative totals for each goalie up to that point in the season
    else:
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_goals'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])[goals_against_col] \
            .cumsum()
        goalie_sorted['_roll_toi'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col])['toi_secs'] \
            .cumsum()
        goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    # for each data_df row, as-of match to the most recent prior game played by that goalie
    left = data_df[[team_goalie_id_col, cons.starttime_est_col]].reset_index()
    left = left.sort_values(cons.starttime_est_col)
    left[team_goalie_id_col] = left[team_goalie_id_col].astype('Float64')

    merged = pd.merge_asof(left, goalie_sorted[[cons.goalie_id_col, cons.starttime_est_col, '_roll_goals', '_roll_toi']],
                            on=cons.starttime_est_col, left_by=team_goalie_id_col, right_by=cons.goalie_id_col,
                            direction='backward', allow_exact_matches=False)

    gaa = (merged['_roll_goals'] * 60) / (merged['_roll_toi'] / 60)  # Goals against per 60 minutes
    
    merged = merged.set_index('index')
    data_df[gaa_n_col] = gaa.set_axis(merged.index)
    
    return data_df


def goalie_rest(data_df, goalie_df, team, verbose=False):

    team_goalie_id_col = f'{team}_goalie_id'

    # precompute, per goalie, the last game played so each data_df row can be looked up instead of re-filtering and re-sorting goalie_df every iteration
    goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
    goalie_sorted['_last_game_played'] = goalie_sorted.groupby(cons.goalie_id_col)[cons.starttime_est_col].shift(1)
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    # for each data_df row, as-of match to the most recent prior game played by that goalie
    left = data_df[[team_goalie_id_col, cons.starttime_est_col]].reset_index()
    left = left.sort_values(cons.starttime_est_col)
    left[team_goalie_id_col] = left[team_goalie_id_col].astype('Float64')

    merged = pd.merge_asof(left, goalie_sorted[[cons.goalie_id_col, cons.starttime_est_col, '_last_game_played']],
                            on=cons.starttime_est_col, left_by=team_goalie_id_col, right_by=cons.goalie_id_col,
                            direction='backward', allow_exact_matches=False)
    merged = merged.set_index('index')
    data_df[cons.goalie_days_rest_col.format(pre=team)] = (merged[cons.starttime_est_col] - merged['_last_game_played']).dt.days

    return data_df


def goalie_starts_n(data_df, goalie_df, team, window, verbose=False):

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
    if window != 'Season':
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_starts'] = goalie_sorted.groupby(cons.goalie_id_col)[cons.starttime_est_col] \
            .transform(_rolling_start_count)
    # compute the number of games each goalie has started so far in the season up to that point
    else:
        goalie_sorted = goalie_df.sort_values([cons.goalie_id_col, cons.starttime_est_col]).reset_index(drop=True)
        goalie_sorted['_roll_starts'] = goalie_sorted.groupby([cons.goalie_id_col, cons.season_name_col]).cumcount() + 1
        
    goalie_sorted = goalie_sorted.sort_values(cons.starttime_est_col)

    # for each data_df row, as-of match to the most recent prior game played by that goalie
    left = data_df[[team_goalie_id_col, cons.starttime_est_col]].reset_index()
    left = left.sort_values(cons.starttime_est_col)
    left[team_goalie_id_col] = left[team_goalie_id_col].astype('Float64')

    merged = pd.merge_asof(left, goalie_sorted[[cons.goalie_id_col, cons.starttime_est_col, '_roll_starts']],
                            on=cons.starttime_est_col, left_by=team_goalie_id_col, right_by=cons.goalie_id_col,
                            direction='backward', allow_exact_matches=False)
    merged = merged.set_index('index')
    data_df[num_starts_n_col] = merged['_roll_starts']

    return data_df