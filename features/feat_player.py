"""Builds player-based features (lineup strength) for the model dataset."""

import json

import numpy as np
import pandas as pd
import constants as cons

from schedule import load_sched_df_features
from utils.file_utils import csvLoad, csvSave

# fallback value/value_per60 used when a player has no prior-season history (league-wide averages)
DEFAULT_VALUE_PER60 = 0.001056387
DEFAULT_VALUE = 35.77355929

DEFAULT_PLAYER_VALUE_FORMULA = {
    'skater': {
        'goals': 1.0,
        'assists': 0.7,
        'powerPlayPoints': 0.3,
        'shorthandedPoints': 0.5,
        'gameWinningGoals': 0.3,
        'shots': 0.05,
        'plusMinus': 0.1,
    },
    'goalie': {
        'savePctg': 500.0,
        'shutouts': 2.0,
        'goalsAgainstAvg': 5.0,
        'wins_per_start': 10.0,
    },
}


def resolve_player_value_formula(formula=None):
    """Merge a user-supplied player value formula (full or partial) over the default weights."""

    resolved = {
        'skater': dict(DEFAULT_PLAYER_VALUE_FORMULA['skater']),
        'goalie': dict(DEFAULT_PLAYER_VALUE_FORMULA['goalie']),
    }

    if formula is None:
        return resolved

    if 'skater' in formula:
        resolved['skater'].update(formula['skater'])
    if 'goalie' in formula:
        resolved['goalie'].update(formula['goalie'])

    for key, value in formula.items():
        if key in ('skater', 'goalie'):
            continue
        if key in resolved['skater']:
            resolved['skater'][key] = value
        elif key in resolved['goalie']:
            resolved['goalie'][key] = value

    return resolved


def player_features_update(data_df_in=pd.DataFrame, verbose=False, player_value_formula=None):
    """Add all player features to data_df_in (or a freshly loaded feature set if empty)."""

    if data_df_in.empty:
        data_df = load_sched_df_features(feat_set_label='player')
    else:
        data_df = data_df_in[cons.player_feature_cols]

    # add player-based features
    player_features = ['lineup_strength']

    # features to add in the future
    future_features = []

    # windows to perform rolling calculations over for certain features
    windows = [4, 7]

    player_df = load_player_df()

    # if verbose: print(f'\tComputing Player values...')
    player_df = compute_player_value(player_df, formula=player_value_formula)

    for feature in player_features:

        # if verbose: print(f'\tAdding {feature}...')

        if feature == 'lineup_strength':
            data_df = lineup_strength(data_df, player_df)
            data_df[cons.lineup_strength_per60_col.format(pre='rel')] = data_df[cons.lineup_strength_per60_col.format(pre='home')] - data_df[cons.lineup_strength_per60_col.format(pre='away')]
            data_df[cons.lineup_strength_col.format(pre='rel')] = data_df[cons.lineup_strength_col.format(pre='home')] - data_df[cons.lineup_strength_col.format(pre='away')]

    if data_df_in.empty:
        print(f'Writing player features...')
        for season in data_df[cons.season_name_col].unique():
            data_df_season = data_df.loc[data_df[cons.season_name_col] == season].copy()
            csvSave(data_df_season, cons.player_features_folder, cons.player_features_filename.format(season=season))

    return data_df, player_features


def load_player_df():
    """Load the per-season player stats dataframe used as the source for all player features."""

    # load the saved player features dataframe
    player_df = csvLoad(cons.player_features_folder, cons.player_data_filename)
    player_df[cons.season_name_col] = player_df[cons.season_name_col].astype('string')
    
    return player_df


def compute_player_value(player_df, formula=None):
    """Aggregate each player's per-season stats and score them into a single `value`/`value_per60`."""

    formula_cfg = resolve_player_value_formula(formula)

    # merge the data with different game types for each playerId/seasonName/teamName
    player_df = player_df.groupby([cons.season_name_col, 'playerId'], as_index=False).agg({
        'gamesPlayed': 'sum',
        'assists': 'sum',
        'totToi': 'sum',
        'gameWinningGoals': 'sum',
        'goals': 'sum',
        'otGoals': 'sum',
        'pim': 'sum',
        'plusMinus': 'sum',
        'points': 'sum',
        'powerPlayGoals': 'sum',
        'powerPlayPoints': 'sum',
        'shorthandedGoals': 'sum',
        'shorthandedPoints': 'sum',
        'shots': 'sum',
        'gamesStarted': 'sum',
        'goalsAgainst': 'sum',
        'losses': 'sum',
        'otLosses': 'sum',
        'shotsAgainst': 'sum',
        'shutouts': 'sum',
        'wins': 'sum',
        'position': 'first'
    })

    # re-compute average features, except for faceoffWinningPctg
    player_df.loc[player_df['position'] != 'goalie', 'shootingPctg'] = player_df['goals'] / player_df['shots'].replace(0, 1)
    player_df.loc[player_df['position'] == 'goalie', 'goalsAgainstAvg'] = player_df['goalsAgainst'] / (player_df['totToi'] / 60)
    player_df.loc[player_df['position'] == 'goalie', 'savePctg'] = 1 - (player_df['goalsAgainst'] / player_df['shotsAgainst'].replace(0, 1))

    # compute skater value based on features
    skater_formula = formula_cfg['skater']
    player_df.loc[player_df['position'] != 'goalie', 'value'] = (
        player_df['goals'] * skater_formula['goals'] +
        player_df['assists'] * skater_formula['assists'] +
        player_df['powerPlayPoints'] * skater_formula['powerPlayPoints'] +
        player_df['shorthandedPoints'] * skater_formula['shorthandedPoints'] +
        player_df['gameWinningGoals'] * skater_formula['gameWinningGoals'] +
        player_df['shots'] * skater_formula['shots'] +
        player_df['plusMinus'] * skater_formula['plusMinus']
    )

    player_df.loc[player_df['position'] != 'goalie', 'value_per60'] = player_df['value'] / player_df['totToi']
    player_df.loc[player_df['totToi'] <= 0, 'value_per60'] = 0

    # compute goalie value based on features - 1916 max
    goalie_formula = formula_cfg['goalie']
    player_df.loc[player_df['position'] == 'goalie', 'value'] = (
        player_df['savePctg'] * goalie_formula['savePctg'] +
        (player_df['shutouts'] * goalie_formula['shutouts']) -
        player_df['goalsAgainstAvg'] * goalie_formula['goalsAgainstAvg'] +
        (player_df['wins'] / player_df['gamesStarted'].replace(0, 1)) * goalie_formula['wins_per_start']
    )

    player_df.loc[player_df['position'] == 'goalie', 'value_per60'] = player_df['value'] / player_df['totToi']
    player_df.loc[player_df['totToi'] <= 0, 'value_per60'] = 0

    player_df = player_df.sort_values(by=['playerId', cons.season_name_col], ascending=True)

    return player_df


def compute_player_asof_value(values_df, playerid, decay_rate=0.7):
    """Exponentially-weighted average of a player's `value`/`value_per60` across prior seasons."""

    if values_df.empty:
        return DEFAULT_VALUE_PER60, DEFAULT_VALUE

    weights = [decay_rate ** i for i in range(len(values_df))]
    value_per60 = np.average(values_df['value_per60'], weights=weights)
    value = np.average(values_df['value'], weights=weights)

    return value_per60, value


def lineup_strength(data_df, player_df):
    """Add each matchup's home/away lineup strength as the sum of each player's as-of value."""

    # Precompute each player's prior-season values once so every matchup can be scored with a
    # direct lookup instead of re-filtering the full player history for each lineup member.
    player_histories = {}
    for player_id, group in player_df.groupby('playerId', sort=False):
        group = group.sort_values(by=cons.season_name_col, ascending=True)
        player_histories[player_id] = {
            cons.season_name_col: group[cons.season_name_col].to_numpy(),
            'value': group['value'].to_numpy(dtype=float),
            'value_per60': group['value_per60'].to_numpy(dtype=float),
        }

    asof_cache = {}

    def get_asof_value(player_id, season_label):
        # cache repeated (player, season) lookups since the same lineup often recurs across games
        cache_key = (player_id, season_label)
        if cache_key in asof_cache:
            return asof_cache[cache_key]

        history = player_histories.get(player_id)
        if history is None:
            asof_cache[cache_key] = (DEFAULT_VALUE_PER60, DEFAULT_VALUE)
            return asof_cache[cache_key]

        # only consider seasons strictly before season_label, to avoid leaking future data
        seasons = history[cons.season_name_col]
        prior_idx = np.searchsorted(seasons, season_label, side='left')
        prior_value_per60 = history['value_per60'][:prior_idx]
        prior_values = history['value'][:prior_idx]

        if len(prior_value_per60) == 0:
            asof_cache[cache_key] = (DEFAULT_VALUE_PER60, DEFAULT_VALUE)
            return asof_cache[cache_key]

        weights = [0.7 ** i for i in range(len(prior_value_per60))]
        value_per60 = np.average(prior_value_per60, weights=weights)
        value = np.average(prior_values, weights=weights)
        asof_cache[cache_key] = (value_per60, value)
        return asof_cache[cache_key]

    # for every matchup, compute the sum value of each lineup's player values based off a weighted average of each player's previous season's value and value_per60
    home_lineup_strengths_per60 = []
    away_lineup_strengths_per60 = []
    home_lineup_strengths = []
    away_lineup_strengths = []

    data_df.loc[data_df[cons.home_lineup_col].isnull(), cons.home_lineup_col] = '[]'
    data_df.loc[data_df[cons.away_lineup_col].isnull(), cons.away_lineup_col] = '[]'

    for index, row in data_df.iterrows():
        home_lineup = json.loads(row[cons.home_lineup_col])
        away_lineup = json.loads(row[cons.away_lineup_col])

        home_strength_per60 = 0
        away_strength_per60 = 0
        home_strength = 0
        away_strength = 0

        for player in home_lineup:
            value_per60, value = get_asof_value(player, row[cons.season_name_col])
            home_strength_per60 += value_per60 if value_per60 is not None else 0
            home_strength += value if value is not None else 0

        for player in away_lineup:
            value_per60, value = get_asof_value(player, row[cons.season_name_col])
            away_strength_per60 += value_per60 if value_per60 is not None else 0
            away_strength += value if value is not None else 0

        home_lineup_strengths_per60.append(home_strength_per60)
        away_lineup_strengths_per60.append(away_strength_per60)
        home_lineup_strengths.append(home_strength)
        away_lineup_strengths.append(away_strength)

    data_df[cons.lineup_strength_per60_col.format(pre='home')] = home_lineup_strengths_per60
    data_df[cons.lineup_strength_per60_col.format(pre='away')] = away_lineup_strengths_per60
    data_df[cons.lineup_strength_col.format(pre='home')] = home_lineup_strengths
    data_df[cons.lineup_strength_col.format(pre='away')] = away_lineup_strengths

    return data_df