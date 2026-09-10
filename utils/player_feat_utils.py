import constants as cons
from utils.file_utils import csvLoad

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