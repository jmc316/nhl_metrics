import argparse
import itertools
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import constants as cons
from features.features import feat_update
from schedule import load_sched_df_features
from utils.skl_utils import preprocess_feature_data


def build_formula_grid(limit=None):
    skater_goal_values = [0.6, 0.8, 1.0, 1.2]
    skater_assist_values = [0.5, 0.7, 0.9]
    skater_pp_values = [0.2, 0.3, 0.4]
    skater_sh_values = [0.3, 0.5, 0.7]
    skater_gwg_values = [0.2, 0.3, 0.5]
    skater_shot_values = [0.03, 0.05, 0.07]
    skater_plusminus_values = [0.05, 0.1, 0.15]

    goalie_save_values = [450.0, 500.0, 550.0]
    goalie_shutout_values = [1.0, 2.0, 3.0]
    goalie_gaa_values = [4.0, 5.0, 6.0]
    goalie_winstart_values = [8.0, 10.0, 12.0]

    grid = []
    for sk_goals, sk_assists, sk_pp, sk_sh, sk_gwg, sk_shots, sk_pm, g_save, g_shut, g_gaa, g_ws in itertools.product(
        skater_goal_values,
        skater_assist_values,
        skater_pp_values,
        skater_sh_values,
        skater_gwg_values,
        skater_shot_values,
        skater_plusminus_values,
        goalie_save_values,
        goalie_shutout_values,
        goalie_gaa_values,
        goalie_winstart_values,
    ):
        grid.append({
            'skater': {
                'goals': sk_goals,
                'assists': sk_assists,
                'powerPlayPoints': sk_pp,
                'shorthandedPoints': sk_sh,
                'gameWinningGoals': sk_gwg,
                'shots': sk_shots,
                'plusMinus': sk_pm,
            },
            'goalie': {
                'savePctg': g_save,
                'shutouts': g_shut,
                'goalsAgainstAvg': g_gaa,
                'wins_per_start': g_ws,
            }
        })

    if limit is not None:
        return grid[:limit]
    return grid


def walk_forward_eval(feature_df, feature_list):
    seasons = sorted(feature_df[cons.season_name_col].unique())
    fold_scores = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_df = feature_df[feature_df[cons.season_name_col].isin(train_seasons)].copy()
        val_df = feature_df[feature_df[cons.season_name_col] == val_season].copy()

        train_df = train_df[train_df[cons.home_team_win_col].notna()].copy()
        val_df = val_df[val_df[cons.home_team_win_col].notna()].copy()

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[feature_list]
        y_train = train_df[cons.home_team_win_col]
        X_val = val_df[feature_list]
        y_val = val_df[cons.home_team_win_col]

        X_train = X_train.dropna(axis=1, how='all')
        X_val = X_val[X_train.columns]

        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        preds = model.predict(X_val)

        fold_scores.append({
            'season': val_season,
            'auc': roc_auc_score(y_val, probs),
            'accuracy': accuracy_score(y_val, preds),
        })

    if not fold_scores:
        raise ValueError('No validation folds were created. Check the season coverage in the feature data.')

    auc_mean = float(np.mean([x['auc'] for x in fold_scores]))
    acc_mean = float(np.mean([x['accuracy'] for x in fold_scores]))
    return {
        'folds': fold_scores,
        'auc': auc_mean,
        'accuracy': acc_mean,
    }


def main():
    parser = argparse.ArgumentParser(description='Grid-search player value coefficients and report AUC/accuracy.')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of parameter combinations to try.')
    args = parser.parse_args()

    base_sched_df = load_sched_df_features()
    formula_grid = build_formula_grid(limit=args.limit)

    results = []
    for formula_cfg in formula_grid:
        feature_df = feat_update(save_feat_data=False, verbose=False, player_value_formula=formula_cfg)
        feature_df, feature_list = preprocess_feature_data(feature_df)
        metrics = walk_forward_eval(feature_df, feature_list)

        result = {
            'formula': formula_cfg,
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy'],
            'folds': metrics['folds'],
        }
        results.append(result)

        print(json.dumps({
            'formula': formula_cfg,
            'auc': round(metrics['auc'], 6),
            'accuracy': round(metrics['accuracy'], 6),
        }, sort_keys=True))

    results.sort(key=lambda item: (item['auc'], item['accuracy']), reverse=True)
    print('\nBEST RESULTS:')
    for item in results[:10]:
        print(json.dumps({
            'formula': item['formula'],
            'auc': round(item['auc'], 6),
            'accuracy': round(item['accuracy'], 6),
        }, sort_keys=True))


if __name__ == '__main__':
    main()
