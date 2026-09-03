from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import numpy as np
import pandas as pd

from features.features import feature_data_load
import utils.skl_utils as sklu

def depth_and_leaf_hyperparameter_tuning(processed_df, feature_list, seasons, target_col):
    
    # Stage 1: coarse sweep on the two most impactful params
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_leaf': [1, 5, 10, 20, 40],
    }

    # initial testing values for hyperparameter tuning
    max_depth_values = [5, 8, 12, 16, 20, None]
    min_samples_leaf_values = [1, 5, 10, 20, 40]

    # second round of testing values for hyperparameter tuning based on your previous results
    max_depth_values = [8, 10, 12, 14, 16]
    min_samples_leaf_values = [30, 35, 40, 45, 50]

    results = []

    for depth in max_depth_values:
        for leaf in min_samples_leaf_values:
            fold_aucs = []
            fold_accs = []

            for i in range(1, len(seasons) - 1):
                train_seasons_list = seasons[:i]
                val_season = seasons[i]

                train_df = processed_df[processed_df['seasonName'].isin(train_seasons_list)].sort_values(
                    ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)
                val_df = processed_df[processed_df['seasonName'] == val_season].sort_values(
                    ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)

                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=42,
                    n_jobs=1  # reproducibility over speed, given your earlier debugging
                )
                model.fit(train_df[feature_list], train_df[target_col])

                probs = model.predict_proba(val_df[feature_list])[:, 1]
                preds = model.predict(val_df[feature_list])

                fold_aucs.append(roc_auc_score(val_df[target_col], probs))
                fold_accs.append(accuracy_score(val_df[target_col], preds))

            results.append({
                'max_depth': depth,
                'min_samples_leaf': leaf,
                'avg_auc': np.mean(fold_aucs),
                'std_auc': np.std(fold_aucs),
                'avg_accuracy': np.mean(fold_accs),
                'fold_aucs': fold_aucs
            })
            print(f"depth={depth}, leaf={leaf}: avg_auc={np.mean(fold_aucs):.4f} (std={np.std(fold_aucs):.4f}), avg_acc={np.mean(fold_accs):.4f}")

    results_df = pd.DataFrame(results).sort_values('avg_auc', ascending=False)
    print(results_df.to_string())

    # Save results for reference given your reproducibility history
    with open('hyperparam_search_stage1.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


def n_estimator_max_features_tuning(processed_df, feature_list, seasons, target_col):

    # initial testing values for hyperparameter tuning
    n_estimators_values = [200, 300, 500, 800]
    max_features_values = ['sqrt', 'log2', None]

    results = []

    for n_est in n_estimators_values:
        for max_feat in max_features_values:
            fold_aucs = []
            fold_accs = []

            for i in range(1, len(seasons) - 1):
                train_seasons_list = seasons[:i]
                val_season = seasons[i]

                train_df = processed_df[processed_df['seasonName'].isin(train_seasons_list)].sort_values(
                    ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)
                val_df = processed_df[processed_df['seasonName'] == val_season].sort_values(
                    ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)

                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=10,
                    min_samples_leaf=40,
                    max_features=max_feat,
                    random_state=42,
                    n_jobs=1  # keep fixed at 1 for reproducibility while comparing configs
                )
                model.fit(train_df[feature_list], train_df[target_col])

                probs = model.predict_proba(val_df[feature_list])[:, 1]
                preds = model.predict(val_df[feature_list])

                fold_aucs.append(roc_auc_score(val_df[target_col], probs))
                fold_accs.append(accuracy_score(val_df[target_col], preds))

            results.append({
                'n_estimators': n_est,
                'max_features': str(max_feat),
                'avg_auc': np.mean(fold_aucs),
                'std_auc': np.std(fold_aucs),
                'avg_accuracy': np.mean(fold_accs),
                'fold_aucs': fold_aucs
            })
            print(f"n_estimators={n_est}, max_features={max_feat}: "
                f"avg_auc={np.mean(fold_aucs):.4f} (std={np.std(fold_aucs):.4f}), "
                f"avg_acc={np.mean(fold_accs):.4f}")

    results_df = pd.DataFrame(results).sort_values('avg_auc', ascending=False)
    print(results_df.to_string())

    # Save for reference, given your reproducibility history
    with open('hyperparam_search_stage2_nest_maxfeat.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


def max_samples_tuning(processed_df, feature_list, seasons, target_col):

    max_samples_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 1.0 == None/default (full bootstrap)

    results = []

    for max_samp in max_samples_values:
        fold_aucs = []
        fold_accs = []

        for i in range(1, len(seasons) - 1):
            train_seasons_list = seasons[:i]
            val_season = seasons[i]

            train_df = processed_df[processed_df['seasonName'].isin(train_seasons_list)].sort_values(
                ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)
            val_df = processed_df[processed_df['seasonName'] == val_season].sort_values(
                ['gameId', 'startTimeEST', 'homeTeamName']).reset_index(drop=True)

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=40,
                max_features='sqrt',
                max_samples=max_samp if max_samp < 1.0 else None,  # None = full bootstrap, sklearn's default
                random_state=42,
                n_jobs=1  # reproducibility while comparing configs
            )
            model.fit(train_df[feature_list], train_df[target_col])

            probs = model.predict_proba(val_df[feature_list])[:, 1]
            preds = model.predict(val_df[feature_list])

            fold_aucs.append(roc_auc_score(val_df[target_col], probs))
            fold_accs.append(accuracy_score(val_df[target_col], preds))

        results.append({
            'max_samples': max_samp,
            'avg_auc': np.mean(fold_aucs),
            'std_auc': np.std(fold_aucs),
            'avg_accuracy': np.mean(fold_accs),
            'fold_aucs': fold_aucs
        })
        print(f"max_samples={max_samp}: avg_auc={np.mean(fold_aucs):.4f} (std={np.std(fold_aucs):.4f}), "
            f"avg_acc={np.mean(fold_accs):.4f}")

    results_df = pd.DataFrame(results).sort_values('avg_auc', ascending=False)
    print(results_df.to_string())

    with open('hyperparam_search_stage3_maxsamples.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":

    # load all of the feature data with all actuals
    feature_df = feature_data_load()

    # pre-process the feature data
    processed_df, feature_list = sklu.preprocess_feature_data(feature_df)

    seasons = sorted(processed_df['seasonName'].unique())
    target_col = 'homeTeamWin'

    print("\nInitiating hyperparameter tuning for Random Forest Classifier...")
    # max_depth_min_samples_leaf_tuning(processed_df, feature_list, seasons, target_col)
    # n_estimator_max_features_tuning(processed_df, feature_list, seasons, target_col)
    max_samples_tuning(processed_df, feature_list, seasons, target_col)