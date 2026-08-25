import numpy as np
import pandas as pd
import constants as cons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from features.feat_team import compute_elo_ratings
from features.features import feature_data_load
from utils.skl_utils import preprocess_feature_data



k_values = [6, 10, 14, 20]
home_adv_values = [0, 25, 50, 75, 100]

feature_df = feature_data_load()
feature_df, feature_list = preprocess_feature_data(feature_df)

seasons = sorted(feature_df[cons.season_name_col].unique())
results = []

for k in k_values:
    for ha in home_adv_values:
        fold_aucs = []

        for i in range(1, len(seasons) - 1):
            train_seasons_list = seasons[:i+1]  # includes val_season's prior seasons
            val_season = seasons[i]

            # Elo must be computed sequentially over ALL games up to and including val_season,
            # so later games get ratings informed only by earlier games
            relevant_seasons = seasons[:i+1]
            subset = feature_df[feature_df[cons.season_name_col].isin(relevant_seasons)].copy()
            subset = subset.sort_values(cons.starttime_est_col).reset_index(drop=True)

            subset = compute_elo_ratings(subset, k=k, home_advantage=ha)
            subset[cons.elo_rat_col.format(pre='rel')] = subset[cons.elo_rat_col.format(pre='home')] - subset[cons.elo_rat_col.format(pre='away')]

            train_df = subset[subset[cons.season_name_col].isin(seasons[:i])]
            val_df = subset[subset[cons.season_name_col] == val_season]

            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(train_df[feature_list], train_df[cons.home_team_win_col])
            probs = model.predict_proba(val_df[feature_list])[:, 1]

            fold_aucs.append(roc_auc_score(val_df[cons.home_team_win_col], probs))

        results.append({
            'k': k,
            'home_advantage': ha,
            'avg_auc': np.mean(fold_aucs),
            'std_auc': np.std(fold_aucs),
            'fold_aucs': fold_aucs
        })
        print(f"k={k}, home_advantage={ha}: avg_auc={np.mean(fold_aucs):.4f}")

results_df = pd.DataFrame(results).sort_values('avg_auc', ascending=False)
print(results_df)
pass