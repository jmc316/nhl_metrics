import shap
import matplotlib

import numpy as np
import pandas as pd
import constants as cons
import matplotlib.pyplot as plt

from utils.file_utils import pklLoad, pklSave, txtSave
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# avoid garbage collection
matplotlib.use('Agg')


def preprocess_feature_data(data_df_in):

    data_df = data_df_in.copy()

    # numeric features that do not need to be scaled or normalized, keep as is for the model
    # num_feats = ['homeGameNumPerc', 'awayGameNumPerc', 'relDaysRest', 'relTravelDist4Days',
    #             'relTravelDist7Days', 'relGamesPlayed4Days', 'relGamesPlayed7Days', 'relCrossedTZ4Days',
    #             'relCrossedTZ7Days']
    num_feats_nonwindow = [cons.reg_game_num_perc_col.format(team='home'), cons.reg_game_num_perc_col.format(team='away'),
                 cons.days_rest_col.format(pre='rel'), cons.elo_rat_col.format(pre='rel')]
    num_feats_window = []
    for window in cons.sched_feat_windows:
        num_feats_window.append(cons.travel_dist_n_days_col.format(pre='rel', n=window))
        num_feats_window.append(cons.games_played_n_days_col.format(pre='rel', n=window))
        num_feats_window.append(cons.crossed_tz_n_days_col.format(pre='rel', n=window))
    for window in cons.team_feat_windows:
        num_feats_window.append(cons.point_per_n_col.format(pre='rel', n=window))
        num_feats_window.append(cons.goal_diff_n_col.format(pre='rel', n=window))
        num_feats_window.append(cons.corsi_per_n_col.format(pre='rel', n=window))
        num_feats_window.append(cons.pp_per_n_col.format(pre='rel', n=window))
        num_feats_window.append(cons.pk_per_n_col.format(pre='rel', n=window))
        # num_feats_window.append(cons.fenwick_per_n_col.format(pre='rel', n=window))
    num_feats = num_feats_nonwindow + num_feats_window

    # boolean categorical features, keep as is for the model
    bool_feats = [cons.is_outdoor_venue_col, cons.is_home_opener_col,
                  cons.is_ret_home_trap_col, cons.is_venue_alt_shock_col]

    # very low cardinality numeric categorical features (< 5 values), keep as is for the model
    low_card_feats = [cons.rival_match_col, cons.market_intensity_col]

    # categorical features to be custom encoded with target encoding
    target_encoding_feats = [cons.venue_timezone_col]
    target_encoding_map = {
        cons.venue_timezone_col: cons.venue_timezone_map
    }
    for feat in target_encoding_feats:
        data_df[feat] = data_df[feat].map(target_encoding_map[feat])

    # medium cardinality categorical features to be one-hot encoded
    # helps avoid implication of false ordering
    # NOTE: venueTimezone added to one-hot encode mapped values above
    one_hot_feats = [cons.day_of_week_col, cons.game_type_col, cons.venue_timezone_col]
    one_hot_feats_new = []
    one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    for col in one_hot_feats:
        one_hot_encoded_df = one_hot_encoder.fit_transform(data_df[[col]])
        one_hot_feats_new.extend(one_hot_encoded_df.columns.tolist())
        data_df = pd.concat([data_df.drop(columns=[col]), one_hot_encoded_df], axis=1)

    # larger cardinality categorical features to be label encoded
    label_encode_feats = [cons.venue_col, cons.season_name_col]
    label_encoder = LabelEncoder()
    for col in label_encode_feats:
        data_df[col] = label_encoder.fit_transform(data_df[col])

    feature_list = num_feats + bool_feats + low_card_feats + one_hot_feats_new + label_encode_feats

    return data_df, feature_list


def model_train(data_df, feature_list, save_model=True):

    actual_df = data_df[data_df[cons.home_team_win_col].notna()]

    if save_model:
        # get the unique seasons in the dataframe, sorted
        seasons = sorted(actual_df[cons.season_name_col].unique())

        # Walk-forward loop: train on all seasons up to N, validate on season N+1
        fold_results = []

        for i in range(1, len(seasons)):  # leave last season out as final holdout test
            train_seasons = seasons[:i]
            val_season = seasons[i]

            train_df = actual_df[actual_df[cons.season_name_col].isin(train_seasons)]
            val_df = actual_df[actual_df[cons.season_name_col] == val_season]

            X_train, y_train = train_df[feature_list], train_df[cons.home_team_win_col]
            X_val, y_val = val_df[feature_list], val_df[cons.home_team_win_col]

            val_model = init_model(random_state_in=42)
            val_model.fit(X_train[feature_list], y_train)

            preds = val_model.predict(X_val)
            probs = val_model.predict_proba(X_val)[:, 1]  # probability of the positive class

            fold_result = {
                'train_seasons': train_seasons,
                'val_season': val_season,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'accuracy': accuracy_score(y_val, preds),
                'auc': roc_auc_score(y_val, probs),
                'log_loss': log_loss(y_val, probs),
                'brier': brier_score_loss(y_val, probs)
            }
            fold_results.append(fold_result)

        fold_results_df = pd.DataFrame(fold_results)
        print('\nValidation Set Results:')
        print(fold_results_df)

        baseline_acc = (actual_df[cons.home_team_win_col] == 1).mean()
        print(f"Baseline Accuracy:          {baseline_acc:.3f}")
        print(f"Avg Model Validation Accuracy:  {fold_results_df['accuracy'].mean():.4f}")
        print(f"Avg Model Validation AUC:  {fold_results_df['auc'].mean():.4f}")
        print(f"Avg Model Validation Log Loss:  {fold_results_df['log_loss'].mean():.4f}")
        print(f"Avg Model Validation Brier Score:  {fold_results_df['brier'].mean():.4f}")

        perm_imp_result = permutation_importance(
            val_model,           # your fitted RandomForestClassifier
            X_val,
            y_val,
            n_repeats=10,    # shuffle each feature 10x, average the effect
            random_state=42,
            n_jobs=-1,       # parallelize across cores
            scoring='accuracy'     # or 'neg_mean_squared_error', etc.
        )

        perm_imp_df = pd.DataFrame({
            'feature': X_val.columns,
            'importance_mean': perm_imp_result.importances_mean,
            'importance_std': perm_imp_result.importances_std
        }).sort_values('importance_mean', ascending=False)

        print('\nPermutation Importance:')
        print(perm_imp_df)

        print('\nFeature Correlation Analysis:')
        corr_matrix = X_val.corr()

        # find all correlations that are greater than .7
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        print('High Correlation Pairs (>|0.7|):')
        for pair in high_corr_pairs:
            print(f'{pair[0]} - {pair[1]}: {pair[2]:.2f}')

        # train a final model on all actual data to use for the prediction set
        print('\nFinalizing model data...')
    final_model = init_model(random_state_in=42)
    final_model.fit(actual_df[feature_list], actual_df[cons.home_team_win_col])

    # Schedule-only baseline (no class_weight):
    # Avg AUC:      ~0.539
    # Avg Accuracy: 0.5352 (baseline: 0.537)
    # Avg Brier:    ~0.254
    # Avg Log Loss: ~0.703

    # Schedule + Team Features (no class_weight):
    # Avg AUC:      0.5976
    # Avg Accuracy: 0.5767 (baseline: 0.537)
    # Avg brier:    0.2436
    # Avg log loss: 0.6812

    # Vegas Model:
    #   Accuracy = ~62-63%
    #   AUC = 68-72%

    if save_model:
        print('\nSaving model file...')
        pklSave(final_model, cons.model_files_folder, cons.sklearn_model_filename)

    return final_model


def model_inference(data_df, feature_list, today_dt, model=None):

    # if the model is not passed in, load it from the pkl file
    if not model:
        model = pklLoad(cons.model_files_folder, cons.sklearn_model_filename)

    x_predict_df = data_df[data_df[cons.home_team_win_col].isna()][feature_list]

    # save the features used to create this model for future reference
    txtSave(feature_list, cons.season_pred_folder.format(date=today_dt), cons.model_features_filename.format(model='skl_rf'))

    # make predictions on the prediction set
    predictset_predictions = model.predict(x_predict_df)

    # update the original data_df with the predictions for the target variables and determine the last period based on the predicted scores
    predict_df = data_df[data_df[cons.home_team_win_col].isna()]
    predict_df[cons.home_team_win_col] = predictset_predictions

    probs = model.predict_proba(x_predict_df)
    home_win_prob = probs[:, 1]  # column 1 = probability of class "1" (home win)
    away_win_prob = probs[:, 0]  # column 0 = probability of class "0" (away win)

    predict_df[cons.home_win_prob_col] = home_win_prob
    predict_df[cons.away_win_prob_col] = away_win_prob

    data_df.update(predict_df[[cons.home_team_win_col,
                               cons.home_win_prob_col,
                               cons.away_win_prob_col
                               ]])

    # fix for bug where prediction is split exactly 50/50
    if not data_df.loc[(data_df[cons.home_win_prob_col]==0.5) & (data_df[cons.away_win_prob_col]==0.5)].empty:
        data_df.loc[(data_df[cons.home_win_prob_col]==0.5) & (data_df[cons.away_win_prob_col]==0.5),
                    [cons.home_team_win_col, cons.home_win_prob_col, cons.away_win_prob_col]] = [1, np.float64(0.5000001), np.float64(0.499999)]

    return data_df, model


def init_model(random_state_in=None):

    """
    SciKit Learn RandomForestRegressor parameters:
    - n_estimators:  The number of trees in the forest (default is 100)
    - criterion:  The function to measure the quality of a split (default is 'squared_error' for regression)
    - max_depth:  The maximum depth of the tree (default is None, which means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
    - min_samples_split:  The minimum number of samples required to split an internal node (default is 2)
    - min_samples_leaf:  The minimum number of samples required to be at a leaf node (default is 1)
    - min_weight_fraction_leaf:  The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node (default is 0.0)
    - max_features:  The number of features to consider when looking for the best split (default is 'auto')
    - max_leaf_nodes:  Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes (default is None)
    - min_impurity_decrease:  A node will be split if this split induces a decrease of the impurity greater than or equal to this value (default is 0.0)
    - bootstrap:  Whether bootstrap samples are used when building trees (default is True)
    - oob_score:  Whether to use out-of-bag samples to estimate the generalization score (default is False)
    - n_jobs:  The number of jobs to run in parallel (default is None, which means 1)
    - random_state:  Controls the randomness of the estimator (default is None)
    - verbose:  Controls the verbosity when fitting and predicting (default is 0)
    - warm_start:  When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble (default is False)
    - ccp_alpha:  Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. (default is 0.0)
    - max_samples:  If bootstrap is True, the number of samples to draw from X to train each base estimator (default is None, which means draw X.shape[0] samples)
    - monotonic_cst:  Monotonic constraints (default is None)
    """

    """
    Project-specific notes for model parameter tuning
    
    """

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state_in,
        n_jobs=-1
        )

    return model


def explain_predictions(pred_data_x, model, home_team, away_team, game_date, today_dt):

    # Build the explainer once, using your trained model
    explainer = shap.TreeExplainer(model)

    # Get SHAP values for the predictions you want to explain
    explanation = explainer(pred_data_x)
    explanation_class1 = explanation[:, :, 1]

    # plot SHAP values
    shap.plots.waterfall(explanation_class1[0], max_display=12, show=False)

    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(f"Predicted Home Win Probability — {away_team} at {home_team}, {game_date}", fontsize=14, pad=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(cons.season_pred_folder.format(date=today_dt) + cons.shap_filename.format(home=home_team, away=away_team, date=game_date))
    plt.close()

    pass