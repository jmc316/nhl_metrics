import pandas as pd
import constants as cons
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def make_predictions(data_df, oob_list, mse_list, rsq_list, load_model=False, save_model=True):

    # data_df.drop(columns=[cons.game_id_col, cons.game_time_col, cons.venue_timezone_col, cons.game_type_col, cons.season_name_col], inplace=True)
    # data_df.drop(columns=[cons.game_date_col], inplace=True)

    data_df[cons.game_date_col] = pd.to_datetime(data_df[cons.game_date_col]).dt.date
    # data_df[cons.season_name_col] = data_df[cons.season_name_col].astype(str)

    label_encoder = LabelEncoder()
    categorical_df = data_df.select_dtypes(include=['object', 'str']).apply(label_encoder.fit_transform)
    numerical_df = data_df.select_dtypes(exclude=['object', 'str'])
    encoded_df = pd.concat([numerical_df, categorical_df], axis=1)
    encoded_df.replace({None: np.nan}, inplace=True) 

    x_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), encoded_df.columns.difference(cons.predict_cols)]
    y_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), cons.predict_cols]
    x_predict_df = encoded_df.loc[encoded_df[cons.home_team_score_col].isna(), encoded_df.columns.difference(cons.predict_cols)]

    if load_model:
        model = pd.read_pickle(f'{cons.model_files_folder}{cons.sklearn_model_filename}')
    else:
        model = init_model()

        model.fit(x_train_df.values, y_train_df.values)

        oob_score = model.oob_score_
        # print(f'Out-of-Bag Score: {oob_score}')
        oob_list.append(oob_score)

    if save_model:
        pd.to_pickle(model, f'{cons.model_files_folder}{cons.sklearn_model_filename}')

    trainset_predictions = model.predict(x_train_df.values)

    mse = mean_squared_error(y_train_df.values, trainset_predictions)
    # print(f'Mean Squared Error: {mse}')
    mse_list.append(mse)

    r2 = r2_score(y_train_df.values, trainset_predictions)
    # print(f'R-squared: {r2}')
    rsq_list.append(r2)

    predictset_predictions = model.predict(x_predict_df.values)

    predict_df = data_df[data_df[cons.last_period_col].isna()]
    predict_df[cons.predict_cols] = predictset_predictions
    predict_df['awayTeamScore_int'] = predict_df[cons.away_team_score_col].round().astype(int)
    predict_df['homeTeamScore_int'] = predict_df[cons.home_team_score_col].round().astype(int)
    predict_df[cons.last_period_col] = np.where(predict_df['homeTeamScore_int'] == predict_df['awayTeamScore_int'], 'OT', 'REG')

    data_df.update(predict_df[cons.predict_cols])

    # importances = model.feature_importances_
    # importance_df = pd.DataFrame({'feature': x_train_df.columns, 'importance': importances})
    # print(importance_df.sort_values(by='importance', ascending=False))

    return data_df


def init_model():

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
    - max_features:  The number of features to consider when looking for the best split (default is 'auto')
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

    model = RandomForestRegressor(n_estimators=100, oob_score=True)

    return model