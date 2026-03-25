import numpy as np
import pandas as pd
import constants as cons

from file_utils import pklLoad, pklSave
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def make_predictions(data_df, oob_list, mse_list, rsq_list, set_model_random_state, debug=False, load_model=True, save_model=False):

    # encode categorical variables using label encoding, and keep numerical variables as is
    label_encoder = LabelEncoder()
    categorical_df = data_df.select_dtypes(include=['object', 'str']).apply(label_encoder.fit_transform)
    numerical_df = data_df.select_dtypes(exclude=['object', 'str'])
    encoded_df = pd.concat([numerical_df, categorical_df], axis=1)
    encoded_df.replace({None: np.nan}, inplace=True) 

    # split the data into training and prediction sets based on the presence of the target variable (home team score)
    x_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), encoded_df.columns.difference(cons.predict_cols)]
    y_train_df = encoded_df.loc[encoded_df[cons.home_team_score_col].notna(), cons.predict_cols]
    x_predict_df = encoded_df.loc[encoded_df[cons.home_team_score_col].isna(), encoded_df.columns.difference(cons.predict_cols)]

    # initialize or load the model, fit it to the training data, and make predictions on the prediction set
    if load_model:
        if debug: print('\t\tLoading existing model...')
        model = pklLoad(cons.model_files_folder, cons.sklearn_model_filename)
    else:
        if debug: print('\t\tCreating new model...')
        if set_model_random_state:
            model = init_model(random_state_in=42)
        else:
            model = init_model()

        model.fit(x_train_df.values, y_train_df.values)

    # save the model to a file for future use
    if save_model:
        pklSave(model, cons.model_files_folder, cons.sklearn_model_filename)

    # make predictions on the training set to calculate metrics
    if debug: print('\t\tMaking predictions...')
    trainset_predictions = model.predict(x_train_df.values)

    trainset_metrics(model, y_train_df, trainset_predictions, oob_list, mse_list, rsq_list, debug)

    # make predictions on the prediction set
    predictset_predictions = model.predict(x_predict_df.values)

    # update the original data_df with the predictions for the target variables and determine the last period based on the predicted scores
    predict_df = data_df[data_df[cons.last_period_col].isna()]
    predict_df[cons.predict_cols] = predictset_predictions
    predict_df[cons.last_period_col] = np.where(abs(predict_df[cons.home_team_score_col] - predict_df[cons.away_team_score_col]) < cons.ot_score_diff, 'OT', 'REG')
    data_df.update(predict_df[cons.predict_cols])

    # check for ties (this is a simplification and could be improved with a more sophisticated approach)
    predicted_ties_df = predict_df[predict_df[cons.home_team_score_col] == predict_df[cons.away_team_score_col]]
    if not predicted_ties_df.empty:
        # randomly assign a winner for each tie
        predicted_ties_df[cons.home_team_score_col] += np.random.choice([-.5, .5], size=len(predicted_ties_df))
        data_df.update(predicted_ties_df[cons.predict_cols])

    # importances = model.feature_importances_
    # importance_df = pd.DataFrame({'feature': x_train_df.columns, 'importance': importances})
    # print(importance_df.sort_values(by='importance', ascending=False))

    return data_df


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

    model = RandomForestRegressor(n_estimators=100,
                                  random_state=random_state_in,
                                  oob_score=True
                                  )

    return model


def trainset_metrics(model, y_train_df, trainset_predictions, oob_list, mse_list, rsq_list, debug=False):
    
    if debug: print('\t\tcalculating metrics...')

    oob_score = model.oob_score_
    if debug: print(f'\t\t\tOut-of-Bag Score: {oob_score}')
    oob_list.append(oob_score)

    mse = mean_squared_error(y_train_df.values, trainset_predictions)
    if debug: print(f'\t\t\tMean Squared Error: {mse}')
    mse_list.append(mse)

    r2 = r2_score(y_train_df.values, trainset_predictions)
    if debug: print(f'\t\t\tR-squared: {r2}')
    rsq_list.append(r2)

    games_correct, total_games, accuracy = game_outcome_metrics(y_train_df, trainset_predictions)
    if debug: print(f'\t\t\tGames Correct: {games_correct}/{total_games} ({accuracy:.2%})')

    return oob_list, mse_list, rsq_list


def game_outcome_metrics(y_train_df, trainset_predictions):

    game_results_df = pd.DataFrame({
        'home_team_score_actual': y_train_df[cons.home_team_score_col],
        'home_team_score_predicted': trainset_predictions[:, 0],
        'away_team_score_actual': y_train_df[cons.away_team_score_col],
        'away_team_score_predicted': trainset_predictions[:, 1]
    })

    game_results_df['correct_outcome'] = np.where(
        (game_results_df['home_team_score_actual'] > game_results_df['away_team_score_actual']) &
        (game_results_df['home_team_score_predicted'] > game_results_df['away_team_score_predicted']) |
        (game_results_df['home_team_score_actual'] < game_results_df['away_team_score_actual']) &
        (game_results_df['home_team_score_predicted'] < game_results_df['away_team_score_predicted']) |
        (game_results_df['home_team_score_actual'] == game_results_df['away_team_score_actual']) &
        (game_results_df['home_team_score_predicted'] == game_results_df['away_team_score_predicted']),
        1, 0
        )
    
    return (sum(game_results_df['correct_outcome']), len(game_results_df), sum(game_results_df['correct_outcome']) / len(game_results_df))
