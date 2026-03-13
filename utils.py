from sklearn.ensemble import RandomForestRegressor

import constants as cons
import pandas as pd


def team_info():
    
    while True:
        try:
            # Fetch the teams
            teams_df = pd.DataFrame(cons.nhl_client.teams.teams())
            break
        except Exception as ex:
            print(f'\t\t... {ex} ...')
            continue
        
    teams_df.rename(columns={'name': 'teamName'}, inplace=True)

    teams_df['conferenceName'] = teams_df['conference'].apply(lambda x: x['name'])
    teams_df['divisionName'] = teams_df['division'].apply(lambda x: x['name'])

    teams_df = teams_df[['teamName', 'conferenceName', 'divisionName']]

    return teams_df


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