from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid

import KNN
import ID3
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
# from scipy.stats import mode
import KNNForest
import utils


def gini(labels):
    """Compute Gini coefficient of array of values"""
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    score = 1 - sum([i ** 2 for i in norm_counts])
    return score


if __name__ == '__main__':
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn_forest', KNNForest.KNNForest(9, 5, 0.6, 0, metric=gini))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    #clf = KNNForest.KNNForest(N=15, k=9, p=0.6, M=1, metric=None).fit(X_train, y_train)
    cv = KFold(shuffle=True, random_state=206180374, n_splits=5)
    parameters = {'M': [1, 10], 'N': [10, 15], 'k': [5, 9], 'p': [0.5, 0.6], 'metric':[gini, None]}
    parameters = {'knn_forest__' + k: v for k, v in parameters.items()}
    grid_search = GridSearchCV(pipe, parameters, cv=cv, verbose=2, scoring=None, n_jobs=1)
    print(pipe.get_params().keys())
    print(list(ParameterGrid(parameters)))
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.cv_results_)
    # model = SelectFromModel(pipe, prefit=True)
