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

def experiment():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn_forest', KNNForest.KNNForest(10, 7, p=None, metric=None))])
    X_train, y_train = utils.load_train()
    parameters = {'M': [2, 20], 'N': [10, 20], 'k': [7, 9, 17]}
    parameters = {'knn_forest__' + k: v for k, v in parameters.items()}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=3)


if __name__ == '__main__':
    pipe, best_params, best_score = experiment()
    print(best_params)
    print(best_score)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))

