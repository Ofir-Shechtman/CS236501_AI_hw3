from sklearn.tree import DecisionTreeClassifier

import KNN
import ID3
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
# from scipy.stats import mode
import utils
from joblib import dump, load


class KNNForest(KNN.KNNClassifier):
    def __init__(self, N, k, p=None, random_state=utils.random_state(), M=0, metric=None):
        super().__init__(k=k)
        self.N = N
        self.M = M
        self.metric = metric
        self.random_state = random_state
        self.p = p


    def fit(self, X, y):
        assert self.N > 0
        assert 0 < self.k < self.N
        #assert 0.3 <= self.weights.all() <= 0.7
        assert len(X) == len(y)
        X, y = check_X_y(X, y)
        if self.p:
            self.weights = np.full(self.N, self.p)
        else:
            self.weights = np.linspace(0.3, 0.7, num=self.N)

        samples = [sample(X, y, p, self.random_state) for p in self.weights]
        self.x_train = np.stack([np.mean(X, axis=0) for X, y in samples])
        # self.trees = [ID3.ID3(self.M, metric=self.metric).fit(X, y) for X, y in samples]
        self.trees = [DecisionTreeClassifier(criterion='entropy', random_state=self.random_state).fit(X, y) for X, y in
                      samples]
        return self

    def _decision(self, indices_mat, x_test):
        '''
        each neighbor decide which label to vote to
        '''
        trees_decision = np.empty(indices_mat.shape, dtype=str)
        for sample, (indices, x) in enumerate(zip(indices_mat, x_test)):
            x = np.expand_dims(x_test[sample], axis=0)
            for tree_idx, tree in enumerate(indices):
                trees_decision[sample, tree_idx] = self.trees[tree].predict(x)[0]
        return trees_decision


def sample(X, y, p, random_state):
    assert len(X) == len(y)
    mask = random_state.choice([True, False], len(X), p=[p, 1 - p])
    return X[mask], y[mask]


def experiment():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn_forest', KNNForest(10, 7, p=None, metric=None))])
    X_train, y_train = utils.load_train()
    parameters = {'M': [2, 20], 'N': [10, 20], 'k': [7, 9, 17], 'p':np.linspace(0.3, 0.7, num=5)}
    parameters = {'knn_forest__' + k: v for k, v in parameters.items()}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=3)


if __name__ == '__main__':
    pipe, best_params, best_score = experiment()
    print(best_params)
    print(best_score)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))