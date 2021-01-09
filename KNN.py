from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
import utils


class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=None, weights=None):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.weights = weights

    def fit(self, X, y):
        assert len(X) == len(y)
        assert len(X) > self.k
        assert self.k > 0
        self.x_train, self.y_train = check_X_y(X, y)

        return self

    def _decision(self, indices, x_test):
        '''
        each neighbor decide which label to vote to
        in this implementation to his y target
        '''
        assert self.y_train is not None
        return np.take(self.y_train, indices)

    def predict(self, x_test: np.ndarray):
        assert x_test.shape[1] == self.x_train.shape[1]
        x_test = check_array(x_test)
        check_is_fitted(self, ['x_train'])
        dist_matrix = euclidean_dist(self.x_train, x_test)
        indices = np.argpartition(dist_matrix, self.k, axis=0)[:self.k].T
        nearest_dists = self._decision(indices, x_test)
        y_pred = majority(nearest_dists, indices, self.weights)
        assert len(y_pred) == len(x_test)
        return y_pred


def majority(array, kneighbors, weights=None):
    classes, indices = np.unique(array, return_inverse=True)
    N, K = array.shape
    indices = indices.reshape(N, K)
    if isinstance(weights, dict):  # weights by value
        keys = classes
        values = np.array([weights[classes[0]], weights[classes[1]]])
        sidx = keys.argsort()
        weights = values[sidx[np.searchsorted(keys, array, sorter=sidx)]]
    elif isinstance(weights, np.ndarray):  # weights by neighbor
        weights = weights[kneighbors]
    else:
        weights = [None] * N
    binned_indices = np.empty((N, 2))
    for i, (idx, weight) in enumerate(zip(indices, weights)):
        binned_indices[i] = np.bincount(idx, weight)

    most_common = classes[np.argmax(binned_indices, axis=1)]
    return most_common


def euclidean_dist(x1, x2):
    x1_square = np.diagonal(np.matmul(x1, x1.T))
    x2_square = np.diagonal(np.matmul(x2, x2.T))
    ab = np.matmul(x1, x2.T) * -2
    return np.sqrt((ab.T + x1_square).T + x2_square + 1e-7)


def experiment(**kw):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier())])
    parameters = {'knn__k': np.arange(1, 250)}
    X_train, y_train = utils.load_train()
    return utils.experiment(pipe, X_train, y_train, parameters, **kw)


def main2():
    pipe, best_params, best_score = experiment(plot=False)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))


def main():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier(k=1))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))


if __name__ == '__main__':
    main()
