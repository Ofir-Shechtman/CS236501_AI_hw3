from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
# from scipy.stats import mode
import utils


class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=None, weights=None):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None
        self.weights = weights

    def fit(self, X, y):
        assert len(X) == len(y)
        assert len(X) > self.k
        assert self.k > 0
        self.x_train, self.y_train = check_X_y(X, y)
        self.n_classes = len(np.unique(self.y_train))
        return self

    def predict(self, x_test: np.ndarray):
        assert x_test.shape[1] == self.x_train.shape[1]
        x_test = check_array(x_test)
        check_is_fitted(self, ['x_train', 'y_train', 'n_classes'])
        dist_matrix = euclidean_dist(self.x_train, x_test)
        indices = np.argpartition(dist_matrix, self.k, axis=0)[:self.k]
        nearest_dists = np.take(self.y_train, indices)
        y_pred = mode(nearest_dists, self.weights)
        assert len(y_pred) == len(x_test)
        return y_pred


def mode(array, weights_dict=None):
    classes, indices = np.unique(array, return_inverse=True)
    K, N = array.shape
    indices = indices.reshape(K, N).T
    if weights_dict:
        keys = classes
        values = np.array([weights_dict[classes[0]], weights_dict[classes[1]]])
        sidx = keys.argsort()
        weights = values[sidx[np.searchsorted(keys, array, sorter=sidx)]]
        weights = weights.T
    else:
        weights = [None]*N
    binned_indices = np.empty((N, 2))
    for i, (idx, weight) in enumerate(zip(indices, weights)):
        binned_indices[i] = np.bincount(idx, weight)

    most_common = classes[np.argmax(binned_indices, axis=1)]
    return most_common


def euclidean_dist(x1, x2):
    x1_square = np.diagonal(np.matmul(x1, x1.T))
    x2_square = np.diagonal(np.matmul(x2, x2.T))

    ab = np.matmul(x1, x2.T) * -2
    assert ((ab.T + x1_square).T + x2_square).all() >= 0
    return np.sqrt((ab.T + x1_square).T + x2_square + 1e-7)


def experiment(verbose=0, range=np.arange(1, 250), scoring=None, weights=None):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier(weights=weights))])
    X_train, y_train = utils.load_train()
    return utils.experiment(pipe, X_train, y_train, 'knn__k', range, verbose=verbose, scoring=scoring)


if __name__ == '__main__':
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier(8))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
