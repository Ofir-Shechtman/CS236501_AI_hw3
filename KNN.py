from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
#from scipy.stats import mode
import utils


class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=None):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def fit(self, X, y):
        assert len(X) == len(y)
        assert len(X) >= self.k
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
        y_pred = mode(nearest_dists, axis=0)
        assert len(y_pred) == len(x_test)
        return y_pred


def euclidean_dist(x1, x2):
    x1_square = np.diagonal(np.matmul(x1, x1.T))
    x2_square = np.diagonal(np.matmul(x2, x2.T))
    ab = np.matmul(x1, x2.T) * -2
    return np.sqrt((ab.T + x1_square).T + x2_square)

def mode(array, axis = 0):
    uniques, indices = np.unique(array, return_inverse=True)

    args_for_bincount_fn = None, np.max(indices) + 1
    binned_indices = np.apply_along_axis(np.bincount,
                                         axis,
                                         indices.reshape(array.shape),
                                         *args_for_bincount_fn)

    most_common = uniques[np.argmax(binned_indices, axis=axis)]
    return most_common

def experiment(verbose=0):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier())])
    X_train, y_train = utils.load_train()
    utils.experiment(pipe, X_train, y_train, 'knn__k', range(1, 50), verbose=verbose)

if __name__ == '__main__':
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier(1))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
