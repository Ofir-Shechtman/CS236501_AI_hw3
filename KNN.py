import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
import utils


class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=None):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.scaler = MinMaxScaler()

    def fit(self, X, y):
        assert len(X) == len(y)
        assert len(X) >= self.k > 0
        self.x_train, self.y_train = check_X_y(X, y)
        self.x_train = self.scaler.fit_transform(self.x_train)

        return self

    def _decision(self, indices, x_test):
        '''
        each neighbor decide which label to vote to
        in this implementation to his y target
        '''
        assert self.y_train is not None
        return np.take(self.y_train, indices)

    def predict(self, x_test):
        assert x_test.shape[1] == self.x_train.shape[1]
        check_is_fitted(self, ['x_train'])
        x_test = check_array(x_test)
        x_test = self.scaler.transform(x_test)
        if len(self.x_train) > self.k:
            dist_matrix = self.euclidean_dist(self.x_train, x_test)
            indices = np.argpartition(dist_matrix, self.k, axis=0)[:self.k].T
        else:
            indices = np.full((len(x_test), self.k), np.arange(self.k))
        nearest_dists = self._decision(indices, x_test)
        y_pred = self.majority(nearest_dists, indices)
        assert len(y_pred) == len(x_test)
        return y_pred

    def euclidean_dist(self, x1, x2):
        x1_square = np.sum(np.square(x1), axis=1)
        x2_square = np.sum(np.square(x2), axis=1)
        ab = np.matmul(x1, x2.T) * -2
        return np.sqrt((ab.T + x1_square).T + x2_square + 1e-7)


    def majority(self, array, kneighbors):
        classes, indices = np.unique(array, return_inverse=True)
        N, K = array.shape
        indices = indices.reshape(N, K)
        binned_indices = np.empty((N, 2))
        for i, idx in enumerate(indices):
            binned_indices[i] = np.bincount(idx, None)

        most_common = classes[np.argmax(binned_indices, axis=1)]
        return most_common


def experiment(**kw):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNNClassifier())])
    parameters = {'knn__k': np.arange(1, 250)}
    X_train, y_train = utils.load_train()
    utils.experiment(pipe, X_train, y_train, parameters, **kw)


def main2():
    pipe, best_params, best_score = experiment(plot=False)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))


def main():
    knn = KNNClassifier(k=1)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))


if __name__ == '__main__':
    main()
