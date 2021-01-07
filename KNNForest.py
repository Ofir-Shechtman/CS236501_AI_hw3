import KNN
import ID3
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
# from scipy.stats import mode
import utils


class KNNForest(KNN.KNNClassifier):
    def __init__(self, N, K, p, M=0, **kw):
        super().__init__(k=K)
        assert N > 0
        assert 0 < K <= N
        assert 0.3 <= p <= 0.7
        self.N = N
        self.p = p
        self.M = M
        self.kw = kw

    def fit(self, X, y):
        assert len(X) == len(y)
        X, y = check_X_y(X, y)
        samples = [sample(X, y, self.p) for _ in range(self.N)]
        self.x_train = np.stack([np.mean(X, axis=0) for X, y in samples])
        self.trees = [ID3.ID3(self.M, **self.kw).fit(X, y) for X, y in samples]

    def _decision(self, indices_mat, x_test):
        '''
        each neighbor decide which label to vote to
        '''
        trees_decision = np.empty(indices_mat.shape, dtype=str)
        for sample, (indices, x) in enumerate(zip(indices_mat, x_test)):
            for tree_idx, tree in enumerate(indices):
                trees_decision[sample, tree_idx] = self.trees[tree].predict(x_test[sample])
        return trees_decision


def sample(X, y, p):
    assert len(X) == len(y)
    mask = np.random.choice([True, False], len(X), p=[p, 1 - p])
    return X[mask], y[mask]


if __name__ == '__main__':
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn_forest', KNNForest(9, 5, 0.6, 0))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    #model = SelectFromModel(pipe, prefit=True)

    print(pipe.score(X_test, y_test))