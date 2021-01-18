from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import KNN
import ID3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y
import numpy as np
import utils


class KNNForest(KNN.KNNClassifier):
    def __init__(self, N, k, p, seed=0, M=0):
        super().__init__(k=k)
        self.N = N
        self.M = M
        self.seed = seed
        self.p = p

    @property
    def weights(self):
        return np.full(self.N, self.p)

    def fit(self, X, y):
        assert self.N > 0
        assert 0 < self.k <= self.N
        assert len(X) == len(y)
        X, y = check_X_y(X, y)
        X = self.scaler.fit_transform(X)
        random_state = utils.random_state(self.seed)
        samples = [self.sample(X, y, p, random_state) for p in self.weights]
        self.x_train = np.stack([np.mean(X, axis=0) for X, y in samples])
        self.trees = [ID3.ID3(self.M).fit(X, y) for X, y in samples]
        return self

    def _decision(self, indices_mat, x_test):
        '''
        each neighbor decide which label to vote to
        '''
        trees_decision = np.empty(indices_mat.shape)
        for sample, (indices, x) in enumerate(zip(indices_mat, x_test)):
            x = np.expand_dims(x_test[sample], axis=0)
            for tree_idx, tree in enumerate(indices):
                trees_decision[sample, tree_idx] = self.trees[tree].predict(x)[0]
        return trees_decision

    def majority(self, array, kneighbors):
        classes, indices = np.unique(array, return_inverse=True)
        N, K = array.shape
        indices = indices.reshape(N, K)
        weights = self.weights[kneighbors]
        binned_indices = np.empty((N, 2))
        for i, (idx, weight) in enumerate(zip(indices, weights)):
            binned_indices[i] = np.bincount(idx, weight)

        most_common = classes[np.argmax(binned_indices, axis=1)]
        return most_common

    @staticmethod
    def sample(X, y, p, random_state):
        assert len(X) == len(y)
        mask = random_state.choice([True, False], len(X), p=[p, 1 - p])
        return X[mask], y[mask]


def experiment(**kw):
    # run without extra parameters for default behavior
    knn_forest = KNNForest(10, 7, p=0.3)
    X_train, y_train = utils.load_train()
    parameters = {'M': [2, 10, 30], 'N': range(20, 150, 10), 'p':np.arange(0.3,0.75,0.05).round(2), 'k':range(20, 150, 10)}
    return utils.experiment(knn_forest, X_train, y_train, parameters, plot=False, n_splits=5, **kw)


def main():
    knn_forest = KNNForest(N=50, k=20, p=0.35, M=2)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    knn_forest.fit(X_train, y_train)
    print(knn_forest.score(X_test, y_test))

if __name__ == '__main__':
    main()