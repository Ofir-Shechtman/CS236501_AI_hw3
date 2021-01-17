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
        # self.trees = [DecisionTreeClassifier(criterion='entropy', random_state=random_state).fit(X, y) for X, y in samples]
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
    knn_forest = KNNForest(10, 7, p=0)
    X_train, y_train = utils.load_train()
    #parameters = {'N': range(10, 100, 10), 'k': range(7, 100, 5), 'p': np.linspace(0.3, 0.7, num=5)}
    parameters = {'M': [2], 'N': [20, 30, 40, 50, 60], 'p':np.arange(0.5,0.8,0.05), 'k':[15, 20, 25, 30], 'seed':[0]}
    return utils.experiment(knn_forest, X_train, y_train, parameters, plot=False, n_splits=4, **kw)

def main2():
    pipe, best_params, best_score = experiment(verbose=1)
    print(best_params)
    print(best_score)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))



def main():
    knn_forest = KNNForest(N=70, k=20, p=0.5, M=2)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    knn_forest.fit(X_train, y_train)
    # from utils import MALIGNANT, BENIGN
    # from sklearn.metrics import confusion_matrix
    # conf_mat = confusion_matrix(y_test, knn_forest.predict(X_test), labels=[BENIGN, MALIGNANT])
    # print(conf_mat)
    print(knn_forest.score(X_test, y_test))

if __name__ == '__main__':
    main()