import KNNForest
import utils
import numpy as np

class improvedKNNForest(KNNForest.KNNForest):
    def __init__(self, N, k, p, seed=utils.MY_ID, M=2):
        super().__init__(N=N, k=k, seed=seed, p=p, M=M)

    @property
    def weights(self):
        return np.linspace(0.3, 0.7, num=self.N)

    def sample(self, X, y, p, random_state):
        assert len(X) == len(y)
        mask = random_state.choice(np.arange(X.shape[1]), int(X.shape[1]*self.p), replace=False)
        Xc = X.copy()
        Xc[:, mask] = 0
        mask = random_state.choice([True, False], len(Xc), p=[p, 1 - p])
        return Xc[mask], y[mask]


def experiment(**kw):
    # run without extra parameters for default behavior
    pipe = improvedKNNForest(10, 7, p=0)
    X_train, y_train = utils.load_train()
    parameters = {'M': [2, 5, 10, 30], 'N': range(20, 150, 10), 'p': np.arange(0.0, 1.0, 0.1).round(2), 'k': range(20, 150, 10)}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=5, **kw)

def main():
    imp_knn_forest = improvedKNNForest(N=100, k=100, p=0.6)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    imp_knn_forest.fit(X_train, y_train)
    print(imp_knn_forest.score(X_test, y_test))

if __name__ == '__main__':
    main()

