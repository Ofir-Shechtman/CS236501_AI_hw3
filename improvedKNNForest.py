from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect, chi2
import KNNForest
import utils
import numpy as np

class improvedKNNForest(KNNForest.KNNForest):
    def __init__(self, N, k, p, p2, seed=utils.MY_ID, M=2):
        super().__init__(N=N, k=k, seed=seed, p=p, M=M)
        self.p2=p2

    #@property
    #def weights(self):
    #    return np.linspace(0.3, 0.7, num=self.N)

    def sample(self, X, y, p, random_state):
        assert len(X) == len(y)
        mask = random_state.choice(np.arange(X.shape[1]), int(X.shape[1]*self.p2), replace=False)
        Xc = X.copy()
        Xc[:, mask] = 0
        mask = random_state.choice([True, False], len(Xc), p=[p, 1 - p])
        return Xc[mask], y[mask]



'''    @staticmethod
    def sample(X, y, p, random_state):
        assert len(X) == len(y)
        mask = random_state.choice([True, False], len(X), p=[p, 1 - p])

        mask2 = random_state.choice(np.arange(X.shape[1]), int(X.shape[1]*1))
        return X[mask][:, mask2], y[mask]'''


def experiment(**kw):
    pipe = improvedKNNForest(10, 7, p=0, p2=0)
    X_train, y_train = utils.load_train()
    #parameters = {'N': range(10, 100, 10), 'k': range(7, 100, 5)}
    parameters = {'M': [2], 'N': [20, 30, 40, 50, 60], 'p':np.arange(0.5,0.7,0.05), 'k':[15, 20, 25, 30], 'p2':np.arange(0.1,0.4,0.05), 'seed':[0, utils.MY_ID]}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=4, **kw)

def experiment2(**kw):
    from sklearn.ensemble import ExtraTreesClassifier
    pipe = RandomForestClassifier()
    X_train, y_train = utils.load_train()
    parameters = {'n_estimators': range(50,90,5), 'min_samples_split': [2,3,4], 'max_samples':[0.7, 0.8, 0.9, None], 'criterion':['gini', 'entropy']}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=7, **kw)


def main2():
    pipe, best_params, best_score = experiment(verbose=1, n_jobs=-1)
    print(best_params)
    print(best_score)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))

def main():
    #imp_knn_forest = improvedKNNForest(N=25, k=7, M=2)
    imp_knn_forest = improvedKNNForest(N=90, k=90, p=0.4, p2=0.6)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    imp_knn_forest.fit(X_train, y_train)

    # y_pred = imp_knn_forest.predict(X_test)
    # w = np.where(y_pred!=y_test)

    print(imp_knn_forest.score(X_test, y_test))

if __name__ == '__main__':
    main2()

