from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import KNNForest
import utils




def experiment(**kw):
    pipe = KNNForest.KNNForest(10, 7, p=None)
    X_train, y_train = utils.load_train()
    parameters = {'M': range(2, 20, 5), 'N': range(10, 60, 5), 'k': range(7, 60, 5), 'p':[None, 0.3, 0.5, 0.7]}
    return utils.experiment(pipe, X_train, y_train, parameters, plot=False, n_splits=5, **kw)


def main2():
    pipe, best_params, best_score = experiment(verbose=0)
    print(best_params)
    print(best_score)
    X_test, y_test = utils.load_test()
    print(pipe.score(X_test, y_test))

def main():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn_forest', KNNForest.KNNForest(N=25, k=7, M=2))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))

if __name__ == '__main__':
    main2()
