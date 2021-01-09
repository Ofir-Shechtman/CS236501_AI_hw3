from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import KNN
import numpy as np
import utils
from utils import SICK, HEALTHY


def sensitive_loss(y_true, y_predicted):
    conf_mat = confusion_matrix(y_true, y_predicted, normalize='true', labels=[HEALTHY, SICK])
    FN = conf_mat[1][0]
    FP = conf_mat[0][1]
    return 0.1 * FP + FN


def experiment(weights=None, **kw):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNN.KNNClassifier(weights=weights))])
    parameters = {'knn__k': np.arange(1, 250)}
    X_train, y_train = utils.load_train()
    sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)
    return utils.experiment(pipe, X_train, y_train, parameters, scoring=sensitive_scorer, **kw)


def main2():
    weights = {SICK: 10, HEALTHY: 1}
    X_test, y_test = utils.load_test()
    for w in [None, weights]:
        pipe, best_params, best_score = experiment(w, plot=True)
        y_predicted = pipe.predict(X_test)
        print(f'k={best_params}')
        print(sensitive_loss(y_test, y_predicted))
        print(pipe.score(X_test, y_test))

def main():
    weights = {SICK: 10, HEALTHY: 1}
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNN.KNNClassifier(k=14, weights=weights))])
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    pipe.fit(X_train, y_train)
    y_predicted = pipe.predict(X_test)
    print(sensitive_loss(y_test, y_predicted))


if __name__ == '__main__':
    main()
