from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import KNN
import ID3
import numpy as np

import utils


def sensitive_loss(y_true, y_predicted):
    sick, healthy = 'M', 'B'
    conf_mat = confusion_matrix(y_true, y_predicted, normalize='true', labels=[healthy, sick])
    FN = conf_mat[1][0]
    FP = conf_mat[0][1]
    return 0.1 * FP + FN


if __name__ == '__main__':
    weights = {'M': 10, 'B': 1}
    sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)
    best_k = KNN.experiment(verbose=0, scoring=sensitive_scorer, weights=weights)

    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNN.KNNClassifier(k=best_k, weights=weights))])
    print(best_k)
    X_train, y_train = utils.load_train()
    pipe.fit(X_train, y_train)
    X_test, y_test = utils.load_test()
    y_predicted = pipe.predict(X_test)
    print(sensitive_loss(y_test, y_predicted))
