from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import KNN
import numpy as np
import utils
from utils import MALIGNANT, BENIGN


class CostSensitiveKNN(KNN.KNNClassifier):

    def __init__(self, k, w1, w2):
        super().__init__(k=k)
        self.w1 = w1
        self.w2 = w2

    def fit(self, X, y):
        assert self.k < self.w1
        super().fit(X, y)

    @property
    def weights_majority(self):
        return {MALIGNANT: self.w1, BENIGN: 1}

    @property
    def weights_dist(self):
        return {MALIGNANT: self.w2, BENIGN: 1}

    def euclidean_dist(self, x1, x2):
        euclidean_mat = super().euclidean_dist(x1, x2)
        w = utils.np_map(self.y_train, self.weights_dist)
        return (euclidean_mat.T * w).T

    def majority(self, array, kneighbors):
        classes, indices = np.unique(array, return_inverse=True)
        N, K = array.shape
        indices = indices.reshape(N, K)
        weights = utils.np_map(array, self.weights_majority)
        binned_indices = np.empty((N, 2))
        for i, (idx, weight) in enumerate(zip(indices, weights)):
            binned_indices[i] = np.bincount(idx, weight)

        most_common = classes[np.argmax(binned_indices, axis=1)]
        return most_common

def sensitive_loss(y_true, y_predicted):
    conf_mat = confusion_matrix(y_true, y_predicted, normalize='true', labels=[BENIGN, MALIGNANT])
    FN = conf_mat[1][0]
    FP = conf_mat[0][1]
    return 0.1 * FP + FN

sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)

def experiment(weights=None, **kw):
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNN.KNNClassifier(weights=weights))])
    parameters = {'knn__k': np.arange(1, 250)}
    X_train, y_train = utils.load_train()
    sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)
    return utils.experiment(pipe, X_train, y_train, parameters, scoring=sensitive_scorer, **kw)


def main2():
    weights = {MALIGNANT: 20, BENIGN: 1}
    X_test, y_test = utils.load_test()
    for w in [None, weights]:
        pipe, best_params, best_score = experiment(w, plot=True)
        y_predicted = pipe.predict(X_test)
        print(f'k={best_params}')
        print(sensitive_loss(y_test, y_predicted))
        print(pipe.score(X_test, y_test))

def find_w():
    pipe = CostSensitiveKNN(1, 1, 1)
    parameters = {'k': range(1,40,5), 'w1':range(1,40,5), 'w2':np.arange(1.0, 2.0, 0.01)}
    #parameters = {'k': [9], 'w1':[9], 'w2':[1.09]}
    X_train, y_train = utils.load_train()
    sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)
    pipe, best_params, best_score = utils.experiment(pipe, X_train, y_train, parameters, scoring=sensitive_scorer, plot=False, n_splits=9, verbose=1)
    X_test, y_test = utils.load_test()
    y_predicted = pipe.predict(X_test)
    print(f'k={best_params}')
    print(sensitive_loss(y_test, y_predicted))
    print(pipe.score(X_test, y_test))

def find_w2():
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    d=[]
    for k in np.arange(1, 50):
        for w1 in np.arange(1, 50):
            for w2 in np.arange(1, 2, 0.01):
                pipe = CostSensitiveKNN(k=k,w1=w1, w2=w2)
                pipe.fit(X_train, y_train)
                y_predicted = pipe.predict(X_test)
                d.append((k,w1, w2,sensitive_loss(y_test, y_predicted)))
    m = min([loss for _, _, _, loss in d])
    print([t for t in d if t[3]==m])
    #print(min(d, key=lambda t:t[2]))


def main():
    wknn = CostSensitiveKNN(k=9, w1=9, w2=1.09)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    wknn.fit(X_train, y_train)
    y_predicted = wknn.predict(X_test)
    print(sensitive_loss(y_test, y_predicted))


if __name__ == '__main__':
    find_w()
