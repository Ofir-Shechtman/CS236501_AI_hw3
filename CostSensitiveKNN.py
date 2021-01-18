from sklearn.metrics import make_scorer, confusion_matrix
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
        assert self.k <= self.w1
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


def experiment(**kw):
    # run without extra parameters for default behavior
    wknn = CostSensitiveKNN(k=None, w1=None, w2=None)
    parameters = {'k': np.arange(1, 20), 'w1': np.arange(1, 20), 'w2': np.arange(0, 2, 0.1)}
    X_train, y_train = utils.load_train()
    sensitive_scorer = make_scorer(sensitive_loss, greater_is_better=False)
    return utils.experiment(wknn, X_train, y_train, parameters, scoring=sensitive_scorer, **kw)




def main():
    wknn = CostSensitiveKNN(k=5, w1=5, w2=0.9)
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    wknn.fit(X_train, y_train)
    y_predicted = wknn.predict(X_test)
    print(sensitive_loss(y_test, y_predicted))


if __name__ == '__main__':
    main()
