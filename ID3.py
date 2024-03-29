from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from abc import ABCMeta
from collections import Counter
import utils


class ID3(BaseEstimator, ClassifierMixin):
    def __init__(self, M=2):
        self.M = M
        self._tree = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        assert self.M >= 2

        self._tree = self._generate_tree(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['_tree'])
        assert X.ndim == 2
        X = check_array(X)
        prediction = []
        for i in range(len(X)):
            answer = self._decision(self._tree, X[i])
            prediction.append(answer)
        return np.array(prediction)

    @staticmethod
    def _decision(root, row):
        node = root
        while not node.is_leaf:
            node = node.next(row[node.attr])
        return node.label

    class Node(metaclass=ABCMeta):
        def __init__(self, is_leaf):
            super().__init__()
            self.is_leaf = is_leaf

    class Leaf(Node):
        def __init__(self, label):
            super().__init__(is_leaf=True)
            self.label = label

    class ContinuousNode(Node):
        def __init__(self, attr, threshold):
            super().__init__(is_leaf=False)
            self.attr = attr
            self.threshold = threshold
            self.less = None
            self.greater = None

        def next(self, value):
            if value < self.threshold:
                return self.less
            else:
                return self.greater

    def _generate_tree(self, X, y):
        assert len(X) == len(y)
        assert len(X) > 0
        if len(np.unique(y)) == 1:  # num of labels
            return self.Leaf(label=y[0])
        assert len(np.unique(y)) == 2
        assert X.ndim == 2
        assert y.ndim == 1
        if len(X) < self.M or len(np.unique(X, axis=0))==1:  # min_samples_split, label is the most common
            most_common = Counter(y).most_common(1)[0]
            return self.Leaf(label=most_common[0])

        node = self._split_attribute(X, y)
        less = np.where(X[:, node.attr] < node.threshold)
        greater = np.where(X[:, node.attr] >= node.threshold)
        node.less = self._generate_tree(X.take(less, axis=0)[0], y.take(less)[0])
        node.greater = self._generate_tree(X.take(greater, axis=0)[0], y.take(greater)[0])
        return node

    def _split_attribute(self, X, y):
        assert len(X) == len(y)
        assert len(X) > 1
        assert len(np.unique(y)) == 2
        assert X.shape[1] > 1

        features = np.apply_along_axis(self._find_threshold, axis=0, arr=X, y=y)
        best_feature = np.argmax(features[0])
        threshold = features[1, best_feature]
        return self.ContinuousNode(best_feature, threshold)

    def _find_threshold(self, x, y):
        assert len(x) == len(y)
        assert len(x) > 1
        assert len(np.unique(y)) == 2
        if len(np.unique(x)) == 1:
            return -1*float('inf'), 0
        assert len(np.unique(x)) > 1
        u = np.unique(x)
        u.sort()
        arr = (u[1:] + u[:-1]) / 2
        v_split = np.vectorize(self._split_by_threshold, signature='(),(n),(n)->(2)')
        entropy_tresh_all = v_split(arr, x, y)
        best_tresh_idx = np.argmax(entropy_tresh_all[:, 0])
        return entropy_tresh_all[best_tresh_idx]

    def _split_by_threshold(self, threshold, x, y):
        assert x.ndim == y.ndim == 1
        gain_before = entropy(y)
        less = np.where(x < threshold)
        greater = np.where(x >= threshold)
        less_label = y.take(less)[0]
        greater_label = y.take(greater)[0]
        less_gain = len(less_label) / len(y) * entropy(less_label)
        greater_gain = len(greater_label) / len(y) * entropy(greater_label)
        return np.array([gain_before - less_gain - greater_gain, threshold])

def entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(2)).sum()


def experiment(**kw):
    # run without extra parameters for default behavior
    id3 = ID3()
    parameters = {'M': [2, 5, 10, 20, 50, 100, 150, 200, 300]}
    X_train, y_train = utils.load_train()
    utils.experiment(id3, X_train, y_train, parameters, **kw)


def main():
    id3 = ID3()
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    id3.fit(X_train, y_train)
    print(id3.score(X_test, y_test))


if __name__ == '__main__':
    main()
