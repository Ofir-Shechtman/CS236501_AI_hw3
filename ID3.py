import math
from typing import Tuple, List, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import pandas as pd
import numpy as np
from abc import ABCMeta

import utils


class ID3(BaseEstimator, ClassifierMixin):
    def __init__(self, M=0):
        self.M = M
        self._tree = None

    def fit(self, X, y):
        # X, y = check_X_y(X, y)

        self._tree = self._generate_tree(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['_tree'])
        # X = check_array(X)
        prediction = []
        for i in range(len(X)):
            answer = self._decision(self._tree, X.iloc[i])
            prediction.append(answer)
        return prediction

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
            self.children = []

        def next(self, value):
            if value < self.threshold:
                return self.children[0]
            else:
                return self.children[1]

    def _generate_tree(self, X: pd.DataFrame, y: pd.Series) -> Node:
        assert len(X) == len(y)
        assert len(X) > 0
        if y.nunique() == 1:  # num of labels
            return self.Leaf(label=y.iloc[0])
        assert y.nunique() == 2
        if len(X.columns) == 1 or len(X) < self.M:  # last attr, label is the most common
            return self.Leaf(label=y.mode().iloc[0])

        node, split = self._split_attribute(X, y)
        for indices in split:
            node.children.append(self._generate_tree(X.loc[indices], y.loc[indices]))
        return node

    @classmethod
    def _split_attribute(cls, X: pd.DataFrame, y: pd.Series) -> Tuple[ContinuousNode, Tuple[List[Any], List[Any]]]:
        assert len(X) == len(y)
        assert len(X) > 1
        assert y.nunique() == 2
        assert len(X.columns) > 1

        attributes_list = [(attribute, cls._find_threshold(X[attribute], y)) for attribute in X]
        attribute, (best_attr, best_threshold, split) = max(attributes_list, key=lambda t: t[1][0])
        return cls.ContinuousNode(attribute, best_threshold), split

    @classmethod
    def _find_threshold(cls, x: pd.Series, y: pd.Series) -> Tuple[float, float, Tuple[List[Any], List[Any]]]:
        assert len(x) == len(y)
        assert len(x) > 1
        assert y.nunique() == 2
        assert x.nunique() > 1

        x.sort_values(ascending=True)
        all_threshold = []
        for i in range(0, len(x) - 1):
            if x.iloc[i] != x.iloc[i + 1]:
                threshold = (x.iloc[i] + x.iloc[i + 1]) / 2
                less = x.where(x <= threshold).dropna().index
                greater = x.where(x > threshold).dropna().index
                ent = gain(y, (less, greater))
                all_threshold.append((ent, threshold, (less, greater)))
        return max(all_threshold, key=lambda t: t[0])


def gain(y, indices_list):
    assert len(indices_list) == 2
    assert len(y) == len(indices_list[0]) + len(indices_list[1])
    # calculate impurity before split
    gain_before = entropy(y)
    # calculate impurity after split
    gain_after = sum([(len(indices) / len(y)) * entropy(y.loc[indices]) for indices in indices_list])

    return gain_before - gain_after


def entropy(s):
    probs = [count / len(s) for count in s.value_counts()]
    return -1 * sum([p * math.log(p, 2) for p in probs])


def experiment():
    id3 = ID3()
    X_train, y_train = utils.load_train()
    utils.experiment(id3, X_train, y_train, 'M', [1, 5, 10, 50, 100])


if __name__ == '__main__':
    id3 = ID3()
    X_train, y_train = utils.load_train()
    X_test, y_test = utils.load_test()
    id3.fit(X_train, y_train)
    print(id3.score(X_test, y_test))
