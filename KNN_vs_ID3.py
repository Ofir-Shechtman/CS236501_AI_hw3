import numpy as np
import matplotlib.pyplot as plt
import math
from ID3 import ID3
from KNN import KNNClassifier


class KNN_vs_ID3:

    @classmethod
    def plot_example1(cls):
        title = "Correct ID3, Wrong KNN for every k"
        X = np.array([[2, 2], [-2, -2], [2, -2]])
        y = np.array([0, 1, 0])
        cls.plot(X, y, title, k=range(1, len(y) + 1))

    @classmethod
    def plot_example2(cls):
        k = 1
        title = "Wrong ID3, Correct KNN with k={}".format(k)
        p = lambda d, r: [math.sin(math.radians(d)) * r, math.cos(math.radians(d)) * r]
        X = np.array(list(map(p, range(0, 360, 40), [2] * 10)) + [[0, 0]])
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        cls.plot(X, y, title, k=k)

    @classmethod
    def plot_example3(cls):
        k = 3
        title = "Wrong ID3, Wrong KNN with k={}".format(k)
        X = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2],
                      [-2, -1], [1, 2], [-1, -2], [2, 1], [-1.5, -1.5]])
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
        cls.plot(X, y, title, k=k)

    @classmethod
    def plot_example4(cls):
        title = "Correct ID3, Correct KNN for every k"
        X = np.array([[1, 0], [-1, 0]])
        y = np.array([0, 1])
        cls.plot(X, y, title, k=range(1, len(y)))

    @staticmethod
    def plot(X, y, title, k):
        n_classes = 2
        plot_colors = "br"
        plot_step = 0.02
        if isinstance(k, int):
            k = [k]
        id3 = ID3().fit(X, y)
        knn_s = [(KNNClassifier(k).fit(X, y), f'KNN(k={k})') for k in k]
        models = [(id3, 'ID3')] + knn_s

        for idx, (clf, name) in enumerate(models):

            # fig = plt.subplot(1, len(models), idx + 1)
            fig = plt.subplot(len(models) // 5 + 1, min(len(models), 5), idx + 1)
            fig.title.set_text(name)
            fig.title.set_fontsize(10)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = 1-Z.reshape(xx.shape)
            Z[0][0] = 0 #contourf fix
            Z[0][1] = 1 #contourf fix
            plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

            for i, color, marker in zip(range(n_classes), plot_colors, ('+', '_')):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, label=str(i),
                            cmap=plt.cm.RdYlBu, s=100, marker=marker, linewidths=2)

        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(top=0.5)
        plt.tight_layout(pad=3 if len(models) <= 5 else 1)
        plt.gcf().set_size_inches(min(len(models), 5) * 2.5, (len(models) // 5) * 2.5 + 3)
        plt.show()


if __name__ == '__main__':
    KNN_vs_ID3.plot_example3()
