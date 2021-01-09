from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import math


def exp1():
    p = lambda d, r: [math.sin(math.radians(d)) * r, math.cos(math.radians(d)) * r]
    X = np.array(list(map(p, range(0, 360, 40), [2] * 10)) + [[0, 0]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    plot_KNN_vs_ID3(X, y, 'title', k=(3, 4, 5, 7, 8))


def exp2():
    X = np.array([[-2, 0], [2, 0], [4, 0], [2, 2], [0, 4],
                  [-2, 2], [-4, 0], [-2, -2], [0, -4], [2, -2]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    plot_KNN_vs_ID3(X, y, 'tit tyjkf tukf kle', k=range(1,8))


def plot_KNN_vs_ID3(X, y, title, k):
    n_classes = 2
    plot_colors = "rb"
    plot_step = 0.02
    if isinstance(k, int):
        k = [k]
    id3 = DecisionTreeClassifier().fit(X, y)
    knn_s = [(KNeighborsClassifier(k).fit(X, y), f'KNN(k={k})') for k in k]
    models = [(id3, 'ID3')] + knn_s

    for idx, (clf, name) in enumerate(models):

        #fig = plt.subplot(1, len(models), idx + 1)
        fig = plt.subplot(len(models)//5+1, min(len(models), 5), idx + 1)
        fig.title.set_text(name)
        fig.title.set_fontsize(10)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        for i, color, marker in zip(range(n_classes), plot_colors, ('_', '+')):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=str(i),
                        cmap=plt.cm.RdYlBu, s=100, marker=marker, linewidths=2)

    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout(pad=0.1)
    plt.gcf().set_size_inches(min(len(models), 5)*2.5, (len(models)//5+1)*2.5)
    plt.show()


if __name__ == '__main__':
    exp2()
