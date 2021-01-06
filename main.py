from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import ID3
import KNN
from joblib import dump, load

import utils
from dataset import load_train, load_test
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

FIT = False

if __name__ == '__main__':
    X_train, y_train = load_train()
    #print(y_train, y_train.mode())
    #id3 = ID3.ID3()
    #id3.fit(X_train, y_train)


    X_test, y_test = load_test()
    if FIT:
        id3 = ID3.ID3()
        id3.fit(X_train, y_train)
        dump(id3, 'id3.joblib')
    else:
        id3 = load('id3.joblib')
    print(id3.score(X_test, y_test))

    grid_search = id3 = load('grid_search.joblib')
    #utils.plot_grid_search(grid_search.cv_results_, 'M')
    ID3.experiment()
    KNN.experiment()

