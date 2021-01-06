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
    #ID3.experiment(3)
    KNN.experiment()

