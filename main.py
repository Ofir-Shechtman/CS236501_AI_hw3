import ID3
import KNN
import CostSensitiveKNN
import KNNForest
import improvedKNNForest
import warnings
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt


# Calling Method
from joblib import dump, load
parameters = {'M': [2, 5, 10], 'N': range(20, 150, 10), 'p':np.arange(0.3,0.7,0.05), 'k':range(20, 150, 10)}
import pickle
with open('imp_knn_forest_results.pickle', 'rb') as handle:
    imp_knn_forest_results = pickle.load(handle)
with open('knn_forest_results.pickle', 'rb') as handle:
    knn_forest_results = pickle.load(handle)
utils.plot_grid_search_compare(knn_forest_results, imp_knn_forest_results, 'mean', 'p', grid=True, where={'M':2, 'N':20, 'k':20})
if __name__ == '__main__':

    #utils.KNN_vs_ID3.plot_example4()
    '''print('__ID3__')
    ID3.main()
    ID3.experiment()
    print('__KNN__')
    KNN.main()
    #KNN.experiment()
    print('__CostSensitiveKNN__')
    CostSensitiveKNN.main()
    print('__KNNForest__')
    KNNForest.main2()
    KNNForest.main()
    print('__improvedKNNForest__')
    improvedKNNForest.main2()
    improvedKNNForest.main()'''