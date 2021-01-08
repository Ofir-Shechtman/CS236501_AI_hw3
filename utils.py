import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np

MY_ID = 206180374
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TO_PREDICT = 'diagnosis'

def random_state(seed=MY_ID):
    return np.random.RandomState(seed=seed)

def _load(input_file, target):
    df = pd.read_csv(input_file)
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def load_train():
    return _load(TRAIN_FILE, TO_PREDICT)


def load_test():
    return _load(TEST_FILE, TO_PREDICT)


def experiment(model, X_train, y_train, parameters,
               random_state=MY_ID,
               n_splits=5,
               verbose=0,
               scoring=None,
               refit=True,
               plot=True):
    if plot:
        assert len(parameters) == 1
    np.random.seed(random_state)
    cv = KFold(shuffle=True, random_state=random_state, n_splits=n_splits)
    grid_search = GridSearchCV(model, parameters, cv=cv, verbose=verbose, scoring=scoring, refit=refit)
    grid_search.fit(X_train, y_train)
    dump(grid_search, 'KNNForest_cv.joblib')
    if plot:
        hp_name = list(parameters.keys())[0]
        plot_grid_search(grid_search.cv_results_, hp_name)
        best_params = grid_search.best_params_.get(hp_name)
    else:
        best_params = grid_search.best_params_
    return grid_search.best_estimator_, best_params, grid_search.best_score_


def plot_grid_search(cv_results, name_param):
    name_param = f'param_{name_param}'
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    grid_param = cv_results.get(name_param)
    ax.plot(grid_param, scores_mean, '-o', label=name_param)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()
