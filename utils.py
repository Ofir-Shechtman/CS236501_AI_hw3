import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt

MY_ID = 206180374
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TO_PREDICT = 'diagnosis'

def _load(input_file, target):
    df = pd.read_csv(input_file)
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def load_train():
    return _load(TRAIN_FILE, TO_PREDICT)

def load_test():
    return _load(TEST_FILE, TO_PREDICT)


def experiment(pipe, X_train, y_train, hp_name, hp_values, **kw):
    random_state = kw.get('random_state', MY_ID)
    n_splits = kw.get('n_splits', 5)
    verbose = kw.get('verbose', 0)
    scoring = kw.get('scoring', None)

    cv = KFold(shuffle=True, random_state=random_state, n_splits=n_splits)
    parameters = {hp_name: hp_values}
    grid_search = GridSearchCV(pipe, parameters, cv=cv, verbose=verbose, scoring=scoring)
    grid_search.fit(X_train, y_train)
    plot_grid_search(grid_search.cv_results_, hp_name)
    return grid_search.best_params_.get(hp_name)


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