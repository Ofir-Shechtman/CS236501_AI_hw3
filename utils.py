import math
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

MY_ID = 206180374
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
UNUSED_FILE = "unused.csv"
ALL = "all.csv"
TO_PREDICT = 'diagnosis'
MALIGNANT, BENIGN = 1, 0


def random_state(seed=MY_ID):
    return np.random.RandomState(seed=seed)


def _load(input_file, target):
    df = pd.read_csv(input_file)
    X = df.drop(target, axis=1)
    X = X.iloc[:, ::-1]  # reverse columns order
    diagnosis_map = {'B': BENIGN, 'M': MALIGNANT}
    y = df[target].map(diagnosis_map)
    return X, y


def load_train():
    return _load(TRAIN_FILE, TO_PREDICT)


def load_test():
    return _load(TEST_FILE, TO_PREDICT)


def load_unused():
    return _load(UNUSED_FILE, TO_PREDICT)


def experiment(model, X_train, y_train, parameters,
               random_state=MY_ID,
               n_splits=5,
               verbose=0,
               scoring=None,
               refit=True,
               plot=False,
               n_jobs=1):
    np.random.seed(random_state)
    cv = KFold(shuffle=True, random_state=random_state, n_splits=n_splits)
    grid_search = GridSearchCV(model, splitter(parameters), cv=cv, verbose=verbose, scoring=scoring, refit=refit,
                               n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    '''import pickle
    with open('grid_search3.pickle', 'wb') as handle:
        cv_results = grid_search.cv_results_
        param_names = [key for key in cv_results.keys() if key.startswith('param_')]
        param_values = [np.unique(cv_results.get(p)) for p in param_names]
        n_splits = len(['_' for key in cv_results.keys() if key.startswith('split')])
        cv_results.update({'n_splits': n_splits,
                           'parameters': {param_name: values for param_name, values in zip(param_names, param_values)}})
        pickle.dump(cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
    if plot and isinstance(parameters, dict) and len(parameters) == 1:
        hp_name = list(parameters.keys())[0]
        plot_grid_search_single(grid_search.cv_results_, hp_name)
        best_params = grid_search.best_params_.get(hp_name)
    else:
        best_params = grid_search.best_params_
    return grid_search.best_estimator_, best_params, grid_search.best_score_

def splitter(parameters):
    return Mk_split(Mk_split(parameters, 'N', 'k'), 'w1', 'k')


def Mk_split(parameters, N_param, k_param):
    if N_param in parameters and k_param in parameters:
        param_list = []
        N_list = parameters.pop(N_param)
        k_list = parameters.pop(k_param)
        for N in N_list:
            parameters.update({k_param: [k for k in k_list if k <= N], N_param: [N]})
            param_list.append(parameters.copy())
        return param_list
    else:
        return parameters


def plot_grid_search_single(cv_results, name_param):
    name_param = f'param_{name_param}'
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    # Plot Grid search scores
    _, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    grid_param = cv_results.get(name_param)
    ax.plot(grid_param, scores_mean, '-o', label=name_param)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()

def cv_results_2_df(cv_results, score, where):
    param_names = [key for key in cv_results.keys() if key.startswith('param_') or key == score]
    params = [cv_results.get(key) for key in param_names]

    df = pd.DataFrame({k: v for k, v in zip(param_names, params)})
    for p, v in where.items():
        df = df.where(df['param_'+p]==v)
    return df

def plot_grid_search(cv_results, score, param1, param2=None, where={}, **kw):
    score = score + '_test_score'
    df = cv_results_2_df(cv_results, score, where)
    index = ['param_' + param1]
    columns = ['param_' + param2] if param2 else []
    scores_mean = pd.pivot_table(df, values=score, index=index,
                                 columns=columns, aggfunc=np.mean)
    ylabel = score.replace('_', ' ')
    title = f'{ylabel} of {param1}'
    if param2:
        title += f' vs {param2}'
    if where:
        title += f', where {where}'
    title = title[0].upper() + title[1:]
    scores_mean.plot(ylabel=ylabel, title=title, figsize=(12, 5), **kw)
    plt.show()

def plot_grid_search_compare(cv_results1, cv_results2, score, param, where={}, **kw):
    score = score + '_test_score'
    df1 = cv_results_2_df(cv_results1, score, where)
    df2 = cv_results_2_df(cv_results2, score, where)
    df = pd.concat([df1, df2], keys=['regular', 'improve'], names=['forest type'])
    index = ['param_' + param]
    columns = ['forest type']
    scores_mean = pd.pivot_table(df, values=score, index=index,
                                 columns=columns, aggfunc=np.mean).dropna()
    ylabel = score.replace('_', ' ')
    title = f'{ylabel} of {param}'
    if where:
        title += f', where {where}'
    title = title[0].upper() + title[1:]
    scores_mean.plot(ylabel=ylabel, title=title, figsize=(12, 5), **kw)
    plt.show()


def np_map(array, d):
    keys = np.array(list(d.keys()))
    values = np.array(list(d.values()))
    sidx = keys.argsort()
    weights = values[sidx[np.searchsorted(keys, array, sorter=sidx)]]
    return weights


def add_keys_prefix(prefix, parameters):
    if isinstance(parameters, dict):
        return {prefix + k: v for k, v in parameters.items()}
    elif isinstance(parameters, list):
        return {prefix + k: v for k, v in parameters.items() for parameters in parameters}


remove_keys_prefix = lambda parameters: {k.split('__')[-1]: v for k, v in parameters.items()}
