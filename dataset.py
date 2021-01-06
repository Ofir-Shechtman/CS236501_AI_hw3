import pandas as pd


def _load(input_file, target):
    df = pd.read_csv(input_file)
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def load_train():
    return _load("train.csv", 'diagnosis')

def load_test():
    return _load("test.csv", 'diagnosis')


