import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import ID3
from joblib import dump, load

input_file = "test.csv"

# comma delimited is the default
df = pd.read_csv(input_file)
est = KBinsDiscretizer(n_bins=343, encode='ordinal', strategy='uniform')
y = df.diagnosis
X = df.iloc[:, 1:-1]
#id3 = ID3.ID3()
#id3.fit(X, y)
#dump(id3, 'id3.joblib')
id3 = load('id3.joblib')
print(id3.score(X, y))

# cross_val_score(id3, X, y, cv=10)
import pickle
