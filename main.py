import pandas as pd
import numpy as np
import sys
from impyute.imputation.cs import mice
from impyute.imputation.cs import fast_knn
from data_functions import data_preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt


def train_random_forest(X, Y, estimators):
    train_x = X[0:3600, :]
    train_y = Y[0:3600]
    validate_x = X[3600:, :]
    validate_y = Y[3600:]
    clf = RandomForestClassifier(n_estimators=estimators)
    clf.n_classes_ = 2
    clf.n_features_ = 6
    clf.fit(train_x, train_y)
    predictions = clf.predict(validate_x).reshape(400, 1)
    result = sum([1 if predictions[i] == validate_y[i] else 0 for i in range(validate_y.shape[0])]) / \
             validate_y.shape[0] * 100
    return result


def train_ada_boost(X, Y, estimators):
    train_x = X[0:3600, :]
    train_y = Y[0:3600]
    validate_x = X[3600:, :]
    validate_y = Y[3600:]
    clf = AdaBoostClassifier(n_estimators=estimators)
    clf.n_classes_ = 2
    clf.n_features_ = 6
    clf.fit(train_x, train_y)
    predictions = clf.predict(validate_x).reshape(400, 1)
    result = sum([1 if predictions[i] == validate_y[i] else 0 for i in range(validate_y.shape[0])]) / \
             validate_y.shape[0] * 100
    return result


sys.setrecursionlimit(100000)
a = pd.read_csv("Case_Assignment/train.csv")
a = a.sort_values(['user_id'])
data_preprocessing(a)
X = a[['target', 'sex', 'dob', 'daily_commute', 'friends_number', 'relationship_status', 'education']].to_numpy()
np.random.shuffle(X)
X, Y = X[:, 1:], X[:, 0]
X3 = fast_knn(X, 400).round()

N = 100
n_models = 10
estimators = 100
random_forest_results = []
ada_boost_results = []
for i in range(N):
    random_forest_results.append(train_random_forest(X3, Y, estimators))
    ada_boost_results.append(train_ada_boost(X3, Y, estimators))
fig = plt.figure()
plt.plot(range(N), ada_boost_results, '.-r')
plt.plot(range(N), random_forest_results, 'o-b')
plt.show()
