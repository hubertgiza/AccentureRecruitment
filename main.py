import pandas as pd
import numpy as np
import sys
from impyute.imputation.cs import fast_knn, mice
from data_functions import data_preprocessing, one_hot_encoding, data_normalization, get_groups_from_json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

COLUMNS = ['target', 'sex', 'dob', 'daily_commute', 'friends_number', 'relationship_status', 'education',
           'hobbies', 'location_population', 'groups']


def train_random_forest(X, Y, estimators, classes, features):
    train_x = X[0:3600, :]
    train_y = Y[0:3600]
    validate_x = X[3600:, :]
    validate_y = Y[3600:]
    clf = RandomForestClassifier(n_estimators=estimators)
    clf.n_classes_ = classes
    clf.n_features_ = features
    clf.fit(train_x, train_y)
    predictions = clf.predict(validate_x).reshape(400, 1)
    result = sum([1 if predictions[i] == validate_y[i] else 0 for i in range(validate_y.shape[0])]) / \
             validate_y.shape[0] * 100
    return result


def train_ada_boost(X, Y, estimators, classes, features):
    train_x = X[0:3600, :]
    train_y = Y[0:3600]
    validate_x = X[3600:, :]
    validate_y = Y[3600:]
    clf = AdaBoostClassifier(n_estimators=estimators)
    clf.n_classes_ = classes
    clf.n_features_ = features
    clf.fit(train_x, train_y)
    predictions = clf.predict(validate_x).reshape(400, 1)
    result = sum([1 if predictions[i] == validate_y[i] else 0 for i in range(validate_y.shape[0])]) / \
             validate_y.shape[0] * 100
    return result


def train_logistic_regression(X, Y, classes, features):
    train_x = X[0:3600, :]
    train_y = Y[0:3600]
    validate_x = X[3600:, :]
    validate_y = Y[3600:]
    clf = LogisticRegression()
    clf.n_classes_ = classes
    clf.n_features_ = features
    clf.fit(train_x, train_y)
    predictions = clf.predict(validate_x).reshape(400, 1)
    result = sum([1 if predictions[i] == validate_y[i] else 0 for i in range(validate_y.shape[0])]) / \
             validate_y.shape[0] * 100
    return result


def train_neural_network(X, Y, epochs, features, classes):
    X_train = X[:3200, :]
    Y_train = one_hot_encoding(classes, Y[:3200])
    X_val = X[3200:3600, :]
    Y_val = one_hot_encoding(classes, Y[3200:3600])
    X_test = X[3600:4000, :]
    Y_test = one_hot_encoding(classes, Y[3600:4000])

    model = models.Sequential()

    model.add(layers.Dense(18, activation='relu', input_shape=(features,)))
    model.add(layers.Dense(18, activation='relu'))
    model.add(layers.Dense(9, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=256, validation_data=(X_val, Y_val))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.evaluate(X_test, Y_test)

    # results = model.predict(X_test)
    # Y_predictions = [0 if values[0] > values[1] else 1 for values in results]
    # sns.heatmap(confusion_matrix(Y[3600:4000], Y_predictions), annot=True)
    # plt.show()


sys.setrecursionlimit(100000)
a = pd.read_csv("Case_Assignment/train.csv")
a = a.sort_values(['user_id'])
data_preprocessing(a)
data_normalization(a)
a = pd.concat([a, get_groups_from_json('train')], axis=1)
X = a[COLUMNS].to_numpy()
np.random.shuffle(X)
X, Y = X[:, 1:], X[:, 0]
X2 = mice(X).round()
X = fast_knn(X, 400).round()

# train_neural_network(X, Y, 20, 9, 2)
N = 100
estimators = 400
random_forest_results = []
random_forest_results2 = []
ada_boost_results = []
logistic_regression_results = []
for i in range(N):
    random_forest_results.append(train_random_forest(X, Y, estimators, 2, 9))
    random_forest_results2.append(train_random_forest(X2, Y, estimators, 2, 9))
    # ada_boost_results.append(train_ada_boost(X, Y, estimators, 2, 9))
    # logistic_regression_results.append(train_logistic_regression(X, Y, 2, 9))
fig = plt.figure()
plt.plot(range(N), random_forest_results, '.-r', label='KNN data')
plt.plot(range(N), random_forest_results2, 'o-b', label='Mice data')
plt.legend()
# plt.plot(range(N), random_forest_results, 'o-b')
# plt.plot(range(N), logistic_regression_results, '+-y')
plt.show()
