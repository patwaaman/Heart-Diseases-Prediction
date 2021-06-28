import pandas as pd
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration " + str(i) + " " + str(cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def my_predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def fit(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = my_predict(w, b, X_test)
    Y_prediction_train = my_predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train, "w": w,
         "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}
    return d

df = pd.read_csv('cleveland.csv', header = None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_conf_test = y_test
y_conf_train = y_train
from sklearn.preprocessing import StandardScaler as ss

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], -1).T

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], -1).T

y_train = np.array(y_train)
y_train = y_train.reshape((y_train.shape[0]), 1).T

y_test = np.array(y_test)
y_test = y_test.reshape((y_test.shape[0], 1)).T

d = fit(X_train, y_train, X_test, y_test, 2000, 0.005, True)

y_pred = d["Y_prediction_test"]
y_pred = np.reshape(y_pred, (y_pred.shape[1], 1))
y_pred = y_pred.tolist()

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_conf_test)

y_pred_train = d["Y_prediction_train"]
y_pred_train = np.reshape(y_pred_train, (y_pred_train.shape[1], 1))
y_pred_train = y_pred_train.tolist()
cm_train = confusion_matrix(y_pred_train, y_conf_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / y_train.shape[1]))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / y_test.shape[1]))
