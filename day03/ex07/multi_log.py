import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)

from my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from minmax import Minmax

def multi_pred(mlrs, X):
    Y_hats = []
    for i, mlr in enumerate(mlrs):
        Y_hat = mlr.predict_(X)
        Y_hats.append(Y_hat)
        
    new_arr = np.stack(Y_hats)
    out = np.argmax(new_arr, axis=0)
    return out


def bin_pred(mlr, X):
    Y_hat = mlr.predict_(X)

    Y_hat[Y_hat >= 0.5] = 1.
    Y_hat[Y_hat <  0.5] = 0.
    return Y_hat

def get_accuracy(mlr, X, Y):
    Y_hat = bin_pred(mlr, X)
    good = np.sum(Y_hat == Y)
    total = Y.shape[0]

    return good / total

def get_mult_accuracy(mlrs, X, Y):
    Y_hat = multi_pred(mlrs, X)
    good = np.sum(Y_hat == Y)
    total = Y.shape[0]

    return good / total

def get_X_Y():
    path_x = "day03/ex07/solar_system_census.csv"
    path_y = "day03/ex07/solar_system_census_planets.csv"

    data_x = pd.read_csv(path_x)
    data_y = pd.read_csv(path_y)

    X = np.array(data_x[["height","weight","bone_density"]]).reshape(-1,3)
    Y = np.array(data_y["Origin"]).reshape(-1,1)


    prep = Minmax()
    X = prep.fit(X).apply(X)

    return X, Y

def get_bin_Y(Y, feature):
    Y_ = Y.copy()
    Y_[(Y == feature)] = 1.
    Y_[(Y != feature)] = 0.
    return Y_

def multi_scatter(X, Y_hat, Y):
    plot_dim = 2
    fig, axs_ = plt.subplots(plot_dim, plot_dim)
    axs = []
    for sublist in axs_:
        for item in sublist:
            axs.append(item)
    
    sns.scatterplot(ax=axs[0], x=X[:,0].reshape(-1), y=X[:,1].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[1], x=X[:,0].reshape(-1), y=X[:,2].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[2], x=X[:,1].reshape(-1), y=X[:,2].reshape(-1), hue=Y_hat.reshape(-1))

    plt.show()

if __name__ == "__main__":
    X, Y = get_X_Y()


    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.7)
    theta = [0] * (X.shape[1] + 1)

    mlrs = []
    for i in range(4):
        mlr_i = MyLogisticRegression(theta, alpha=1e-2, max_iter=100000)
        Y_train_i = get_bin_Y(Y_train, float(i))
        mlr_i.fit_(X_train, Y_train_i)

        Y_test_i = get_bin_Y(Y_test, float(i))
        acc = get_accuracy(mlr_i, X_test, Y_test_i)
        print(f"Accuracy for class {i} is {acc}")
        mlrs.append(mlr_i)


    Y_hat = multi_pred(mlrs, X)
    acc = get_mult_accuracy(mlrs, X_test, Y_test)
    print(f"Accuracy is {acc}")
    multi_scatter(X, Y_hat, Y)