import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from minmax import Minmax

def bin_pred(X):
    Y_hat = mlr.predict_(X)

    Y_hat[Y_hat >= 0.5] = 1.
    Y_hat[Y_hat <  0.5] = 0.
    return Y_hat

def get_accuracy(mlr, X, Y):
    Y_hat = bin_pred(X)
    good = np.sum(Y_hat == Y)
    total = Y.shape[0]

    return good / total

def get_X_Y():
    path_x = "day04/ex09/solar_system_census.csv"
    path_y = "day04/ex09/solar_system_census_planets.csv"

    data_x = pd.read_csv(path_x)
    data_y = pd.read_csv(path_y)

    X = np.array(data_x[["height","weight","bone_density"]]).reshape(-1,3)
    Y = np.array(data_y["Origin"]).reshape(-1,1)


    prep = Minmax()
    X = prep.fit(X).apply(X)

    return X, Y

def get_bin_Y(Y, feature):
    k = -1
    Y[(Y == feature)] = k
    Y[(Y != k)] = 0.
    Y[(Y == k)] = 1.
    return Y

def multi_scatter(X, Y_hat, Y):
    plot_dim = 2
    fig, axs_ = plt.subplots(plot_dim, plot_dim)
    axs = []
    for sublist in axs_:
        for item in sublist:
            axs.append(item)
    
    # Y[(Y == 1.)] = 0.9
    # Y[(Y == 0.)] = 0.1

    
    sns.scatterplot(ax=axs[0], x=X[:,0].reshape(-1), y=X[:,1].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[1], x=X[:,0].reshape(-1), y=X[:,2].reshape(-1), hue=Y_hat.reshape(-1))
    sns.scatterplot(ax=axs[2], x=X[:,1].reshape(-1), y=X[:,2].reshape(-1), hue=Y_hat.reshape(-1))

    # for idx_feature, feature in enumerate(X.T):
    #     axs[idx_feature].scatter(feature, Y_hat, s=1, c='r', label=f"Pred")
    #     axs[idx_feature].scatter(feature, Y, s=1, c='b', label="True")
    #     axs[idx_feature].legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", 
            "--zipcode", 
            help="Solar citizens origin area", 
            type=float, 
            choices=[0., 1. , 2., 3.],
            required=True)
    args = parser.parse_args()

    X, Y = get_X_Y()
    Y = get_bin_Y(Y, args.zipcode)
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)


    theta = [0] * (X.shape[1] + 1)
    mlr = MyLogisticRegression(theta, alpha=1e-2, max_iter=100000)
    mlr.fit_(X_train, Y_train)

    acc = get_accuracy(mlr, X_test, Y_test)
    print(f"Accuracy is {acc}")

    Y_hat = bin_pred(X)
    multi_scatter(X, Y_hat, Y)