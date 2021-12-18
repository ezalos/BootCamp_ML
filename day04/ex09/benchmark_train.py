import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from polynomial_model import add_polynomial_features

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from minmax import Minmax

from other_metrics import f1_score_
from confusion_matrix import confusion_matrix_

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


def metricos(y_hat, y, label):
    TP = ((y == label) & (y_hat == label)).sum()
    TN = ((y != label) & (y_hat != label)).sum()

    FP = ((y != label) & (y_hat == label)).sum()
    FN = ((y == label) & (y_hat != label)).sum()
    
    return (TP, TN, FP, FN)


def f1(y_hat, y, label):
    # y and yhat for 1 class
    TP, TN, FP, FN = metricos(y_hat, y, label)

    # print(f"\t{TP = } {TN = } {FP = } {FN = }")

    return TP / (TP + (1/2) * (FP + FN))


def weighted_f1(y_hat, y):
    funo = 0
    for i in range(4):
        label = float(i)
        funo += f1(y_hat, y, label) * np.sum(y == label)
    funo = funo / y.shape[0]
    return funo

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

def model_save(data, poly, lambd):
    path = os.path.join(os.path.dirname(__file__), f"model_p{poly}_l{lambd}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def model_load(poly, lambd):
    path = os.path.join(os.path.dirname(__file__), f"model_p{poly}_l{lambd}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class OneVsAll():
    def __init__(self, alpha=1e-3, max_iter=10000, lambda_=0.5) -> None:
        self.mlrs = []
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, Y):
        self.mlrs = []
        for i in range(4):
            label = float(i)

            Y_i = get_bin_Y(Y, label)
            theta = [0] * (X.shape[1] + 1)
            mlr_i = MyLogisticRegression(
                    theta,
                    alpha=self.alpha,
                    max_iter=self.max_iter, 
                    penalty='l2')
            mlr_i.fit_(X, Y_i, lambda_=self.lambda_)

            Y_hat_bin = bin_pred(mlr_i, X)
            # Y_hat_bin = mlr_i.predict_(X)
            # print(f"{Y_hat_bin = }")
            # print(f"\tLabel {label} f1: {f1(Y_hat_bin, Y_i, 1)}")

            self.mlrs.append(mlr_i)

    def predict(self, X):
        Y_hat = multi_pred(self.mlrs, X)
        return Y_hat

    def f1(self, X, Y):
        Y_hat = self.predict(X)
        return weighted_f1(Y_hat, Y)


if __name__ == "__main__":
    X, Y = get_X_Y()

    for pol in range(3, 4):
        X_poly = add_polynomial_features(X, pol)

        X_train, X_test, Y_train, Y_test = data_spliter(X_poly, Y, 0.7)
        alpha=1e-1
        max_iter=20000
        for lambda_ in [0., 0.1, 0.3, 0.5, 0.7, 1.]:
            mlrs = []
            ova = OneVsAll(alpha=alpha, max_iter=max_iter, lambda_=lambda_)
            ova.fit(X_train, Y_train)
            print(f"Pol {pol} Lambda {lambda_} f1: {ova.f1(X_test, Y_test)}")
            model_save(ova, pol, lambda_)