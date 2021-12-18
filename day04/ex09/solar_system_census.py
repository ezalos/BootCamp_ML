from benchmark_train import OneVsAll, get_X_Y
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from polynomial_model import add_polynomial_features
from data_spliter import data_spliter
import os

def model_load(poly, lambd):
    path = os.path.join(os.path.dirname(__file__), f"model_p{poly}_l{lambd}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def nice_3d_plot(X, Y, Y_hat):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ["blue", "violet", "green", "orange"]
    height = X[:, 0].reshape(-1, 1)
    weight = X[:, 1].reshape(-1, 1)
    bone_density = X[:, 2].reshape(-1, 1)
    for i in range(4):
        label = float(i)
        index_y = np.where(np.all(Y == label, axis = 1))
        ax.scatter(height[index_y], weight[index_y], bone_density[index_y], facecolors='none', edgecolors=colors[i], s=90, label = f"truth for house {i}")
        index_hat = np.where(np.all(Y_hat == label, axis = 1))
        ax.scatter(height[index_hat], weight[index_hat], bone_density[index_hat], color=colors[i], s=80, label = f"predictions for house {i}")
    ax.set_xlabel('height')
    ax.set_ylabel('weight')
    ax.set_zlabel('bone_density')
    ax.set_title("Best Model Predictions Vs Truth")
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.show()
    

if __name__ == "__main__":
    X, Y = get_X_Y()

    for pol in range(3, 4):
        X_poly = add_polynomial_features(X, pol)

        X_train, X_test, Y_train, Y_test = data_spliter(X_poly, Y, 0.7)
        theta = [0] * (X.shape[1] + 1)
        alpha=1e-2
        max_iter=20000
        f1s = []
        lambdas = [0., 0.1, 0.3, 0.5, 0.7, 1.]
        for lambda_ in lambdas:
            ova = model_load(pol, lambda_)
            ova.fit(X_train, Y_train)
            print(f"Pol {pol} Lambda {lambda_} f1: {ova.f1(X_test, Y_test)}")
            if lambda_ != 0.5 or pol != 3:
                best = ova
            f1s.append(ova.f1(X_test, Y_test))

    sns.barplot(x=lambdas, y=f1s)
    plt.show()


    nice_3d_plot(X, Y, ova.predict(X_poly))
