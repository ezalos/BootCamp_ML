import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from my_linear_regression import MyLinearRegression as MyLR
path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features
path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from polynomial_train import continuous_plot
path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter


def model_save(data, poly):
    path = os.path.join(os.path.dirname(__file__), f"model_{poly}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def model_load(poly):
    path = os.path.join(os.path.dirname(__file__), f"model_{poly}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def continuous_plot(x, y, i, lr):
	continuous_x = np.arange(1,7.01, 0.01).reshape(-1,1)
	x_ = add_polynomial_features(continuous_x, i)
	y_hat = lr.predict(x_)
	plt.scatter(x.T[0],y)
	plt.plot(continuous_x, y_hat, color='orange')
	plt.show()

def one_loop(X, Y, poly=1):
    X_poly = add_polynomial_features(X, poly)
    X_train, X_test, Y_train, Y_test = data_spliter(X_poly, Y, 0.8)

    theta = [0] * (poly * X.shape[1] + 1)
    alpha=1 / (100 * (10 ** poly))

    lr = MyLR(thetas=theta, alpha=alpha, max_iter=50000)

    lr.fit_(X_train, Y_train)
    model_save(lr, poly)
    continuous_plot(X_poly, Y, poly, lr)
    cost = lr.cost_(Y_test, lr.predict(X_test))
    print(f"{cost = }")
    return cost


if __name__ == "__main__":
    data = pd.read_csv("day02/ex10/space_avocado.csv")

    X = np.array(data[["weight","prod_distance","time_delivery"]]).reshape(-1,3)
    Y = np.array(data["target"]).reshape(-1,1)

    cost = []
    for i in range(1, 5):
        c = one_loop(X, Y, poly=i)
        cost.append(c)


    # my_lreg.multi_scatter(X,Y)
