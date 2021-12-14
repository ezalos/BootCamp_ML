import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from my_linear_regression import MyLinearRegression as MyLR
path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features


def continuous_plot(x, y, i, lr):
	# Build the model:
	# Plot:
	## To get a smooth curve, we need a lot of data points
	continuous_x = np.arange(1,7.01, 0.01).reshape(-1,1)
	x_ = add_polynomial_features(continuous_x, i)
	y_hat = lr.predict(x_)
	# print(x.shape, y.shape)
	plt.scatter(x.T[0],y)
	plt.plot(continuous_x, y_hat, color='orange')
	plt.show()


def one_loop(Xpill, Yscore, i):
	# print(Xpill)
	x = add_polynomial_features(Xpill, i)
	# print(x)
	if i == 4:
		theta = [-20, 160, -80, 10, -1]
	elif i == 5:
		theta = [1140, -1850, 1110, -305, 40, -2]
	elif i == 6:
		theta = [9110, -18015, 13400, -4935, 966, -96.4, 3.86]
	else:
		theta = [0] * (i + 1)

	if i == 5:
		alpha = 1e-8
	elif i == 6:
		alpha = 1e-9
	else:
		alpha=1 / (100 * (10 ** i))

	lr = MyLR(thetas=theta, alpha=alpha, max_iter=50000)
	lr.fit_(x, Yscore)
	continuous_plot(x, Yscore, i, lr)
	cost = lr.cost_(Yscore, lr.predict(x))
	print(f"{cost = }")
	return cost

if __name__ == "__main__":
	data = pd.read_csv("day02/ex08/are_blue_pills_magics.csv")

	Xpill = np.array(data["Micrograms"]).reshape(-1,1)
	Yscore = np.array(data["Score"]).reshape(-1,1)

	cost = []
	for i in range(1, 6 + 1):
		print(f"Polynomial fit n*{i}")
		c = one_loop(Xpill, Yscore, i)
		cost.append(c)

	legend = [f"Pol {i}" for i in range(1, 6 + 1)]
	plt.bar(legend,cost)
	plt.show()
