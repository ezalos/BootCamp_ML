import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from prediction import predict_
path = os.path.join(os.path.dirname(__file__), '..', 'ex03')
sys.path.insert(1, path)
from my_linear_regression import MyLinearRegression as MyLR


file_path = os.path.join(os.path.dirname(__file__), 'are_blue_pills_magics.csv')
data = pd.read_csv(file_path)

Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict(Xpill)
Y_model2 = linear_model2.predict(Xpill)



def plot(x, y, lr):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exceptions.
	"""
	plt.plot(x, y, 'o', c='b')

	y_ = lr.predict(x)
	plt.plot(x, y_, 'g--')
	plt.scatter(x, y_, c='g')
	plt.show()


def plot_cost(x, y):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exceptions.
	"""
	# plt.plot(x, y, 'o')
	# x = np.linspace(-15,5,100)
	plt.ylim((10, 50))
	plt.xlim((-13, -4.5))
	ran = 15
	upd = ran * 2 / 6
	for t0 in np.arange(89 - ran, 89 + ran, upd):
		cost_list = []
		theta_list =[]
		for t1 in np.arange(-8 -100, -8 + 100, 0.1):
			lr = MyLR(thetas=[t0, t1], alpha=1e-3, max_iter=50000)
			y_ = lr.predict(x)
			mse_c = lr.cost_(y, y_)#[0][0]
			cost_list.append(mse_c)
			theta_list.append(t1)
			# print(cost_list[-1])
		label = "θ[0]=" + str(int(t0 * 10) / 10)
		print(label, "done!")
		plt.plot(theta_list, cost_list, label=label)
	plt.xlabel("θ[1]")
	plt.ylabel("MSE(θ[0], θ[1])")
	plt.legend(loc='upper left')
	plt.show()


lr = MyLR(thetas=[0, 0], alpha=1e-3, max_iter=50000)
lr.fit_(Xpill, Yscore)
plot(Xpill, Yscore, lr)

plot_cost(Xpill, Yscore)

