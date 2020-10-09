import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict(Xpill)
Y_model2 = linear_model2.predict(Xpill)


# print("Me: ", linear_model1.mse_(Yscore, Y_model1))
# print("Sc: ", mean_squared_error(Yscore, Y_model1))
# print()
#
# print("Me: ", linear_model2.mse_(Yscore, Y_model2))
# print("Sc: ", mean_squared_error(Yscore, Y_model2))




def plot(x, y, theta):
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
	plt.plot(x, y, 'o')

	plt.plot(x, theta[1] * x + theta[0])
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
	plt.ylim((0, 200))
	plt.xlim((-15, -5))
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
		label = "Th0=" + str(int(t0 * 10) / 10)
		print(label, "done!")
		plt.plot(theta_list, cost_list, label=label)
	plt.xlabel("Th1")
	plt.ylabel("MSE(Th0, Th1)")
	plt.legend(loc='upper left')
	plt.show()
# lr = MyLR(thetas=[89.0, -8], alpha=5e-8, max_iter=1500000)
lr = MyLR(thetas=[0, 0], alpha=1e-3, max_iter=50000)
plot_cost(Xpill, Yscore)
# lr = MyLR(thetas=[49.9399813, -1.42892958], alpha=1e-7, max_iter=2000000)

# lr.gradient(Xpill, Yscore)
# print(lr.mse_(Yscore, lr.predict(Xpill)).mean())
# lr = MyLR(thetas=[89.0, -8], alpha=5e-8, max_iter=1500000)
# lr.gradient(Xpill, Yscore)
# print(lr.mse_(Yscore, lr.predict(Xpill)).mean())

# lr.fit_(Xpill, Yscore)
# plot(Xpill, Yscore, lr.theta)
