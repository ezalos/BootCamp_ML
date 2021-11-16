import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from prediction import predict_
from tools import add_intercept



def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
	,→ The three arrays must have compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a 2 * 1 vector.
	Returns:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	x_ = x
	m = x_.shape[0]

	j = (1/m) * (x_.T @ (x_ @ theta - y))
	return j

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.
	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
	,→ examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
	,→ examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient
	,→ descent
	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""
	x_ = add_intercept(x)
	theta_ = 0
	for i in range(max_iter):
		# if not i % 100000:
		# 	print(i * 100 / max_iter, "%")
		# 	print(theta)
		# 	print(theta_)
		theta_ = (gradient(x_, y, theta).sum(axis=1) * alpha).reshape((-1, 1))
		theta = theta - theta_
	return theta


if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	# theta= np.array([[1], [1]])
	# print("# Example 0:")
	# theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
	# print(theta1)
	# # Output:
	# print("""
	# array([[1.40709365],
	# [1.1150909 ]])
	# """)
	# print()

	# print("# Example 1:")
	# print(predict_(x, theta1))
	# # Output:
	# print("""
	# array([[15.3408728 ],
	# [25.38243697],
	# [36.59126492],
	# [55.95130097],
	# [65.53471499]])
	# """)

	x = np.array(range(1,101)).reshape(-1,1)
	y = 0.75*x + 5
	theta = np.array([[1.],[1.]])
	print(fit_(x, y, theta, 5e-4, 20000))
	print("[[4.03112103], [0.76446193]]")



	# - with x = np.array(range(1,101)).reshape(-1,1), 
	# y = 0.75*x + 5 and 
	# theta = np.array([[1.],[1.]])
	# fit_(x, y, theta, 1e-5, 2000) 
	# should return a result close to 
	# [[4.03112103], [0.76446193]].
