import numpy as np
from tools import add_intercept
from prediction import predict_

def simple_gradient(x, y, theta):
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
	m = x.shape[0]
	# print(predict_(x, theta))
	# print(y)
	# print(predict_(x, theta) - y)
	j0 = (predict_(x, theta) - y).sum() / m
	j1 = ((predict_(x, theta) - y).dot(x)) / m
	# print(j1)
	return np.array([j0, j1])

if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
	print("# Example 0:")
	theta1 = np.array([2, 0.7])
	print(simple_gradient(x, y, theta1))
	# Output:
	print("array([21.0342574, 587.36875564])")
	print()

	print("# Example 1:")
	theta2 = np.array([1, -0.4])
	print(simple_gradient(x, y, theta2))
	# Output:
	print("array([58.86823748, 2229.72297889])")
