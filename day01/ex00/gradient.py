import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
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
	j0 = (predict_(x, theta) - y).sum() / m
	j1 = ((predict_(x, theta) - y).T.dot(x)) / m
	a =  np.array([j0, j1.squeeze()]).reshape((-1, 1))
	return a

	

if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	print("# Example 0:")
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	print(simple_gradient(x, y, theta1))
	# Output:
	print("array([-19.0342574, -586.66875564])")
	print()

	print("# Example 1:")
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(simple_gradient(x, y, theta2))
	# Output:
	print("array([-57.86823748, -2230.12297889])")


	x = np.array(range(1,11)).reshape((-1, 1))
	y = 1.25*x

	theta = np.array([[1.],[1.]])
	print(f"Student:\n{simple_gradient(x, y, theta)}")
	print(f"Truth  :\n{np.array([[-0.375],[-4.125]])}")
	print()

	theta = np.array([[1.],[-0.4]])
	print(f"Student:\n{simple_gradient(x, y, theta)}")
	print(f"Truth  :\n{np.array([[-8.075],[-58.025]])}")
	print()

	theta = np.array([[0.],[1.25]])
	print(f"Student:\n{simple_gradient(x, y, theta)}")
	print(f"Truth  :\n{np.array([[0.],[0.]])}")
	print()
