import numpy as np

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept

def simple_predict(x, theta):
	"""Computes the prediction vector y_hat from two non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a matrix of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
	Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exception.
	"""
	# print(x, x.shape)
	# print(theta, theta.shape)
	print(f"{x.shape = }")
	x_ = add_intercept(x)
	print(f"{x_.shape = }")
	print(f"{theta.shape = }")
	theta = theta.reshape((-1, 1))
	print(f"{theta.shape = }")
	# print(theta, theta.shape)
	# ans = 0
	return x_.dot(theta)

if __name__ == "__main__":
	x = np.arange(1,13).reshape((4,-1))
	print("#Example 1:")
	theta1 = np.array([5, 0, 0, 0])
	print(simple_predict(x, theta1))
	# Ouput:
	print("array([5., 5., 5., 5.])")
	# Do you understand why y_hat contains only 5's here?
	print()

	print("#Example 2:")
	theta2 = np.array([0, 1, 0, 0])
	print(simple_predict(x, theta2))
	# Output:
	print("array([ 1., 4., 7., 10.])")
	# Do you understand why y_hat == x[:,0] here?
	print()

	print("#Example 3:")
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
	print(simple_predict(x, theta3))
	# Output:
	print("array([ 9.64, 24.28, 38.92, 53.56])")
	print()

	print("#Example 4:")
	theta4 = np.array([-3, 1, 2, 3.5])
	print(simple_predict(x, theta4))
	# Output:
	print("array([12.5, 32. , 51.5, 71. ])")
	print()

	# print("PRE CORRECTION:")
	# x = np.ones((6,4))
	# theta = np.ones((4,1))
	# print(simple_predict(x, theta))

	print("CORRECTION:")
	print("Test 1")
	x = (np.arange(1,13)).reshape(-1,2)
	theta = np.ones(3).reshape(-1,1)
	print(simple_predict(x, theta))
	print("array([[ 4.], [ 8.], [12.], [16.], [20.], [24.]])")
	print()

	print("Test 2")
	x = (np.arange(1,13)).reshape(-1,3)
	theta = np.ones(4).reshape(-1,1)
	print(simple_predict(x, theta))
	print("array([[ 7.], [16.], [25.], [34.]])")
	print()

	print("Test 3")
	x = (np.arange(1,13)).reshape(-1,4)
	theta = np.ones(5).reshape(-1,1)
	print(simple_predict(x, theta))
	print("array([[11.], [27.], [43.]])")
	print()

    #   - x = (np.arange(1,13)).reshape(-1,2) and theta = np.ones(3).reshape(-1,1)
    #     function should returned: array([[ 4.], [ 8.], [12.], [16.], [20.], [24.]])
    #   - x = (np.arange(1,13)).reshape(-1,3) and theta = np.ones(4).reshape(-1,1)
    #     the function should returned: array([[ 7.], [16.], [25.], [34.]])
    #   - x = (np.arange(1,13)).reshape(-1,4) and theta = np.ones(5).reshape(-1,1)
    #     the function should returned: array([[11.], [27.], [43.]])