import numpy as np
import sys
path = "/Users/ldevelle/42/42-AI/BootCamp_ML/day00/ex04"
# path = '/home/ezalos/42/Bootcamp_Python/bootcamp_machine-learning/day00/ex05'
sys.path.insert(1, path)
from tools import add_intercept

def predict_(x, theta):
	"""Computes the prediction vector y_hat from two non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimensions m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	y_hat as a numpy.ndarray, a vector of dimension m * 1.
	None if x or theta are empty numpy.ndarray.
	None if x or theta dimensions are not appropriate.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x) == 0:
		return None
	x = add_intercept(x)
	if len(theta) != x.shape[1]:
		return None
	return x.dot(theta)


if __name__ == "__main__":
	x = np.arange(1,6)
	print("#Example 1:")
	theta1 = np.array([5, 0])
	print(predict_(x, theta1))
	# Ouput:
	print("array([5., 5., 5., 5., 5.])")
	# Do you remember why y_hat contains only 5's here?
	print()

	print("#Example 2:")
	theta2 = np.array([0, 1])
	print(predict_(x, theta2))
	# Output:
	print("array([1., 2., 3., 4., 5.])")
	# Do you remember why y_hat == x here?
	print()

	print("#Example 3:")
	theta3 = np.array([5, 3])
	print(predict_(x, theta3))
	# Output:
	print("array([ 8., 11., 14., 17., 20.])")
	print()

	print("#Example 4:")
	theta4 = np.array([-3, 1])
	print(predict_(x, theta4))
	# Output:
	print("array([-2., -1., 0., 1., 2.])")
	print()
