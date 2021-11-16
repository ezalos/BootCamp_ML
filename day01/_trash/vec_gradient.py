import numpy as np
import sys
path = "/Users/ldevelle/42/42-AI/BootCamp_ML/day00/ex04"
# path = '/home/ezalos/42/Bootcamp_Python/bootcamp_machine-learning/day00/ex05'
sys.path.insert(1, path)
from tools import add_intercept

def vec_gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
	,→ The three arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a matrice of dimension (m, n).
	y: has to be an numpy.ndarray, a vector of dimension (m, 1).
	theta: has to be an numpy.ndarray, a vector of dimension (n, 1).
	Returns:
	The gradient as a numpy.ndarray, a vector of dimensions (n, 1), containg the result of
	,→ the formula for all j.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""

	x_ = add_intercept(x)
	m = x_.shape[0]
	n = x_.shape[1]

	if y.shape[0] != x.shape[0] or theta.shape[0] != x.shape[1] + 1:
		print("X shape: ", x.shape)
		print("Y shape: ", y.shape)
		print("T shape: ", theta.shape)
		return None

	elem = np.matmul(x_, theta) - y
	answer = np.matmul(x.T, elem) / m

	return answer

if __name__ == "__main__":
	X = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])

	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	theta = np.array([0, 3, 0.5, -6])
	print(vec_gradient(X, Y, theta))
	print("# array([ -37.35714286, 183.14285714, -393.])")
	print()

	theta = np.array([0, 0, 0, 0])
	print(vec_gradient(X, Y, theta))
	print("# array([ 0.85714286, 23.28571429, -26.42857143])")
	print()

	print(vec_gradient(X, add_intercept(X).dot(theta), theta))
	print("# array([0., 0., 0.])")
	print()
