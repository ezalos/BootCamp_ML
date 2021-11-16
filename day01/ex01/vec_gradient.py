import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
from prediction import predict_

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
	x_ = add_intercept(x)
	m = x_.shape[0]

	j = (1/m) * (x_.T @ (x_ @ theta - y))
	return j

if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	print("# Example 0:")
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	print(gradient(x, y, theta1))
	# print(np.gradient(predict_(x, theta1), y))
	# Output:
	print("array([[-19.0342574], [-586.66875564]])")
	print()

	print("# Example 1:")
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(gradient(x, y, theta2))
	# Output:
	print("array([[-57.86823748], [-2230.12297889]])")



	def unit_test(n, theta, answer, f):
		x = np.array(range(1,n+1)).reshape((-1, 1))
		y = f(x)
		print(f"Student:\n{gradient(x, y, theta)}")
		print(f"Truth  :\n{answer}")
		print()


	theta = np.array([[1.],[1.]])
	answer = np.array([[-11.625], [-795.375]])
	unit_test(100, theta, answer, lambda x:1.25 * x)

	answer = np.array([[-124.125], [-82957.875]])
	unit_test(1000, theta, answer, lambda x:1.25 * x)

	answer = np.array([[-1.24912500e+03], [-8.32958288e+06]])
	unit_test(10000, theta, answer, lambda x:1.25 * x)

	theta = np.array([[4], [-1]])
	answer = np.array([[-13.625], [-896.375]])
	unit_test(100, theta, answer, lambda x:-0.75 * x + 5)

	answer = np.array([[-126.125], [-83958.875]])
	unit_test(1000, theta, answer, lambda x:-0.75 * x + 5)

	answer = np.array([[-1.25112500e+03], [-8.33958388e+06]])
	unit_test(10000, theta, answer, lambda x:-0.75 * x + 5)



#  The function should handle vectors (x and y) of dimension (n, 1),
#       where n can be 100, 1000 or 10000.
#       With the vectors x np.array(range(1,n+1)) and y equals to 1.25*x,
#       - with n=100 and theta = [[1.],[1.]], simple_gradient must return [[-11.625], [-795.375]]
#       - with n=1000 and theta = [[1.],[1.]], simple_gradient must return [[-124.125], [-82957.875]]
#       - with n=10000 and theta = [[1.],[1.]], simple_gradient must return [[-1.24912500e+03], [-8.32958288e+06]]
#       With the vectors x np.array(range(1,n+1)) and y equals to -0.75-x + 5,
#       - with n=100 and theta = [[4], [-1]], simple_gradient must return [[-13.625], [-896.375]]
#       - with n=1000 and theta = [[4], [-1]], simple_gradient must return [[-126.125], [-83958.875]]
#       - with n=10000 and theta = [[4], [-1]], simple_gradient must return [[-1.25112500e+03], [-8.33958388e+06]]
