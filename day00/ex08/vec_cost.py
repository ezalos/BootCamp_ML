import numpy as np
import sys
sys.path.insert(1, '/home/ezalos/42/Bootcamp_Python/\
bootcamp_machine-learning/day00/ex05')
from prediction import predict_

def cost_(y, y_hat):
	"""
	Description:
	Calculates the value of cost function.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector
	Returns:
	J_value : has to be a float.
	None if there is a dimension matching problem between X, Y or theta.
	Raises:
	This function should not raise any Exception.
	"""
	if len(y.shape) > 1:
		return None
	res = (1 / (2 * y.shape[0])) * (y_hat - y).dot(y_hat - y)
	return -res if res < 0 else res

if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])

	print("# Example 1:")
	print(cost_(X, Y))
	print(2.142857142857143)
	print()

	print("# Example 2:")
	print(cost_(X, X))
	print(0.0)
	print()

	x3 = np.array([0, 15, -9, 7, 12, 3, -21])
	theta3 = np.array([[0.], [1.]])
	y_hat3 = predict_(x3, theta3)
	y3 = np.array([2, 14, -13, 5, 12, 4, -19])

	print("# Example 5:")
	print(cost_(y3, y_hat3))
	print(4.285714285714286)
	print()

	print("# Example 6:")
	print(cost_(y3, y3))
	print(0.0)
