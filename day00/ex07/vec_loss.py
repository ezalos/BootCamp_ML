import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'ex04')
sys.path.insert(1, path)
from prediction import predict_

def loss_(y, y_hat):
	"""
	Computes the half mean squared error of two non-empty numpy.array, without any for loop.
	The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.
	Returns:
	The half mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.array.
	None if y and y_hat does not share the same dimensions.
	None if y or y_hat is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	res = (1 / (2 * y.shape[0])) * (y_hat - y).dot(y_hat - y)
	return -res if res < 0 else res

if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])

	print("# Example 1:")
	print(loss_(X, Y))
	print(2.142857142857143)
	print()

	print("# Example 2:")
	print(loss_(X, X))
	print(0.0)
	print()

	x3 = np.array([0, 15, -9, 7, 12, 3, -21])
	theta3 = np.array([[0.], [1.]])
	y_hat3 = predict_(x3, theta3)
	y3 = np.array([2, 14, -13, 5, 12, 4, -19])

	print("# Example 5:")
	print(loss_(y3, y_hat3))
	print(2.142857142857143)
	print()

	print("# Example 6:")
	print(loss_(y3, y3))
	print(0.0)



	y_hat = np.array([1, 2, 3, 4])
	y = np.array([0, 0, 0, 0])
	
	print(loss_(y, y_hat))
	print("3.75")