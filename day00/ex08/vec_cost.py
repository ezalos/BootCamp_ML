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
	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	print(y_hat1)
	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

	X = np.array([0, 15, -9, 7, 12, 3, -21])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])

	print("# Example 1:")
	print(cost_(X, Y))
	# Output:
	print(2.142857142857143)

	print("# Example 2:")
	# Example 2:
	print(cost_(X, X))
	# Output:
	print(0.0)

	# Example 2:
	print("# Example 2:")
	print(cost_(y1, y_hat1))
	# Output:
	print(3.0)
	print()

	x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	theta2 = np.array([[0.05], [1.], [1.], [1.]])
	y_hat2 = predict_(x2, theta2)
	y2 = np.array([[19.], [42.], [67.], [93.]])

	# Example 4:
	print("# Example 4:")
	print(cost_(y2, y_hat2))
	# Output:
	print(4.238750000000004)
	print()

	x3 = np.array([0, 15, -9, 7, 12, 3, -21])
	theta3 = np.array([[0.], [1.]])
	y_hat3 = predict_(x3, theta3)
	print(y_hat3)
	y3 = np.array([2, 14, -13, 5, 12, 4, -19])

	# Example 5:
	print("# Example 5:")
	print(cost_(y3, y_hat3))
	# Output:
	print(4.285714285714286)
	print()

	# Example 6:
	print("# Example 6:")
	print(cost_(y3, y3))
	# Output:
	print(0.0)
