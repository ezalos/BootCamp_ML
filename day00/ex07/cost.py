import numpy as np
import sys
sys.path.insert(1, '/home/ezalos/42/Bootcamp_Python/\
bootcamp_machine-learning/day00/ex05')
from prediction import predict_

def cost_elem_(y, y_hat):
	"""
	Description:
	Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector.
	Returns:
	J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
	None if there is a dimension matching problem between X, Y or theta.
	Raises:
	This function should not raise any Exception.
	"""
	cost_func = lambda y, y_, m: (1 / (2 * m)) * (y - y_) ** 2
	res = np.array([cost_func(i, j, len(y)) for i, j in zip(y, y_hat)])
	return res

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
	return np.sum(cost_elem_(y, y_hat))

if __name__ == "__main__":
	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	print(y_hat1)
	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

	# Example 1:
	print("# Example 1:")
	print(cost_elem_(y1, y_hat1))
	# Output:
	print("array([[0.], [0.1], [0.4], [0.9], [1.6]])")
	print()

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

	# Example 3:
	print("# Example 3:")
	print(cost_elem_(y2, y_hat2))
	# Output:
	print("array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])")
	print()

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
