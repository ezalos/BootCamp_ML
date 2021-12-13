import numpy as np

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

	res = (1 / (2 * y.shape[0])) * (y_hat - y).T.dot(y_hat - y)
	res = -res if res < 0 else res
	return float(res)

if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1,1)
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1,1)

	print("# Example 0:")
	print(loss_(X, Y))
	print(2.142857142857143)
	print()

	print("# Example 1:")
	print(loss_(X, X))
	print(0.0)
	print()

	print("Correction:")
	n = 10
	y = (np.ones(n)).reshape(-1,1)
	y_hat = (np.zeros(n)).reshape(-1,1)
	print(f"{loss_(y, y_hat) = }")
	print(f"Answer = {0.5}")

	y = (np.ones(n)).reshape(-1,1)+4
	y_hat = (np.zeros(n)).reshape(-1,1)
	print(f"{loss_(y, y_hat) = }")
	print(f"Answer = {12.5}")

	y = (np.ones(7)).reshape(-1,1)+4
	y_hat = (np.arange(7)).reshape(-1,1)
	print(f"{loss_(y, y_hat) = }")
	print(f"Answer = {4}")