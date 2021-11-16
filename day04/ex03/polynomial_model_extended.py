import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to vector x by raising its values up to the power given in
	,→ argument.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	power: has to be an int, the power up to which the components of vector x are going to
	,→ be raised.
	Returns:
	The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he
	,→ polynomial feature values for all training examples.
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	new = []
	new.append(x)
	for i in range(power - 1):
		new.append(x ** (i + 2))
	return np.concatenate(tuple(new),axis=1)

if __name__ == "__main__":
	x = np.arange(1,11).reshape(5, 2)
	print("x: ", x)
	print("# Example 1:")
	res = add_polynomial_features(x, 3)
	print(res)
	print("""
	# Output:
		array([[ 1, 2, 1, 4, 1, 8],
		[ 3, 4, 9, 16, 27, 64],
		[ 5, 6, 25, 36, 125, 216],
		[ 7, 8, 49, 64, 343, 512],
		[ 9, 10, 81, 100, 729, 1000]])
	""")
	print("# Example 2:")
	res = add_polynomial_features(x, 5)
	print(res)
	print("""
	# Output:
		array([[ 1, 2, 1, 4, 1, 8, 1, 16],
		[ 3, 4, 9, 16, 27, 64, 81, 256],
		[ 5, 6, 25, 36, 125, 216, 625, 1296],
		[ 7, 8, 49, 64, 343, 512, 2401, 4096],
		[ 9, 10, 81, 100, 729, 1000, 6561, 10000]])
	""")
