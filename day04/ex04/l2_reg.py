import numpy as np

def iterative_l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	l2 = 0
	for i in theta[1:]:
		l2 += i * i
	return l2

def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	return theta[1:].T @ theta[1:]

if __name__ == "__main__":
	x = np.array([2, 14, -13, 5, 12, 4, -19])
	print("# Example 1:")
	res = iterative_l2(x)
	print(res)
	print("""
	# Output:
	911.0
	""")
	print("# Example 2:")
	res = l2(x)
	print(res)
	print("""
	# Output:
	911.0
	""")
	y = np.array([3,0.5,-6])
	print("# Example 3:")
	res = iterative_l2(y)
	print(res)
	print("""
	# Output:
	36.25
	""")
	print("# Example 4:")
	res = l2(y)
	print(res)
	print("""
	# Output:
	36.25
	""")
