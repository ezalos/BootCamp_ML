import numpy as np

def add_intercept(x):
	"""Adds a column of 1's to the non-empty numpy.ndarray x.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	Returns:
	X as a numpy.ndarray, a vector of dimension m * 2.
	None if x is not a numpy.ndarray.
	None if x is a empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	vec_one = np.ones(x.shape[0])
	result = np.column_stack((vec_one, x))
	return result

if __name__ == "__main__":
	# Example 1:
	x = np.arange(1,6)
	print(x)
	print(add_intercept(x))
	# Output:
	print("""
	[[1., 1.],
	[1., 2.],
	[1., 3.],
	[1., 4.],
	[1., 5.]]
	""")
	print()

	# Example 2:
	y = np.arange(1,10).reshape((3,3))
	print(y)
	print(add_intercept(y))
	# Output:
	print("""
	[[1., 1., 2., 3.],
	[1., 4., 5., 6.],
	[1., 7., 8., 9.]]
	""")
