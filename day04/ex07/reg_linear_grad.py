import numpy as np

def add_intercept(x):
	vec_one = np.ones(x.shape[0])
	result = np.column_stack((vec_one, x))
	return result

def reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray, with two
	,→ for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula
	,→ for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	pass

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray, without
	,→ any for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula
	,→ for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	x_ = add_intercept(x)
	m = y.shape[0]
	hyp = (x_.T @ ((x_ @ theta) - y))
	# print(hyp)
	theta[0] = 0
	reg = lambda_ * theta
	# print(reg)
	res = (hyp + reg) / m
	return res

if __name__ == "__main__":
	x = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])


	print("# Example 1.1:")
	res = reg_linear_grad(y, x, theta, 1)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-195.64714286],
	[ 863.46571429],
	[-644.52142857]])
	""")


	print("# Example 1.2:")
	res = vec_reg_linear_grad(y, x, theta, 1)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-195.64714286],
	[ 863.46571429],
	[-644.52142857]])
	""")


	print("# Example 2.1:")
	res = reg_linear_grad(y, x, theta, 0.5)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-195.86142857],
	[ 862.71571429],
	[-644.09285714]])
	""")


	print("# Example 2.2:")
	res = vec_reg_linear_grad(y, x, theta, 0.5)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-195.86142857],
	[ 862.71571429],
	[-644.09285714]])
	""")


	print("# Example 3.1:")
	res = reg_linear_grad(y, x, theta, 0.0)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-196.07571429],
	[ 861.96571429],
	[-643.66428571]])
	""")


	print("# Example 3.2:")
	res = vec_reg_linear_grad(y, x, theta, 0.0)
	print(res)
	print("""
	# Output:
	array([[ -60.99 ],
	[-196.07571429],
	[ 861.96571429],
	[-643.66428571]])
	""")
