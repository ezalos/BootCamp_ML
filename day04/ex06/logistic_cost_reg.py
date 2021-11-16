import numpy as np

def reg_log_cost_(y, y_hat, theta, lambda_):
	"""Computes the regularized cost of a linear regression model from two non-empty
	,→ numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	The regularized cost as a float.
	None if y, y_hat, or theta are empty numpy.ndarray.
	None if y and y_hat do not share the same dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	m = y.shape[0]
	one = np.ones(m)
	eps = 1e-15
	y_true = y.T @ np.log(y_hat + eps)
	y_false = (one - y.T) @ np.log(one - y_hat + eps)
	hyp_cost = -(y_true + y_false) / m
	reg = lambda_ * (theta[1:].T @ theta[1:]) / (2 * m)
	res = hyp_cost + reg
	return res

if __name__ == "__main__":
	y = np.array([1, 1, 0, 0, 1, 1, 0])
	y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
	theta = np.array([1, 2.5, 1.5, -0.9])

	print("# Example 0:")
	res = reg_log_cost_(y, y_hat, theta, .5)
	print(res)
	print("""
	# Output:
	0.43377043716475955
	""")

	print("# Example 1:")
	res = reg_log_cost_(y, y_hat, theta, .05)
	print(res)
	print("""
	# Output:
	0.13452043716475953
	""")

	print("# Example 2:")
	res = reg_log_cost_(y, y_hat, theta, .9)
	print(res)
	print("""
	# Output:
	0.6997704371647596
	""")
