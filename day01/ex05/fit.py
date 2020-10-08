import numpy as np
from vec_gradient import gradient
from prediction import predict_

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.
	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
	,→ examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
	,→ examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient
	,→ descent
	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""
	# theta_ = gradient(x, y, theta).sum(axis=1)
	# print("TH", theta_)
	# print("THa", theta_ * alpha)
	# max_iter = 0
	for i in range(max_iter):
		if not i % 100000:
			print(i * 100 / max_iter, "%")
			print(theta)
		theta_ = gradient(x, y, theta).sum(axis=1) * alpha
		theta = theta - theta_
	return theta
	pass


if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta= np.array([1, 1])
	print("# Example 0:")
	theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
	print(theta1)
	# Output:
	print("""
	array([[1.40709365],
	[1.1150909 ]])
	""")
	print()

	print("# Example 1:")
	print(predict_(x, theta1))
	# Output:
	print("""
	array([[15.3408728 ],
	[25.38243697],
	[36.59126492],
	[55.95130097],
	[65.53471499]])
	""")
