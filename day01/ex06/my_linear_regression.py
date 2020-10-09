import numpy as np
import matplotlib.pyplot as plt


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

class MyLinearRegression():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas=[0, 0], alpha=0.001, n_cycle=1000, max_iter=10000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = np.array(thetas)
		self.n_cycle = n_cycle
		self.graph = None
		self.cost = []

	def gradient(self, x, y):
		"""Computes a gradient vector from three non-empty numpy.ndarray,
			without any for-loop. The three arrays must have compatible dimensions.
		Args:
			x: has to be an numpy.ndarray, a vector of dimension m * 1.
			y: has to be an numpy.ndarray, a vector of dimension m * 1.
			theta: has to be an numpy.ndarray, a 2 * 1 vector.
		Returns:
			The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
			None if x, y, or theta are empty numpy.ndarray.
			None if x, y and theta do not have compatible dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		x_ = add_intercept(x)
		# y_ = np.array(y).reshape((-1,1))
		m = x_.shape[0]
		# print(x_.shape)
		# print(y.shape)
		# print(y_.shape)
		# print((self.theta - y).shape)
		# j = (1/m) * (x_.T @ ((x_ @ self.theta) - y)).T)).sum(axis=1)
		# j = (1/m) * (x_.T @ ((x_ @ self.theta) - y)).mean(axis=1)
		j = (x_.T @ ((x_ @ self.theta) - y)).mean(axis=1)
		# print(j)
		return j

	def plot(self, x, y):
		"""Plot the data and prediction line from three non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
		Returns:
		Nothing.
		Raises:
		This function should not raise any Exceptions.
		"""
		if self.graph == None:
			plt.ion()
			self.graph = True
			# self.fig = plt.figure()
			# self.graph = self.fig.add_subplot(111)
		else:
			plt.clf()
			# self.graph.clear()
		plt.plot(x, y, 'o')
		plt.plot(x, self.theta[1] * x + self.theta[0])
		if True:
			y_hat = self.predict(x)
			for x_i, y_hat_i, y_i in zip(x, y_hat, y):
				plt.plot([x_i, x_i], [y_i, y_hat_i], 'r--')
		# plt.draw()
		plt.pause(0.000000000001)
		plt.show()

	def fit_(self, x, y):
		"""
		Description:
			Fits the model to the training dataset contained in x and y.
		Args:
			x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
				examples, 1).
			y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
				examples, 1).
			theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
			alpha: has to be a float, the learning rate
			max_iter: has to be an int, the number of iterations done during the gradient
				descent
		Returns:
			new_theta: numpy.ndarray, a vector of dimension 2 * 1.
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		self.cost = []
		for i in range(self.max_iter):
			if not i % 10000:
				print(i * 100 / self.max_iter, "%")
				self.plot(x, y)
				print(self.theta)
				self.cost.append(self.mse_((add_intercept(x) @ self.theta), y).mean())
				print(self.cost[-1])
			theta_ = self.gradient(x, y) * self.alpha
			self.theta = self.theta - theta_
		return self.theta

	def cost_elem_(self, y_hat, y):
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
		m = len(y)
		cost = (1 / (2 * m)) * np.abs((y_hat - y) ** 2).sum(axis=1)
		# cost = (1 / (2 * m)) * np.abs((y_hat - y).dot(y - y_hat)).sum(axis=1)
		return cost

	def cost_(self, y_hat, y):
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
		# if len(y.shape) > 1:
		# 	return None
		res = (1 / (2 * y.shape[0])) * (y_hat - y).dot(y - y_hat).sum()
		return abs(res)
		# return ((y_hat - y) ** 2).sum()

	def mse_(self, y, y_hat):
		"""
		Description:
		Calculate the MSE between the predicted output and the real output.
		Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
		Returns:
		mse: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		# if len(y.shape) > 1 or y.shape != y_hat.shape :
		# 	return None
		res = (1 / (y.shape[0])) * (y_hat - y).T.dot(y_hat - y)
		return abs(res).sum(axis=1)

	def rmse_(self, y, y_hat):
		"""
		Description:
		Calculate the RMSE between the predicted output and the real output.
		Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
		Returns:
		rmse: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		if len(y.shape) > 1 or y.shape != y_hat.shape :
			return None
		res = (1 / (y.shape[0])) * (y_hat - y).dot(y_hat - y)
		return sqrt(abs(res))

	def mae_(self, y, y_hat):
		"""
		Description:
		Calculate the MAE between the predicted output and the real output.
		Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
		Returns:
		mae: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		if len(y.shape) > 1 or y.shape != y_hat.shape :
			return None
		res = (1 / (y.shape[0])) * abs(y_hat - y).sum()
		return abs(res)


	def r2score_(self, y, y_hat):
		"""
		Description:
		Calculate the R2score between the predicted output and the output.
		Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
		Returns:
		r2score: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		if len(y.shape) > 1 or y.shape != y_hat.shape :
			return None

		top = (y - y_hat)
		bot = (y - y.mean())
		top = top ** 2
		bot = bot ** 2
		top = top.sum()
		bot = bot.sum()
		res = 1 - (top / bot)
		return abs(res)


	def predict(self, x):
		"""Computes the prediction vector y_hat from two non-empty numpy.ndarray.
		Args:
			x: has to be an numpy.ndarray, a vector of dimensions m * 1.
			theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
		Returns:
			y_hat as a numpy.ndarray, a vector of dimension m * 1.
			None if x or theta are empty numpy.ndarray.
			None if x or theta dimensions are not appropriate.
		Raises:
			This function should not raise any Exception.
		"""
		if len(x) == 0:
			return None
		x = add_intercept(x)
		if len(self.theta) != x.shape[1]:
			return None
		return x @ self.theta

if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	MyLR = MyLinearRegression
	lr1 = MyLR([2, 0.7])


	# Example 0.0:
	print("# Example 0.0:")
	print(lr1.predict(x))
	# Output:
	print("""
	array([[10.74695094],
	[17.05055804],
	[24.08691674],
	[36.24020866],
	[42.25621131]])
	""")


	# Example 0.1:
	print("# Example 0.1:")
	print(lr1.cost_elem_(lr1.predict(x),y))
	# Output:
	print("""
	array([[77.72116511],
	[49.33699664],
	[72.38621816],
	[37.29223426],
	[78.28360514]])
	""")

	# Example 0.2:
	print("# Example 0.2:")
	print(lr1.cost_(lr1.predict(x),y))
	# Output:
	print(315.0202193084312)

	# Example 1.0:
	print("# Example 1.0:")
	lr2 = MyLR(thetas=[1, 1], alpha=5e-8, max_iter=1500000)
	# print(lr2.fit_(x, y))
	print(lr2.theta)
	# Output:
	print("""
	array([[1.40709365],
	[1.1150909 ]])
	""")


	# Example 1.1:
	print("# Example 1.1:")
	print(lr2.predict(x))
	# Output:
	print("""
	array([[15.3408728 ],
	[25.38243697],
	[36.59126492],
	[55.95130097],
	[65.53471499]])
	""")


	# Example 1.2:
	print("# Example 1.2:")
	print(lr2.cost_elem_(lr1.predict(x),y))
	# Output:
	print("""
	array([[35.6749755 ],
	[ 4.14286023],
	[ 1.26440585],
	[29.30443042],
	[22.27765992]])
	""")


	# Example 1.3:
	print("# Example 1.3:")
	print(lr2.cost_(lr1.predict(x),y))
	# Output:
	print("92.66433192085971")
