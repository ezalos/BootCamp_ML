import numpy as np
import matplotlib.pyplot as plt
import math

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
	def __init__(self, thetas=[0, 0], alpha=0.001, max_iter=100000):
		"""
			Description:
				generator of the class, initialize self.
			Args:
				theta: has to be a list or a numpy array,
					it is a vector of dimension (number of features + 1, 1).
			Raises:
				This method should noot raise any Exception.
		"""
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = np.array(thetas).reshape(-1, 1)
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
		x_ = x
		m = x_.shape[0]

		hypothesis = (x_ @ self.theta)
		loss = hypothesis - y
		gradient = (x_.T @ loss) / m

		return gradient

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
		x_ = add_intercept(x)
		for i in range(self.max_iter):
			gradient = self.gradient(x_, y).sum(axis=1)
			theta_update = (gradient * self.alpha).reshape((-1, 1))
			self.theta = self.theta - theta_update
		return self.theta

	def loss_elem_(self, y_hat, y):
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
		cost_func = lambda y, y_, m: (y - y_) ** 2
		res = np.array([cost_func(i, j, len(y)) for i, j in zip(y, y_hat)])
		return res

	def loss_(self, y_hat, y):
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
		res = (1 / (2 * y.shape[0])) * (y_hat - y).T.dot(y - y_hat).sum()
		return abs(res)

	def predict_(self, x):
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
		if x.shape[1] + 1 == self.theta.shape[0]:
			x = add_intercept(x)
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
	print(lr1.loss_elem_(lr1.predict(x),y))
	# Output:
	print("""
array([[710.45867381],
       [364.68645485],
       [469.96221651],
       [108.97553412],
       [299.37111101]])
	""")

	# Example 0.2:
	print("# Example 0.2:")
	print(lr1.loss_(lr1.predict(x),y))
	# Output:
	print(195.34539903032385)


	# Example 1.0:
	print("# Example 1.0:")
	lr2 = MyLR(thetas=[1, 1], alpha=5e-8, max_iter=1500000)
	print(lr2.fit_(x, y))
	print(lr2.theta)
	# Output:
	print("""
	array([[1.40709365],
	[1.1150909 ]])
	""")


	# import sys
	# sys.exit()

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
	print(lr2.loss_elem_(y, lr2.predict(x)))
	# Output:
	print("""
array([[486.66604863],
       [115.88278416],
       [ 84.16711596],
       [ 85.96919719],
       [ 35.71448348]])
	""")


	# Example 1.3:
	print("# Example 1.3:")
	print(lr2.loss_(y, lr2.predict(x)))
	# Output:
	print("80.83996294128525")
