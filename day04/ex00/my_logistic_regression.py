import numpy as np

def add_intercept(x):
	vec_one = np.ones(x.shape[0])
	result = np.column_stack((vec_one, x))
	return result

class MyLogisticRegression():
	"""
	Description:
	My personnal logistic regression to classify things.
	"""

	def __init__(self, thetas, alpha=1e-4, n_cycle=10e5):
		self.alpha = alpha
		self.n_cycle = int(n_cycle)
		self.thetas = np.array(thetas).reshape(-1, 1)
		# Your code here

	def loss(self, y, y_hat, eps=1e-15):
		m = len(y)
		y_true = y.T @ np.log(y_hat + eps)
		one = np.ones(m)
		y_false = (one - y.T) @ np.log(one - y_hat + eps)
		j_theta = y_true + y_false
		j_theta = - j_theta / m
		return j_theta

	def gradient(self, x, y):
		m = x.shape[0]
		x_ = add_intercept(x)
		gradient = (x_.T @ ((self.predict_(x)) - y)) / m
		return gradient

	def fit_(self, x, y):
		for i in range(self.n_cycle + 1):
			theta = self.gradient(x, y) * self.alpha
			self.thetas = self.thetas - theta

	def predict_(self, x):
		x_ = add_intercept(x)
		exp = - x_ @ self.thetas
		res = 1 / (1 + np.exp(exp))
		return res

	def cost_(self, x, y):
		y_hat = self.predict_(x)
		res = self.loss(y, y_hat)
		return res.mean()

if __name__ == "__main__":
	MyLR = MyLogisticRegression
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])
	mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09])

	print("# Example 0:")
	res = mylr.predict_(X)
	print(res)
	print("""
	# Output:
	array([[0.99930437],
	[1. ],
	[1. ]])
	""")


	print("# Example 1:")
	res = mylr.cost_(X,Y)
	print(res)
	print("""
	# Output:
	11.513157421577004
	""")

	print("# Example 2:")
	mylr.fit_(X, Y)
	res = mylr.thetas
	print(res)
	print("""
	# Output:
	array([[ 1.04565272],
	[ 0.62555148],
	[ 0.38387466],
	[ 0.15622435],
	[-0.45990099]])
	""")


	print("# Example 3:")
	res = mylr.predict_(X)
	print(res)
	print("""
	# Output:
	array([[0.72865802],
	[0.40550072],
	[0.45241588]])
	""")


	print("# Example 4:")
	res = mylr.cost_(X,Y)
	print(res)
	print("""
	# Output:
	0.5432466580663214
	""")
