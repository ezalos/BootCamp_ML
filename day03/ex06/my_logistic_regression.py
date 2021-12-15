import numpy as np

def sigmoid_(x):
    return 1 / (1 + np.exp(-x))

def add_intercept(x):
	vec_one = np.ones(x.shape[0])
	result = np.column_stack((vec_one, x))
	return result

class MyLogisticRegression():
	"""
	Description:
	My personnal logistic regression to classify things.
	"""
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = np.array(theta).reshape(-1, 1)

	def predict_(self, x):
		if x.shape[1] + 1 == self.theta.shape[0]:
			x = add_intercept(x)
		self.theta = self.theta.reshape((-1, 1))
		z = x.dot(self.theta)
		a = sigmoid_(z)

	def loss_elem_(self, x, y):
		pass

	def loss_(self, x, y):
		eps = 1e-15
		m = y.shape[0]
		y_hat = self.predict_(x)


		log_true = np.dot(y.T, np.log(y_hat + eps))
		log_false = np.dot((1 - y).T, np.log((1 - y_hat) + eps))
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = np.squeeze(J)

		return J

	def gradient(self, x, y):
		m = x.shape[0]

		# hypothesis = x_.dot(theta)
		hypothesis = self.predict_(x)
		loss = hypothesis - y
		grad = (x.T @ loss) / m

		return grad

	def fit_(self, x, y):
		x_ = add_intercept(x)
		for i in range(self.max_iter):
			gradient = self.gradient(x_, y).sum(axis=1)
			theta_update = (gradient * self.alpha).reshape((-1, 1))
			self.theta = self.theta - theta_update
		return self.theta

if __name__ == "__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])

	MyLR = MyLogisticRegression
	mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09])

	print("# Example 0:")
	print(f"{mylr.predict_(X) = }")
	print("# Output:")
	print("""array([[0.99930437],
	[1. ],
	[1. ]])""")
	print()

	print("# Example 1:")
	print(f"{mylr.loss_(X,Y) = }")
	print("# Output:")
	print("11.513157421577004")
	print()

	print("# Example 2:")
	print(f"{mylr.fit_(X, Y) = }")
	print(f"{mylr.theta = }")
	print("# Output:")
	print("""array([[ 1.04565272],
	[ 0.62555148],
	[ 0.38387466],
	[ 0.15622435],
	[-0.45990099]])""")
	print()

	print("# Example 3:")
	print(f"{mylr.predict_(X) = }")
	print("# Output:")
	print("""array([[0.72865802],
	[0.40550072],
	[0.45241588]])""")
	print()

	print("# Example 4:")
	print(f"{mylr.loss_(X,Y) = }")
	print("# Output:")
	print("0.5432466580663214")
	print()
