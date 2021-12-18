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
	def __init__(self, theta, alpha=0.001, max_iter=10000, penalty='l2'):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = np.array(theta).reshape(-1, 1)
		self.penalty = penalty

	def predict_(self, x):
		if x.shape[1] + 1 == self.theta.shape[0]:
			x = add_intercept(x)
		self.theta = self.theta.reshape((-1, 1))
		z = x.dot(self.theta)
		a = sigmoid_(z)
		return a

	def gradient(self, x, y, lambda_=0.5):
		m = x.shape[0]

		hypothesis = self.predict_(x)
		loss = hypothesis - y
		grad = (x.T @ loss)

    
		theta_ = self.theta.copy()
		theta_[0] = 0

		if self.penalty != 'l2':
			lambda_ = 0
		regularization = lambda_ * theta_
		reg_gradient = (1/m) * (grad + regularization)

		return reg_gradient

	def fit_(self, x, y, lambda_=0.5):
		x_ = add_intercept(x)
		for i in range(self.max_iter):
			gradient = self.gradient(x_, y, lambda_).sum(axis=1)
			theta_update = (gradient * self.alpha).reshape((-1, 1))
			self.theta = self.theta - theta_update
		return self.theta

	def loss_elem_(self, x, y, lambda_=0.5):
		eps = 1e-15
		m = y.shape[0]
		y_hat = self.predict_(x)
		if self.penalty != 'l2':
			lambda_ = 0
		def cost_func(y, y_, m, lambda_):
			log_true = (y * np.log(y_ + eps))
			log_false = ((1 - y) * np.log(1 - y_ + eps))
			hyp_cost=  (log_true + log_false) / m
			reg = lambda_ * (np.dot(self.theta[1:], self.theta[1:])) / (2 * m)
			res = hyp_cost + reg
			return res
		res = np.array([cost_func(i, j, len(y), lambda_) for i, j in zip(y, y_hat)])
		return res

	def loss_(self, x, y, lambda_=0.5):
		eps = 1e-15
		m = y.shape[0]
		y_hat = self.predict_(x)

		log_true = np.dot(y.T, np.log(y_hat + eps))
		log_false = np.dot((1 - y).T, np.log((1 - y_hat) + eps))
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = np.squeeze(J)

		if self.penalty != 'l2':
			lambda_ = 0
		reg = lambda_ * (np.dot(self.theta[1:].T, self.theta[1:])) / (2 * m)

		res = J + reg
		return res

if __name__ == "__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])

	MyLR = MyLogisticRegression
	# mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09], penalty=None)
	mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09])

	print("# Example 0:")
	print(f"{mylr.predict_(X) = }")
	print()

	print("# Example 1:")
	print(f"{mylr.loss_(X,Y) = }")
	print()

	print("# Example 2:")
	print(f"{mylr.fit_(X, Y) = }")
	print(f"{mylr.theta = }")
	print()

	print("# Example 3:")
	print(f"{mylr.predict_(X) = }")
	print()

	print("# Example 4:")
	print(f"{mylr.loss_(X,Y) = }")
	print()
