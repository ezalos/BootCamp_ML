import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict(Xpill)


def continuous_plot(x, y, i, lr):
	# Build the model:
	# Plot:
	## To get a smooth curve, we need a lot of data points
	continuous_x = np.arange(1,7.01, 0.01).reshape(-1,1)
	x_ = add_polynomial_features(continuous_x, i)
	y_hat = lr.predict(x_)
	print(x.shape, y.shape)
	plt.scatter(x.T[0],y)
	plt.plot(continuous_x, y_hat, color='orange')
	plt.show()


cost = []

x = add_polynomial_features(Xpill, 10)
big_theta = [[ 2.03333758e-06],
 [ 4.76503382e-06],
 [ 1.29939248e-05],
 [ 3.79946877e-05],
 [ 1.12691614e-04],
 [ 3.25797609e-04],
 [ 8.76644495e-04],
 [ 2.01101984e-03],
 [ 3.02151256e-03],
 [-1.12991082e-03],
 [ 9.48325917e-05]]

# big_theta_futur = [[ 2.07037841e-06],
#  [ 4.83925060e-06],
#  [ 1.31593092e-05],
#  [ 3.83642999e-05],
#  [ 1.13422797e-04],
#  [ 3.26767863e-04],
#  [ 8.75990025e-04],
#  [ 2.00179965e-03],
#  [ 2.99573196e-03],
#  [-1.12062352e-03],
#  [ 9.40458287e-05]]
# big_theta += (np.array(big_theta_futur) - np.array(big_theta)) * 10
# big_theta = big_theta.tolist()

# lr = MyLR(thetas=big_theta, alpha=5e-16, max_iter=10000000)
# lr.fit_(x, Yscore)

for i in range(2, 5):
	x = add_polynomial_features(Xpill, i)
	lr = MyLR(thetas=[0] * (i + 1), alpha=1 / (100 * (10 ** i)), max_iter=1000 * (10 ** i))
	lr.fit_(x, Yscore)
	cost.append(lr.cost)
	plt.ioff()
	continuous_plot(x, Yscore, i, lr)

for i, c in enumerate(cost):
	plt.plot(c,label="Polynomial of degree " + str(i + 2))
plt.legend(loc='best')
plt.show()
