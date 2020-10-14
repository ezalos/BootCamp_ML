import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

if __name__ == "__main__":
	data = pd.read_csv("spacecraft_data.csv")
	X = np.array(data[['Age','Thrust_power','Terameters']])
	X_1 = np.array(data[['Age']])
	X_2 = np.array(data[['Thrust_power']])
	X_3 = np.array(data[['Terameters']])
	Y = np.array(data[['Sell_price']])

	# print("Ex 1.a")
	# my_lreg = MyLR([1.0, 1.0], alpha=1e-3, max_iter=60000)
	# my_lreg.fit_(X_1, Y)
	# my_lreg.scatter(X_1, Y)
	#
	# print("Ex 1.b")
	# my_lreg = MyLR([1.0, 1.0], alpha=1e-4, max_iter=60000)
	# my_lreg.fit_(X_2, Y)
	# my_lreg.scatter(X_2, Y)
	#
	# print("Ex 1.c")
	# my_lreg = MyLR([1.0, 1.0], alpha=1e-4, max_iter=500000)
	# my_lreg.fit_(X_3, Y)
	# my_lreg.scatter(X_3, Y)

	my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
	# print(my_lreg.mse_(X,Y).sum() / 2)
	# print("144044.877...")
	# print()

	my_lreg.fit_(X,Y)
	print(my_lreg.theta)
	print("array([[334.994...],[-22.535...],[5.857...],[-2.586...]])")
	print()
	my_lreg.multi_scatter(X,Y)

	# print(my_lreg.mse_(X,Y))
	# print("586.896999...")
	# print()
