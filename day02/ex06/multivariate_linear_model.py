import pandas as pd
import numpy as np
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from my_linear_regression import MyLinearRegression as MyLR

if __name__ == "__main__":
	file_path = "spacecraft_data.csv"
	file_path = "day02/ex06/spacecraft_data.csv"
	data = pd.read_csv(file_path)
	X = np.array(data[['Age','Thrust_power','Terameters']])
	X_1 = np.array(data[['Age']])
	X_2 = np.array(data[['Thrust_power']])
	X_3 = np.array(data[['Terameters']])
	Y = np.array(data[['Sell_price']])

	print("Ex 1.a")
	print("Sell_price vs Age")
	my_lreg = MyLR([1.0, 1.0], alpha=1e-3, max_iter=60000)
	my_lreg.fit_(X_1, Y)
	print(f"{my_lreg.theta = }")
	print(f"Answer should be close to [[647.04], [-12.99]]")
	print(f"MSE Score: {my_lreg.mse_(Y, my_lreg.predict(X_1)).mean()}")
	# my_lreg.scatter(X_1, Y)
	
	print("Ex 1.b")
	print("Sell_price vs Thrust_power")
	my_lreg = MyLR([1.0, 1.0], alpha=1e-4, max_iter=60000)
	my_lreg.fit_(X_2, Y)
	print(f"{my_lreg.theta = }")
	print(f"Answer should be close to [[39.88920787  4.32705362]]")
	print(f"MSE Score: {my_lreg.mse_(Y, my_lreg.predict(X_2)).mean()}")
	# my_lreg.scatter(X_2, Y)
	
	print("Ex 1.c")
	print("Sell_price vs Terameters")
	my_lreg = MyLR([1.0, 1.0], alpha=1e-4, max_iter=500000)
	my_lreg.fit_(X_3, Y)
	print(f"{my_lreg.theta = }")
	print(f"Answer should be close to [[744.67913252  -2.86265137]]")
	print(f"MSE Score: {my_lreg.mse_(Y, my_lreg.predict(X_3)).mean()}")
	# my_lreg.scatter(X_3, Y)

	my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
	# print(my_lreg.mse_(X,Y).sum() / 2)
	# print("144044.877...")
	# print()

	print("CORRECTION: A new hope:")
	my_lreg.fit_(X,Y)
	print(my_lreg.theta)
	# print("array([[334.994...],[-22.535...],[5.857...],[-2.586...]])")
	print("array([[383.94598482 -24.29984158   5.67375056  -2.66542654]])")
	print(f"MSE Score: {my_lreg.mse_(Y, my_lreg.predict(X)).mean()}")
	print()
	my_lreg.multi_scatter(X,Y)

	# print(my_lreg.mse_(X,Y))
	# print("586.896999...")
	# print()
