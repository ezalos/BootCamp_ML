import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

if __name__ == "__main__":
	data = pd.read_csv("spacecraft_data.csv")
	X = np.array(data[['Age','Thrust_power','Terameters']])
	Y = np.array(data[['Sell_price']])
	my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], alpha=1e-5, max_iter=6000000)

	print(my_lreg.mse_(X,Y).sum() / 2)
	print("144044.877...")
	print()

	my_lreg.fit_(X,Y)
	print(my_lreg.theta)
	print("array([[334.994...],[-22.535...],[5.857...],[-2.586...]])")
	print()

	print(my_lreg.mse_(X,Y))
	print("586.896999...")
	print()
