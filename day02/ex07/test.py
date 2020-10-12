import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

if __name__ == "__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

	print("# Example 0:")
	print(mylr.predict(X))
	print("# Output:")
	print("array([[8.], [48.], [323.]])")
	print()

	print("# Example 1:")
	print(mylr.cost_elem_(X,Y))
	print("# Output:")
	print("array([[37.5], [0.], [1837.5]])")
	print()

	print("# Example 2:")
	print(mylr.cost_(X,Y))
	print("# Output:")
	print(1875.0)
	print()

	# sys.lol()
	print("# Example 3:")
	mylr.fit_(X, Y)
	print(mylr.theta)
	print("# Output:")
	print("array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])")
	print()

	print("# Example 4:")
	print(mylr.predict(X))
	print("# Output:")
	print("array([[23.499..], [47.385..], [218.079...]])")
	print()

	print("# Example 5:")
	print(mylr.cost_elem_(X,Y))
	print("# Output:")
	print("array([[0.041..], [0.062..], [0.001..]])")
	print()

	print("# Example 6:")
	print(mylr.cost_(X,Y))
	print("# Output:")
	print("0.1056..")
