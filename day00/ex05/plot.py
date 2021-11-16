import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'ex04')
sys.path.insert(1, path)
from prediction import predict_
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
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
	plt.plot(x, y, 'o')

	plt.plot(x, predict_(x, theta))
	plt.show()
	pass

if __name__ == "__main__":
	x = np.arange(1,6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	#Example 1:
	theta1 = np.array([4.5, -0.2])
	plot(x, y, theta1)
	# Output:

	#Example 2:
	theta2 = np.array([-1.5, 2])
	plot(x, y, theta2)
	# Output:

	# Example 3:
	theta3 = np.array([3, 0.3])
	plot(x, y, theta3)
	# Output:

	# plot(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
	# plot(np.array([0, 1]), np.array([0, 1]), np.array([1, 1]))
	# plot(np.array([0, 2]), np.array([0, 0]), np.array([-1, 1]))
