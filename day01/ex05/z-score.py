import numpy as np

def zscore(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the z-score
	standardization.
	Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	x' as a numpy.ndarray.
	None if x is a non-empty numpy.ndarray.
	Raises:
	This function shouldn't raise any Exception.
	"""
	x = x.reshape(-1, 1)
	zs = x - x.mean()
	zs = zs / x.std()
	return zs



if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(zscore(X))
	print("""array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559,
	0.17240647, -1.89647119])""")
	print()

	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	print(zscore(Y))
	print("""array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
	0.28795027, -1.72770165])""")
