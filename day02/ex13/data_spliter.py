import numpy as np

def data_spliter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set,
		while respecting the given proportion of examples to be kept in the traning set.
	Args:
		x: has to be an numpy.ndarray, a matrix of dimension m * n.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		proportion: has to be a float, the proportion of the dataset that will be assigned to
			the training set.
	Returns:
		(x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray
		None if x or y is an empty numpy.ndarray.
		None if x and y do not share compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	pass

if __name__ == "__main__":
	x1 = np.array([1, 42, 300, 10, 59])
	y = np.array([0,1,0,1,0])
	# Example 1:
	data_spliter(x1, y, 0.8)
	# Output:
	(array([ 1, 59, 42, 300]), array([10]), array([0, 0, 0, 1]), array([1]))


	# Example 2:
	data_spliter(x1, y, 0.5)
	# Output:
	(array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 1, 0]))
	x2 = np.array([ [ 1, 42],
	[300, 10],
	[ 59, 1],
	[300, 59],
	[ 10, 42]])
	y = np.array([0,1,0,1,0])


	# Example 3:
	data_spliter(x2, y, 0.8)
	# Output:
	(array([[ 10, 42],
	[300, 59],
	[ 59, 1],
	[300, 10]]), array([[ 1, 42]]), array([0, 1, 0, 1]), array([0]))


	# Example 4:
	data_spliter(x2, y, 0.5)
	# Output:
	(array([[59, 1],
	[10, 42]]), array([[300, 10],
	[300, 59],
	[ 1, 42]]), array([0, 0]), array([1, 1, 0])

	)
	# Be careful! The way tuples of arrays are displayed could be a bit confusing...
	#
	# In the last example, the tuple returned contains the following arrays:
	# array([[59, 1],
	# [10, 42]])
	#
	# array([[300, 10],
	# [300, 59]
	#
	# array([0, 0])
	#
	# array([1, 1, 0]))
