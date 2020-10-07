import sys
sys.path.insert(1, '/home/ezalos/42/Bootcamp_Python/bootcamp_machine-learning/day00/ex00')
from vector import Vector

class Matrix():
	def fill_from_data(self, data):
		# print("fill_from_data")
		if type([]) != type(data):
			raise ValueError("ERROR: Data should be a list, not ", type(shape))
		len_1 = len(data)
		self.data = []
		for i in data:
			elem = Vector(i)
			if elem == None:
				raise ValueError("ERROR: Line " + str(i) + "cant be converted to vector")
			self.data.append(elem)
			if self.data[0].length != self.data[-1].length:
				raise ValueError("ERROR: All lines of data should have same length")
		self.shape = (len_1, self.data[0].length)
		return self

	def fill_from_shape(self, shape):
		# print("fill_from_shape")
		if type(()) != type(shape):
			raise ValueError("ERROR: Shape should be a tuple, not ", type(shape))
		if len(shape) != 2:
			raise ValueError("ERROR: Shape should be of len 2")
		self.data = []
		for lin in range(shape[0]):
			elem = [0.0] * shape[1]
			elem = Vector(elem)
			# print(elem)
			if elem == None:
				raise ValueError("ERROR: Line " + str(i) + "cant be converted to vector")
			self.data.append(elem)
		self.shape = shape
		return self

	def __init__(self, data = [], shape = ()):
		# print("Data: ", data)
		# print("Shape: ", shape)
		if data != None:
			if self.fill_from_data(data) == None:
				raise ValueError
		elif shape != None:
			if self.fill_from_shape(shape) == None:
				raise ValueError
		else:
			raise ValueError("You need to specify either data or shape")
		# print(self.__str__())
		# print()
		# print()

	def tolist(self):
		data = []
		for i in self.data:
			data.append(i.values)
		return data

	def T(self):
		data = []
		for lin in range(self.shape[1]):
			data.append([])
			for col in range(self.shape[0]):
				data[lin].append(self.data[col].values[lin])
		shape = (self.shape[1], self.shape[0])
		return Matrix(data, shape)

	def __add__(self, other):
		new_mat = []
		if type(other) == type(self):
			if other.shape == self.shape:
				for i in range(self.shape[0]):
					new_mat.append(self.data[i] + other.data[i])
					if new_mat[-1] == None:
						return None
		elif type(other) == type(Vector()):
			for i in range(self.shape[0]):
				new_mat.append(self.data[i] + other)
		else:
			return None
		return Matrix(new_mat)


	def __radd__(self, other):
		return self.__add__(other)
	# add : scalars and vectors, can have errors with vectors.


	def __sub__(self, other):
		if type(other) == type(self):
			new_mat = []
			for i in other.data:
				new_mat.append(-1 * i)
				if new_mat[-1] == None:
					return None
			return self.__add__(Matrix(new_mat))
		elif type(other) == type(Vector()):
			neg_other = other * -1
			return self.__add__(neg_other)
		return None

	def __rsub__(self, other):
		new_mat = []
		for i in self.data:
			new_mat.append(-1 * i)
			if new_mat[-1] == None:
				return None
		return Matrix(new_mat).__add__(other)
	# sub : scalars and vectors, can have errors with vectors.


	def __truediv__(self, other):
		if type(other) == type(0) or type(other) == type(0.0):
			new_mat = []
			for i in self.data:
				new_mat.append(other / i)
				if new_mat[-1] == None:
					return None
			return Matrix(new_mat)
		return None


	def __rtruediv__(self, other):
		if type(other) == type(0) or type(other) == type(0.0):
			new_mat = []
			for i in self.data:
				new_mat.append(i / other)
				if new_mat[-1] == None:
					return None
			return Matrix(new_mat)
		return None
	# div : only scalars.


	def __mul__(self, other):
		if type(other) == type(0) or type(other) == type(0.0):
			new_mat = []
			for i in self.data:
				new_mat.append(other * i)
				if new_mat[-1] == None:
					return None
			return Matrix(new_mat)
		elif type(other) == type(Vector([0.0])):
			new_mat = []
			for i in self.data:
				new_mat.append(i * other)
				if new_mat[-1] == None:
					return None
			return Vector(new_mat)
		elif type(other) == type(Matrix([[0.0]])):
			if self.shape[1] != other.shape[0]:
				print("ERROR: Matrix multiplication needs compatible sizes")
				return None
			new_mat = []
			secn = other.T()
			for i, v_1 in enumerate(self.data):
				new_mat.append([])
				for j, v_2 in enumerate(secn.data):
					new_mat[i].append(v_1 * v_2)
			return Matrix(new_mat)
		return None

	def __rmul__(self, other):
		return self.__mul__(other)
	# mul : scalars and vectors, can have errors with vectors,
	# return a scalar is we perform Vector * Vector (dot product)


	def __str__(self):
		data = "["
		for i in self.data:
			data += str(i.values)
			data += ", "
		if len(data) > 1:
			data = data[:-2]
		data += "]"
		msg = "Matrix: " + str(self.shape) + " " + data
		return msg
