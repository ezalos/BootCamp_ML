class Matrix():
	def __init__(self, data_or_shape):
		if type(data_or_shape) == list:
			self._init_data(data_or_shape)
		elif type(data_or_shape) == tuple:
			self._init_shape(data_or_shape)
		else:
			raise TypeError(f"data_or_shape should be of type list or tuple")

	def _init_shape(self, shape):
		self.shape = shape
		x, y = shape
		self.data = []
		for i in range(x):
			row = [0] * y
			self.data.append(row)
	
	def _init_data(self, data):
		if type(data[0]) != list:
			raise ValueError("Data needs to have 2 dimensions, not 1")
		self.data = data
		self.shape = len(data), len(data[0])
		for i, row in enumerate(self.data):
			if len(row) != self.shape[1]:
				msg = f"Row {i} has len {len(row)} but should be {len(data[0])}"
				raise ValueError(msg)
			for a in row:
				if type(a) != int and type(a) != float:
					raise TypeError(f"data can only contain int and float, not {type(a)}")

	def T(self):
		matrix = []
		for i in range(len(self.data[0])):
			line = []
			for j in range(len(self.data)):
				line.append(self.data[j][i])
			matrix.append(line)
		return type(self)(matrix)

	# add : only matrices of same dimensions.
	def __add__(self, other):
		if type(other) != Matrix and type(other) != Vector:
			raise ValueError(f"Accepted types are Matrix and vector, not {type(other)}")
		if self.shape != other.shape:
			raise ValueError(f"Matrix addition need to have same shape {self.shape} vs {other.shape}")
		matrix = []
		for i in range(len(self.data)):
			line = []
			for j in range(len(self.data[0])):
				elem = self.data[i][j] + other.data[i][j]
				line.append(elem)
			matrix.append(line)
		return type(self)(matrix)

	def __radd__(self, other):
		raise ValueError("Addition are only between Matrix")

	# sub : only matrices of same dimensions.
	def __sub__(self, other):
		return self.__add__(other * -1)

	def __rsub__(self, other):
		raise ValueError("Substraction are only between Matrix")

	# div : only scalars.
	def __truediv__(self, other):
		if type(other) != int and type(other) != float:
			raise TypeError(f"true div can only be done with scalar")
		if other == 0:
			raise ZeroDivisionError
		matrix = []
		for i in range(len(self.data)):
			line = []
			for j in range(len(self.data[0])):
				elem = self.data[i][j] / other
				line.append(elem)
			matrix.append(line)
		return type(self)(matrix)


	def __rtruediv__(self, other):
		raise ValueError("A scalar can't be divided by a Matrix")

	# mul : scalars, vectors and matrices , can have errors with vectors and matrices,
	# returns a Vector if we perform Matrix * Vector mutliplication.
	def __mul__(self, other):
		if type(other) == int or type(other) == float:
			matrix = []
			for i in range(len(self.data)):
				line = []
				for j in range(len(self.data[0])):
					elem = self.data[i][j] * other
					line.append(elem)
				matrix.append(line)
			return type(self)(matrix)
		elif type(other) == Matrix or type(other) == Vector:
			if self.shape[1] != other.shape[0]:
				raise ValueError(f"Incompatible shapes for matmult: {self.shape} and {other.shape}")
			return self.__matmult__(other)
		else:
			raise TypeError(f"Wrong type {type(other)}, supported types: Matrix, Vector, int and float.")

	def __matmult__(self, other):
		matrix = []
		for i in range(len(self.data)):
			line = []
			for j in range(len(other.data[0])):
				elem = 0
				for k in range(len(other.data)):
					elem += self.data[i][k] * other.data[k][j]
				line.append(elem)
			matrix.append(line)
		m = Matrix(matrix)
		if 1 in m.shape:
			m = Vector(m.data)
		return m

	def __rmul__(self, other):
		if type(other) == int or type(other) == float:
			return self.__mul__(other)
		else:
			raise TypeError(f"Wrong type {type(other)}, rmul supported types are int and float.")

	def __str__(self):
		RED = '\033[91m'
		GREEN = '\033[92m'
		BLUE = '\033[94m'
		RESET = '\033[0m'
		spacing = 1
		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				f_len = len(str(self.data[i][j]))
				spacing = max(f_len, spacing)
		spacing += 1
		out = ""
		out += f"{self.__class__.__name__}:\n"
		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				e = str(self.data[i][j])
				out += (spacing - len(e)) * ' ' + e
			out += "\n"
		out += f"Shape: {self.shape}\n"
		return out


class Vector(Matrix):
	def __init__(self, data_or_shape):
		super(Vector, self).__init__(data_or_shape)
		if self.shape[0] != 1 and self.shape[1] != 1:
			raise ValueError(f"Vector shape can't be {self.shape}, use Matrix instead")

	def dot(self, other):
		if type(other) != Vector:
			raise TypeError(
				f"Dot product is only between Vectors, not with {type(other)}")
		if self.shape != other.shape:
			raise ValueError(f"Dot product cant be done on vector of different shape")

		if self.shape[0] != 1:
			a, b = self.T(), other.T()
		else:
			a, b = self, other

		data = []
		for i in a.data[0]:
			nb = 0
			for ii in b.data[0]:
				nb += i * ii
			data.append(nb)
		
		v = Vector([data])
		if self.shape[0] != 1:
			v = v.T()
		return v

	# def __mul__(self, other):
	# 	if type(other) == Vector:
	# 		matrix = []
	# 		for i in range(len(self.data)):
	# 			line = []
	# 			for j in range(len(self.data[0])):
	# 				elem = self.data[i][j] * other
	# 				line.append(elem)
	# 			matrix.append(line)
	# 		return type(self)(matrix)
	# 	elif type(other) == Matrix or type(other) == Vector:
	# 		if self.shape[1] != other.shape[0]:
	# 			raise ValueError(
	# 				f"Incompatible shapes for matmult: {self.shape} and {other.shape}")
	# 		return self.__matmult__(other)
	# 	else:
	# 		raise TypeError(
	# 			f"Wrong type {type(other)}, supported types: Matrix, Vector, int and float.")
