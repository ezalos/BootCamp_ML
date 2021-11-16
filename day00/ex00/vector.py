from matrix import Matrix

class Vector(Matrix):
	def __init__(self, data_or_shape):
		super(Vector, self).__init__(data_or_shape)
		if self.shape[0] != 1 and self.shape[1] != 1:
			raise ValueError(f"Vector shape can't be {self.shape}, use Matrix instead")

	def dot(self, other):
		if type(other) != Vector:
			raise TypeError(f"Dot product is only between Vectors, not with {type(other)}")
		if self.shape != other.shape:
			raise ValueError(f"Dot product cant be done on vector of different shape")

		a = self.T() if self.shape[0] != 1 else self
		b = other.T() if other.shape[0] != 1 else other

		data = []
		for i in a.data[0]:
			nb = 0
			for ii in b.data[0]:
				nb += i * ii
			data.append(nb)
		return Vector([data])

