import math
import numpy as np

class TinyStatistician():
	def mean(self, x):
		if len(x) == 0:
			return None
		_sum = 0.0
		for i in x:
			_sum += i
		return float(_sum) / len(x)

	def median(self, x):
		size = len(x)
		if size % 2 == 1:
			return sorted(x)[size // 2]
		else:
			x = sorted(x)
			a = x[(size // 2) - 1]
			b = x[size // 2]
			res = (a + b) / 2
			return res


	def percentile(self, data, perc: int):
		size = len(data)
		return sorted(data)[int(math.ceil((size * perc) / 100)) - 1]

	def quartile(self, x):
		if len(x) == 0:
			return None
		
		q1 = self.percentile(x, 25)
		q3 = self.percentile(x, 75)

		return [q1, q3]

	def var(self, x):
		if len(x) == 0:
			return None
		mean = self.mean(x)

		res = 0.0
		for i in x:
			res += float(i - mean) ** 2.0

		res = res / float(len(x))
		return res

	def std(self, x):
		if len(x) == 0:
			return None
		res = self.var(x) ** 0.5
		return res
