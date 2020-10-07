import numpy as np

class TinyStatistician():
	def mean(self, x):
		if len(x) == 0:
			return None
		sum = 0.0
		for i in x:
			sum += i
		return float(sum / len(x))

	def median(self, x):
		return self.quartile(x, 2)

	def quartile(self, x, percentile):
		if len(x) == 0:
			return None

		x = sorted(x)
		mid = len(x) / 4
		if percentile == 1 or percentile == 25:
			percentile = 1
		elif percentile == 2 or percentile == 50:
			percentile = 2
		elif percentile == 3 or percentile == 75:
			percentile = 3
		else:
			return None

		mid = float(percentile * len(x) + (4 - percentile)) / 4.0

		sec = mid - int(mid)
		mid = int(mid) - 1

		return (x[mid] * (1 - sec) + x[mid + 1] * sec)

	def var(self, x):
		if len(x) == 0:
			return None
		mean = self.mean(x)
		var = 0.0
		for i in x:
			var += (i - mean) ** 2
		var /= float(len(x))
		return var

	def std(self, x):
		if len(x) == 0:
			return None
		return self.var(x) ** 0.5

if __name__ == "__main__":
	tstat = TinyStatistician()
	a = [1, 42, 300, 10, 59]
	print("Values: ", a)
	print()

	print("Mean: ")
	print(tstat.mean(a))
	print(82.4)
	print()

	print("median: ")
	print(tstat.median(a))
	print(42.0)
	print()

	print("quartile(a, 25): ")
	print(tstat.quartile(a, 25))
	print(10.0)
	print()

	print("quartile(a, 75): ")
	print(tstat.quartile(a, 75))
	print(59.0)
	print()

	print("var: ")
	print(tstat.var(a))
	print(12279.439999999999)
	print()

	print("std: ")
	print(tstat.std(a))
	print(110.81263465868862)
	print()
