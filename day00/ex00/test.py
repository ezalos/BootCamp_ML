# python3 -m unittest -v test.py
from vector import Vector
import numpy as np
import unittest

class VectorInit(unittest.TestCase):
    def test_init_tab(self):
        test = [1, 2, 1, 3]
        self.assertEqual(Vector(test).values, test)

    def test_init_tuple(self):
        test = (-2, 5)
        self.assertEqual(Vector(test).values, list(range(test[0], test[1])))

    def test_init_int(self):
        test = 2
        self.assertEqual(Vector(test).values, list(range(test)))

class VectorAdd(unittest.TestCase):
	def test_addition_1(self):
		v1 = [0.0, 1.0, 2.0, 3.0]
		v2 = [10.0, 11.0, 12.0, 10]

		v1_me = Vector(v1)
		v1_np = np.array(v1)

		v2_me = Vector(v2)
		v2_np = np.array(v2)

		res_me = v1_me + v2_me
		res_np = v1_np + v2_np

		self.assertEqual(res_me.values, res_np.tolist())

	def test_addition_2(self):
		v1 = [0.0]
		v2 = [10.0]

		v1_me = Vector(v1)
		v1_np = np.array(v1)

		v2_me = Vector(v2)
		v2_np = np.array(v2)

		res_me = v1_me + v2_me
		res_np = v1_np + v2_np

		self.assertEqual(res_me.values, res_np.tolist())

	def test_addition_int(self):
		v1 = [0.0]

		v1_me = Vector(v1)
		v1_np = np.array(v1)

		res_me = v1_me + 2
		res_np = v1_np + 2

		self.assertEqual(res_me.values, res_np.tolist())

	def test_addition_int_rev(self):
		v1 = [0.0]

		v1_me = Vector(v1)
		v1_np = np.array(v1)

		res_me = 2 + v1_me
		res_np = 2 + v1_np

		self.assertEqual(res_me.values, res_np.tolist())


v1 = [0.0, 1.0, 2.0, 3.0]
v1_me = Vector(v1)
v1_np = np.array(v1)

v2 = [10.0, 11.0, 12.0, 10]
v2_me = Vector(v2)
v2_np = np.array(v2)


print(v1_me)
print(v2_me)

print("\nADD")
print(v1_me + v2_me)
print(v1_me + 2)
print(1 + v2_me)

print("\nSUB")
print(v1_me - v2_me)
print(v1_me - 2)
print(1 - v2_me)

print("\nDIV")
print(v1_me / v2_me)
print(10 / v2_me)
print(0 / v2_me)
print(v2_me / 0)
print(v2_me / 10)

print("\nMUL")
print(v1_me)
print(v2_me)
print(v2_me * v1_me)
print(v2_me * 10)
print(10 * v2_me)
print()

if __name__ == '__main__':
    unittest.main()
