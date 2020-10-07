from matrix import Matrix
import numpy as np

def test_error(data, shape, msg=""):
	m_err = None
	try:
		print("TEST " + msg, "\n\tData: ", data, "\n\tShape: ", shape)
		m_err = Matrix(data, shape)
	except Exception as e:
		print(e, "\nRES: ", m_err)
		print("SUCCESS\n")
		return True
	else:
		print("RES: ", m_err)
		print("FAILURE\n")
		return False



test_error([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0]], None, "Diff size lines")
test_error([[],[],[]], None, "Empty empty")
test_error([], None, "Empty")
test_error(None, None, "None None")
test_error(None, (1,2,3), "Tuple 3")
test_error(None, (1,), "Tuple 1")
test_error(None, (), "Tuple void")

print()
print()

def test_init(data, shape, msg=""):
	print("INIT TEST: ", msg)
	print("Data: ", data)
	print("Shape: ", shape)
	m = Matrix(data, shape)
	print(m)
	print()

test_init(None, (2,3))
test_init([[0]], (2,3))
test_init([[0.0, 1.0, 2.0, 3.0],
[0.0, 2.0, 4.0, 6.0]], None)
test_init([[0.0, 1.0],
[2.0, 3.0],
[4.0, 5.0],
[6.0, 7.0]], None)

print()
print()

def test_valid(v1, v2, ope):
	v1_me = Matrix(v1)
	v1_np = np.array(v1)

	v2_me = Matrix(v2)
	v2_np = np.array(v2)

	if ope == "+":
		res_me = v1_me + v2_me
		res_np = v1_np + v2_np
	elif ope == "-":
		res_me = v1_me - v2_me
		res_np = v1_np - v2_np
	elif ope == "*":
		res_me = v1_me * v2_me
		res_np = v1_np.dot(v2_np)
	elif ope == "/":
		res_me = v1_me / v2_me
		res_np = v1_np / v2_np
	else:
		print("Wrong ope")
		return

	print("test_valid")
	if res_me.tolist() == res_np.tolist():
		print("SUCCESS")
	else:
		print("Mat_1: ", v1)
		print("Mat_1: ", v2)
		print("OPE: ", ope)
		print("NP: ", res_np)
		print("ME: ", res_me)
		print("FAILURE")
	print()

test_valid([[0.0, 1.0, 2.0, 3.0],
[0.0, 2.0, 4.0, 6.0]], [[0.0, 1.0],
[2.0, 3.0],
[4.0, 5.0],
[6.0, 7.0]], "*")
