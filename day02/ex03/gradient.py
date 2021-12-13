import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of
        the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if (
        len(y.shape) != 2 or len(x.shape) != 2 or len(theta.shape) != 2 
    ):
        return None
    if (
        y.shape[1] != 1
        or x.shape[0] != y.shape[0] 
        or x.shape[1] != theta.shape[0] - 1
    ):
        return None
    # try:
    x = add_intercept(x)
    # y = y.reshape((-1,1))
    # theta = theta.reshape((-1,1))

    hypothesis = x.dot(theta)
    loss = hypothesis - y
    grad = (x.T @ loss) / x.shape[0]
    return grad
    # except:
    #     return None

if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    theta1 = np.array([3,0.5,-6]).reshape(-1, 1)
    print("# Example 0:")
    print(gradient(x, y, theta1))
    # Output:
    print("array([ -37.35714286, 183.14285714, -393. ])")
    print()

    print("# Example 1:")
    theta2 = np.array([0,0,0])
    print(gradient(x, y, theta2))
    # Output:
    print("array([ 0.85714286, 23.28571429, -26.42857143])")


    # print("Test 1")
    # x = np.random.randint(1,10,(3,3))
    # y = np.random.randint(1,8,(3,1))
    # theta = np.random.randint(1,8,(3,1))
    # print(f"{x.shape = }")
    # print(f"{y.shape = }")
    # print(f"{theta.shape = }")
    # print(f"{gradient(x, y, theta) = }")
    # print(f"Should be None")
    
    # print("Test 2")
    # x = np.random.randint(1,10,(3,3))
    # y = np.random.randint(1,8,(4,1))
    # theta = np.random.randint(1,8,(4,1))
    # print(f"{gradient(x, y, theta) = }")
    # print(f"Should be None")

    # print("Test 3")
    # x = np.random.randint(1,10,(3,3))
    # y = np.random.randint(1,8,(3,2))
    # theta = np.random.randint(1,8,(4,1))
    # print(f"{x.shape = }")
    # print(f"{y.shape = }")
    # print(f"{theta.shape = }")
    # print(f"{gradient(x, y, theta) = }")
    # print(f"Should be None")


    print("Correction follow up")
    x = np.ones(10).reshape(-1,1)
    theta = np.array([[1], [1]])
    y = np.ones(10).reshape(-1,1)
    print(f"{gradient(x, y, theta) = }")
    print("array([[1], [1]])")

    x = (np.arange(1,25)).reshape(-1,2)
    theta = np.array([[3],[2],[1]])
    y = np.arange(1,13).reshape(-1,1)
    # print(f"{x.shape = }")
    # print(f"{y.shape = }")
    # print(f"{theta.shape = }")
    print(f"{gradient(x, y, theta) = }")
    print("""array([[ 33.5       ],
       [521.16666667],
       [554.66666667]])""")

    x = (np.arange(1,13)).reshape(-1,3)
    theta = np.array([[5],[4],[-2],[1]])
    y = np.arange(9,13).reshape(-1,1)
    print(f"{gradient(x, y, theta) = }")
    print("array([[ 11. ], [ 90.5], [101.5], [112.5]])")