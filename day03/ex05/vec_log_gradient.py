import numpy as np

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
path = os.path.join(os.path.dirname(__file__),  '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict

def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
        containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]

    x_ = add_intercept(x)
    
    # hypothesis = x_.dot(theta)
    hypothesis = logistic_predict(x, theta)
    loss = hypothesis - y
    grad = (x_.T @ loss) / m

    return grad


if __name__ == "__main__":
    
    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    print(f"{vec_log_gradient(x1, y1, theta1) = }")
    print("# Output:")
    print("""array([[-0.01798621],
    [-0.07194484]])""")
    print()

    print("# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(f"{vec_log_gradient(x2, y2, theta2) = }")
    print("# Output:")
    print("""array([[0.3715235 ],
    [3.25647547]])""")
    print()

    print("# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(f"{vec_log_gradient(x3, y3, theta3) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-0.90334809],
    [-2.01756886],
    [-2.10071291],
    [-3.27257351]])""")
    print()



    print("CORRECTION:")
    print("Test:")
    x=np.array([[0, 0], [0, 0]])
    y=np.array([[0], [0]])
    theta=np.array([[0], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")
    print("Ans = [[0.5], [0], [0]]")
    print()

    print("Test:")
    x=np.array([[1, 1], [1, 1]])
    y=np.array([[0], [0]])
    theta=np.array([[1], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")
    print("Ans = [[0.73105858], [0.73105858], [0.73105858]]")
    print()
    
    print("Test:")
    x=np.array([[1, 1], [1, 1]])
    y=np.array([[1], [1]])
    theta=np.array([[1], [0], [0]])
    print(f"{vec_log_gradient(x, y, theta) = }")
    print("Ans = [[-0.2689414213], [-0.2689414213], [-0.2689414213]]")
    print()
    
    print("Test:")
    x=np.array([[1, 1], [1, 1]])
    y=np.array([[0], [0]])
    theta=np.array([[1], [1], [1]])
    print(f"{vec_log_gradient(x, y, theta) = }")
    print("Ans = [[0.95257412682], [0.95257412682], [0.95257412682]]")
    print()
    