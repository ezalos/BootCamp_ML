import numpy as np

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
path = os.path.join(os.path.dirname(__file__),  '..', 'ex00')
sys.path.insert(1, path)
from sigmoid import sigmoid_

def logistic_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
        theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
        y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
        with expected and compatible shapes.
        None: otherwise.
    Raises:
        This function should not raise any Exception.
    """
    x_ = add_intercept(x)
    theta = theta.reshape((-1, 1))
    z = x_.dot(theta)
    a = sigmoid_(z)
    return a


if __name__ == "__main__":# Example 1
    x = np.array([[4]])
    theta = np.array([[2], [0.5]])
    print(logistic_predict(x, theta))
    # Output:
    print("""array([[0.98201379]])""")
    # Example 1
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict(x2, theta2))
    # Output:
    print("""array([[0.98201379],
    [0.99624161],
    [0.97340301],
    [0.99875204],
    [0.90720705]])""")
    # Example 2
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict(x3, theta3))
    # Output:
    print("""array([[0.03916572],
    [0.00045262],
    [0.2890505 ]])""")


    print("\n\nCORRECTION:")


    x=np.array([0])
    theta=np.array([[0], [0]])
    print(logistic_predict(x, theta))
    print("ans = np.array([[0.5]])")
    print()

    x=np.array([1])
    theta=np.array([[1], [1]])
    print(logistic_predict(x, theta))
    print("ans = np.array([[0.880797077978]])")
    print()

    x=np.array([[1, 0], [0, 1]])
    theta=np.array([[1], [2], [3]])
    print(logistic_predict(x, theta))
    print("ans = np.array([[0.952574126822], [0.982013790038]])")
    print()
    
    x=np.array([[1, 1], [1, 1]])
    theta=np.array([[1], [2], [3]])
    print(logistic_predict(x, theta))
    print("ans = np.array([[0.997527376843], [0.997527376843]])")
    print()
