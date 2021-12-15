import numpy as np

import sys
import os
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
path = os.path.join(os.path.dirname(__file__),  '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict

def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Return:
        The logistic loss value as a float.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    e = 1e-15
    m = y.shape[0]

    eps = e
    log_true = np.dot(y.T, np.log(y_hat + eps))

    log_false = np.dot((1 - y).T, np.log((1 - y_hat) + eps))

    log_prob = log_true + log_false

    J = - (1 / m) * log_prob
    J = np.squeeze(J)

    return J


if __name__ == "__main__":
    
    print("# Example 1:")
    y1 = np.array([[1]])
    x1 = np.array([[4]])
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict(x1, theta1)
    print(f"{log_loss_(y1, y_hat1) = }")
    print("# Output:")
    print("0.01814992791780973")
    print()

    print("# Example 2:")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict(x2, theta2)
    print(f"{log_loss_(y2, y_hat2) = }")
    print("# Output:")
    print("2.4825011602474483")
    print()

    print("# Example 3:")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict(x3, theta3)
    print(f"{log_loss_(y3, y_hat3) = }")
    print("# Output:")
    print("2.9938533108607053")
    print()

    print("CORRECTION:")
    y=np.array([[0], [0]])
    y_hat=np.array([[0], [0]])
    print(f"{log_loss_(y, y_hat) = }")
    print("Ans = 0")
    print()

    y=np.array([[0], [1]])
    y_hat=np.array([[0], [1]])
    print(f"{log_loss_(y, y_hat) = }")
    print("Ans = 0")
    print()

    y=np.array([[0], [0], [0]])
    y_hat=np.array([[1], [0], [0]])
    print(f"{log_loss_(y, y_hat) = }")
    print("Ans = 11.51292546")
    print()

    y=np.array([[0], [0], [0]])
    y_hat=np.array([[1], [0], [1]])
    print(f"{log_loss_(y, y_hat) = }")
    print("Ans = 23.02585093")
    print()

    y=np.array([[0], [1], [0]])
    y_hat=np.array([[1], [0], [1]])
    print(f"{log_loss_(y, y_hat) = }")
    print("Ans = 34.53877639")
    print()
