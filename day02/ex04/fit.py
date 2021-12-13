import numpy as np
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from tools import add_intercept
from prediction import predict_

def gradient(x, y, theta):
    x_ = x
    m = x_.shape[0]

    hypothesis = (x_ @ theta)
    loss = hypothesis - y
    gradient = (x_.T @ loss) / m

    return gradient

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
            examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training
            examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient
            descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    x_ = add_intercept(x)
    for i in range(max_iter):
        dt = gradient(x_, y, theta).sum(axis=1)
        theta_update = (dt * alpha).reshape((-1, 1))
        theta = theta - theta_update
    return theta


if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    print("# Example 0:")
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print(theta2)
    # Output:
    print("array([[41.99..],[0.97..], [0.77..], [-1.20..]])")
    print()

    print("# Example 1:")
    print(predict_(x, theta2))
    # Output:
    print("array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])")


    print("CORRECTION: ")
    x = np.arange(1,13).reshape(-1,3)
    y = np.arange(9,13).reshape(-1,1)
    theta = np.array([[5], [4], [-2], [1]])
    alpha = 1e-2
    max_iter = 10000
    print(f"{fit_(x, y, theta, alpha = alpha, max_iter=max_iter)}")
    print(f"Answer = array([[ 7.111..],[ 1.0],[-2.888..],[ 2.222..]])")

    x = np.arange(1,31).reshape(-1,6)
    theta = np.array([[4],[3],[-1],[-5],[-5],[3],[-2]])
    y = np.array([[128],[256],[384],[512],[640]])
    alpha = 1e-4
    max_iter = 42000
    print(f"{fit_(x, y, theta, alpha=alpha, max_iter=max_iter)}")
    print(f"""Answer = array([[ 7.01801797]
        [ 0.17717732]
        [-0.80480472]
        [-1.78678675]
        [ 1.23123121]
        [12.24924918]
        [10.26726714]])""")
    
