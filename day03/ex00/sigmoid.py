import numpy as np

# import sys
# import os
# path = os.path.join(os.path.dirname(__file__), '..', 'utils')
# sys.path.insert(1, path)
# from tools import add_intercept


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.

    Args:
        x: has to be an numpy.array, a vector
    Return:
        The sigmoid value as a numpy.array.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    print("\n# Example 1:")
    x = np.array(-4)
    print(sigmoid_(x))
    # Output:
    print("array([[0.01798620996209156]])")
    print("\n# Example 2:")
    x = np.array(2)
    print(sigmoid_(x))
    # Output:
    print("array([[0.8807970779778823]])")
    print("\n# Example 3:")
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
    # Output:
    print("array([[0.01798620996209156], [0.8807970779778823], [0.5]])")

    print("\n\nCORRECTION:")

    x=np.array([0])
    print(sigmoid_(x))
    print("ans = array([0.5])")

    x=np.array([1])
    print(sigmoid_(x))
    print("ans = np.array([0.73105857863])")

    x=np.array([-1])
    print(sigmoid_(x))
    print("ans = np.array([0.26894142137])")

    x=np.array([50])
    print(sigmoid_(x))
    print("ans = array([1])")

    x=np.array([-50])
    print(sigmoid_(x))
    print("ans = array([1.928749847963918e-22])")

    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    print(sigmoid_(x))
    print("ans = array([0.07585818, 0.18242552, 0.37754067, 0.62245933, 0.81757448, 0.92414182])")