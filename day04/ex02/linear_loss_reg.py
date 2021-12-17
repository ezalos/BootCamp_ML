import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized cost of a linear regression model from two non-empty
    ,→ numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized cost as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    m = y.shape[0]
    hyp_cost = (y_hat - y).dot(y_hat - y)
    reg = lambda_ * (theta[1:].T @ theta[1:])
    res = (hyp_cost + reg) / (2 * m)
    return res

if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
    theta = np.array([1, 2.5, 1.5, -0.9])

    print("# Example 0:")
    res = reg_loss_(y, y_hat, theta, .5)
    print(res)
    print("""
    # Output:
    0.8503571428571429
    """)

    print("# Example 1:")
    res = reg_loss_(y, y_hat, theta, .05)
    print(res)
    print("""
    # Output:
    0.5511071428571429
    """)

    print("# Example 2:")
    res = reg_loss_(y, y_hat, theta, .9)
    print(res)
    print("""
    # Output:
    1.116357142857143
    """)

    print("CORRECTION:")
    y=np.arange(10,100,10)
    y_hat=np.arange(9.5,95,9.5)
    theta=np.array([-10,3,8])
    lambda_=0.5
    print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
    print("Ans = 5.986111111111111")
    print()

    lambda_ = 5
    print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
    print("Ans = 24.23611111111111")
    print()

    y = np.arange(-15,15,0.1)
    y_hat = np.arange(-30,30,0.2)
    theta=np.array([42,24,12])
    lambda_=0.5
    print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
    print("Ans = 38.10083333333307")
    print()

    lambda_=8
    print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
    print("Ans = 47.10083333333307")
    print()
