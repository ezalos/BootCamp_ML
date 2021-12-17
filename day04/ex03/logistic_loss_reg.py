import numpy as np

def reg_log_loss_(y, y_hat, theta, lambda_):
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
    eps = 1e-15

    y_true = y.T.dot(np.log(y_hat + eps))
    y_false = (1 - y).T.dot(np.log(1 - y_hat + eps))
    hyp_cost = -(y_true + y_false) / m

    reg = lambda_ * (np.dot(theta[1:], theta[1:])) / (2 * m)

    res = hyp_cost + reg
    return res

if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0])
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
    theta = np.array([1, 2.5, 1.5, -0.9])

    print("# Example 0:")
    res = reg_log_loss_(y, y_hat, theta, .5)
    print(res)
    print("""
    # Output:
    0.43377043716475955
    """)

    print("# Example 1:")
    res = reg_log_loss_(y, y_hat, theta, .05)
    print(res)
    print("""
    # Output:
    0.13452043716475953
    """)

    print("# Example 2:")
    res = reg_log_loss_(y, y_hat, theta, .9)
    print(res)
    print("""
    # Output:
    0.6997704371647596
    """)



    print("CORRECTION:")
    y = np.array([0, 1, 0, 1])
    y_hat = np.array([0.4, 0.79, 0.82, 0.04])
    theta = np.array([5, 1.2, -3.1, 1.2])

    print(f"{reg_log_loss_(y, y_hat, theta, .5) = }")
    print("Ans = 2.2006805525617885")
    print()

    print(f"{reg_log_loss_(y, y_hat, theta, .75) = }")
    print("Ans = 2.5909930525617884")
    print()

    print(f"{reg_log_loss_(y, y_hat, theta, 1.0) = }")
    print("Ans = 2.981305552561788")
    print()

    print(f"{reg_log_loss_(y, y_hat, theta, 0.0) = }")
    print("Ans = 1.4200555525617884")
    print()

