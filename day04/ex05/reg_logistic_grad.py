import numpy as np

def add_intercept(x):
    vec_one = np.ones(x.shape[0])
    result = np.column_stack((vec_one, x))
    return result

def predict_(x, theta):
    def sigmoid_(x):
        return 1 / (1 + np.exp(-x))
    z = x.dot(theta)
    a = sigmoid_(z)
    return a

def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
    with two for-loops. The three arrays must have compatible shapes.

    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    # return None
    return vec_reg_logistic_grad(y, x, theta, lambda_)

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
    without any for-loop. The three arrays must have compatible shapes.

    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    x_ = add_intercept(x)
    m = y.shape[0]

    # hypothesis = (x_ @ theta)
    hypothesis = predict_(x_, theta)
    loss = hypothesis - y
    gradient = x_.T @ loss
    
    theta_ = theta.copy()
    theta_[0] = 0

    regularization = lambda_ * theta_
    reg_gradient = (1/m) * (gradient + regularization)

    return reg_gradient

if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
            [2, 4, 5, 5],
            [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("# Example 1.1:")
    print(f"{reg_logistic_grad(y, x, theta, 1) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-1.40334809],
    [-1.91756886],
    [-2.56737958],
    [-3.03924017]])""")
    print()

    print("# Example 1.2:")
    print(f"{vec_reg_logistic_grad(y, x, theta, 1) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-1.40334809],
    [-1.91756886],
    [-2.56737958],
    [-3.03924017]])""")
    print()
    
    print("# Example 2.1:")
    print(f"{reg_logistic_grad(y, x, theta, 0.5) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-1.15334809],
    [-1.96756886],
    [-2.33404624],
    [-3.15590684]])""")
    print()
    
    print("# Example 2.2:")
    print(f"{vec_reg_logistic_grad(y, x, theta, 0.5) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-1.15334809],
    [-1.96756886],
    [-2.33404624],
    [-3.15590684]])""")
    print()
    
    print("# Example 3.1:")
    print(f"{reg_logistic_grad(y, x, theta, 0.0) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-0.90334809],
    [-2.01756886],
    [-2.10071291],
    [-3.27257351]])""")
    print()
    
    print("# Example 3.2:")
    print(f"{vec_reg_logistic_grad(y, x, theta, 0.0) = }")
    print("# Output:")
    print("""array([[-0.55711039],
    [-0.90334809],
    [-2.01756886],
    [-2.10071291],
    [-3.27257351]])""")
    print()
    
    print("CORRECTION:")
    x = np.array([[0, 2], [3, 4],[2, 4], [5, 5],[1, 3], [2, 7]])
    y = np.array([[0], [1], [1], [0], [1], [0]])
    theta = np.array([[-24.0], [-15.0], [3.0]])
    
    print(f"{vec_reg_logistic_grad(y, x, theta, 0.5) = }")
    print("""array([[-0.5       ],
            [-2.25      ],
            [-1.58333333]])""")
    print()

    print(f"{vec_reg_logistic_grad(y, x, theta, 0.05) = }")
    print("""array([[-0.5       ],
            [-1.125     ],
            [-1.80833333]])""")
    print()

    print(f"{vec_reg_logistic_grad(y, x, theta, 2.0) = }")
    print("""array([[-0.5       ],
            [-6.        ],
            [-0.83333333]])""")
    print()
