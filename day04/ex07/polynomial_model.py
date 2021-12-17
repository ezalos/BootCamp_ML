import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in
        argument.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to
            be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he
            polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    return np.concatenate([x ** (i + 1) for i in range(power)], axis=1)

if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)
    print("# Example 1:")
    print(add_polynomial_features(x, 3))
    # Output:
    print("""array([[ 1, 1, 1],
    [ 2, 4, 8],
    [ 3, 9, 27],
    [ 4, 16, 64],
    [ 5, 25, 125]])""")
    print("# Example 2:")
    print(add_polynomial_features(x, 6))
    # Output:
    print("""array([[ 1, 1, 1, 1, 1, 1],
    [ 2, 4, 8, 16, 32, 64],
    [ 3, 9, 27, 81, 243, 729],
    [ 4, 16, 64, 256, 1024, 4096],
    [ 5, 25, 125, 625, 3125, 15625]])""")


    print("CORRECTION:")
    print("\nTest 1:")
    x1 = np.arange(1,6).reshape(-1,1)
    print(f"{x1 = }")
    x1_poly = add_polynomial_features(x1, 5)
    print(f"{x1_poly = }")
    print("""array([[ 1,  1,   1,   1,    1],
               [ 2,  4,   8,  16,   32],
               [ 3,  9,  27,  81,  243],
               [ 4, 16,  64, 256, 1024],
               [ 5, 25, 125, 625, 3125]])""")

    print("\nTest 2:")
    x2 = np.arange(10,40, 10).reshape(-1,1)
    print(f"{x2 = }")
    x2_poly = add_polynomial_features(x2, 5)
    print(f"{x2_poly = }")
    print("""array([[       1,       10,      100,     1000,    10000,   100000],
       [       1,       20,      400,     8000,   160000,  3200000],
       [       1,       30,      900,    27000,   810000, 24300000]])""")

    print("\nTest 3:")
    x3 = np.arange(10,40, 10).reshape(-1,1)/10
    print(f"{x3 = }")
    x3_poly = add_polynomial_features(x3, 3)
    print(f"{x3_poly = }")
    print("""array([[ 1.,  1.,  1.,  1.],
       [ 1.,  2.,  4.,  8.],
       [ 1.,  3.,  9., 27.]])""")
    
