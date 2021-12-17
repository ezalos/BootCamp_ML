import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in
    ,→ argument.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to
    ,→ be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he
    ,→ polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    new = []
    new.append(x)
    for i in range(power - 1):
        new.append(x ** (i + 2))
    return np.concatenate(tuple(new),axis=1)

if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)
    print("# Example 1:")
    res = add_polynomial_features(x, 3)
    print(res)
    print("""
    # Output:
    array([[ 1, 1, 1],
    [ 2, 4, 8],
    [ 3, 9, 27],
    [ 4, 16, 64],
    [ 5, 25, 125]])
    """)
    print("# Example 2:")
    res = add_polynomial_features(x, 6)
    print(res)
    print("""
    # Output:
    array([[ 1, 1, 1, 1, 1, 1],
    [ 2, 4, 8, 16, 32, 64],
    [ 3, 9, 27, 81, 243, 729],
    [ 4, 16, 64, 256, 1024, 4096],
    [ 5, 25, 125, 625, 3125, 15625]])
    """)

    x = np.arange(1,11).reshape(5, 2)
    print("# Example 1:")
    print(f"{add_polynomial_features(x, 3) = }")
    print("# Output:")
    print("""array([[ 1, 2, 1, 4, 1, 8],
    [ 3, 4, 9, 16, 27, 64],
    [ 5, 6, 25, 36, 125, 216],
    [ 7, 8, 49, 64, 343, 512],
    [ 9, 10, 81, 100, 729, 1000]])""")
    
    print("# Example 2:")
    print(f"{add_polynomial_features(x, 5) = }")
    print("# Output:")
    print("""array([[     1,      2,      1,      4,      1,      8,      1,     16, 1,     32],
       [     3,      4,      9,     16,     27,     64,     81,    256, 243,   1024],
       [     5,      6,     25,     36,    125,    216,    625,   1296, 3125,   7776],
       [     7,      8,     49,     64,    343,    512,   2401,   4096,  16807,  32768],
       [     9,     10,     81,    100,    729,   1000,   6561,  10000, 59049, 100000]])
    """)


    print("CORRECTION:")
    x1 = np.ones(10).reshape(5, 2)
    print(f"{add_polynomial_features(x1, 3) = }")
    print("""[[   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]
        [   1    1    1    1    1    1]]""")

    x = np.arange(1,6, 1).reshape(-1,1)
    X = np.hstack((x, -x))
    print(f"{add_polynomial_features(X, 3) = }")
    print("""array([[   1,   -1,    1,    1,    1,   -1],
            [   2,   -2,    4,    4,    8,   -8],
            [   3,   -3,    9,    9,   27,  -27],
            [   4,   -4,   16,   16,   64,  -64],
            [   5,   -5,   25,   25,  125, -125]])""")