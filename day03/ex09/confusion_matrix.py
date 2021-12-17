import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def df_to_modify(y_source, y_to_change):
    labels = y_source.columns
    for column in labels:
        if column not in y_to_change.columns:
            y_to_change[column] = np.full((y_to_change.shape[0], ), 0)
    y_to_change = y_to_change.reindex(columns = labels)
    return labels, y_to_change

def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    y_dummies = pd.get_dummies(y)
    y_hat_dummies = pd.get_dummies(y_hat)
    if (len(y_dummies.columns) < len(y_hat_dummies.columns)):
        all_labels, y_dummies = df_to_modify(y_hat_dummies, y_dummies)
    else:
        all_labels, y_hat_dummies = df_to_modify(y_dummies, y_hat_dummies)
    if (labels != None):
        y_hat_dummies = y_hat_dummies[labels]
        y_dummies = y_dummies[labels]
    else:
        labels = all_labels
    y = y_dummies.to_numpy()
    y_hat = y_hat_dummies.to_numpy()
    confusion_m = y.T @ y_hat
    if (df_option == True):
        df = pd.DataFrame(confusion_m, columns = labels, index = labels)
        return df
    return (y.T @ y_hat)



if __name__ == "__main__":
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])

    print("# Example 1:")
    print("## your implementation")
    print(f"{confusion_matrix_(y, y_hat) = }")
    print("## sklearn implementation")
    print(f"{confusion_matrix(y, y_hat) = }")
    print()

    print("# Example 2:")
    print("## your implementation")
    print(f"{confusion_matrix_(y, y_hat, labels=['dog', 'norminet']) = }")
    print("## sklearn implementation")
    print(f"{confusion_matrix(y, y_hat, labels=['dog', 'norminet']) = }")
    print()

    print("# Example 3:")
    print(f"{confusion_matrix_(y, y_hat, df_option=True) = }")
    print()

    print("# Example 4:")
    print(f"{confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True) = }")
    print()


    print("CORRECTION:")

    y_true=np.array(['a', 'b', 'c'])
    y_hat=np.array(['a', 'b', 'c'])
    print(f"{confusion_matrix_(y_true, y_hat) = }")
    print("should return a numpy.array or pandas.DataFrame full of zeros except the diagonal which should be full of ones.")
    print()

    y_true=np.array(['a', 'b', 'c'])
    y_hat=np.array(['c', 'a', 'b'])
    print(f"{confusion_matrix_(y_true, y_hat) = }")
    # print(f"{confusion_matrix_(y_hat, y_true) = }")
    print('should return "np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])"')
    print()

    y_true=np.array(['a', 'a', 'a'])
    y_hat=np.array(['a', 'a', 'a'])
    print(f"{confusion_matrix_(y_true, y_hat) = }")
    print("should return np.array([3])")
    print()

    y_true=np.array(['a', 'a', 'a'])
    y_hat=np.array(['a', 'a', 'a'])
    print(f"{confusion_matrix_(y_true, y_hat, labels=[]) = }")
    print("return None, an empty np.array or an empty pandas.Dataframe.")
    print()