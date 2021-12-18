import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import numpy as np

def reshape(*args):
    new_args = []
    for arg in args:
        if (len(arg.shape) == 1):
            new_args.append(arg.reshape(-1, 1))
        else:
            new_args.append(arg)
    return new_args

def get_metrics(y, y_hat, pos_label = 1):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for prediction, thruth in zip(y[:, 0], y_hat[:, 0]):
        if prediction == thruth:
            if thruth == pos_label:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if thruth == pos_label:
                false_positive += 1
            else:
                false_negative += 1
    return (true_positive, true_negative, false_positive, false_negative)

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (type(y) != type(y_hat) or type(y) != np.ndarray):
        return None
    y, y_hat = reshape(y, y_hat)
    if type(y[0][0] == str):
        pos_label = y[0][0]
    else:
        pos_label = 1
    tp, tn, fp, fn = get_metrics(y, y_hat, pos_label)
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (type(y) != type(y_hat) or type(y) != np.ndarray or (type(pos_label) != int and type(pos_label) != str)):
        return None
    y, y_hat = reshape(y, y_hat)
    tp, tn, fp, fn = get_metrics(y, y_hat, pos_label)
    if tp == 0:
        precision_score = 0
    else:
        precision_score = tp / (tp + fp)
    return precision_score

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (type(y) != type(y_hat) or type(y) != np.ndarray or (type(pos_label) != int and type(pos_label) != str)):
        return None
    y, y_hat = reshape(y, y_hat)
    tp, tn, fp, fn = get_metrics(y, y_hat, pos_label)
    recall_score = tp / (tp + fn)
    return recall_score

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if (type(y) != type(y_hat) or type(y) != np.ndarray or (type(pos_label) != int and type(pos_label) != str)):
        return None
    y, y_hat = reshape(y, y_hat)
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    if (precision + recall) == 0:
        return 0
    F1_score = (2 * precision * recall) / (precision + recall)
    return F1_score

if __name__ == "__main__":
    print("# Example 1:")
    y_hat = np.array([[1],[ 1],[ 0],[ 1],[ 0],[ 0],[ 1],[ 1]])
    y = np.array([[1],[ 0],[ 0],[ 1],[ 0],[ 1],[ 0],[ 0]])
    
    print("# Accuracy")
    print("## your implementation")
    print(f"{accuracy_score_(y, y_hat) = }")
    print("## sklearn implementation")
    print(f"{accuracy_score(y, y_hat) = }")
    print()
    
    print("# Precision")
    print("## your implementation")
    print(f"{precision_score_(y, y_hat) = }")
    print("## sklearn implementation")
    print(f"{precision_score(y, y_hat) = }")
    print()

    print("# Recall")
    print("## your implementation")
    print(f"{recall_score_(y, y_hat) = }")
    print("## sklearn implementation")
    print(f"{recall_score(y, y_hat) = }")
    print()

    print("# F1-score")
    print("## your implementation")
    print(f"{f1_score_(y, y_hat) = }")
    print("## sklearn implementation")
    print(f"{f1_score(y, y_hat) = }")
    print()
    
    y_hat = np.array([[1],[ 1],[ 0],[ 1],[ 0],[ 0],[ 1],[ 1]])
    y = np.array([[1],[ 0],[ 0],[ 1],[ 0],[ 1],[ 0],[ 0]])
    print(f"{accuracy_score_(y, y_hat) = }")
    print(f"{accuracy_score(y, y_hat) = }")
    print()

    print(f"{precision_score_(y, y_hat) = }")
    print(f"{precision_score(y, y_hat) = }")
    print()

    print(f"{recall_score_(y, y_hat) = }")
    print(f"{recall_score(y, y_hat) = }")
    print()

    print(f"{f1_score_(y, y_hat) = }")
    print(f"{f1_score(y, y_hat) = }")
    print()


    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(f"{accuracy_score_(y, y_hat) = }")
    print(f"{accuracy_score(y, y_hat) = }")
    print()

    print(f"{precision_score_(y, y_hat, pos_label='dog') = }")
    print(f"{precision_score(y, y_hat, pos_label='dog') = }")
    print()

    print(f"{recall_score_(y, y_hat, pos_label='dog') = }")
    print(f"{recall_score(y, y_hat, pos_label='dog') = }")
    print()

    print(f"{f1_score_(y, y_hat, pos_label='dog') = }")
    print(f"{f1_score(y, y_hat, pos_label='dog') = }")
    print()
    