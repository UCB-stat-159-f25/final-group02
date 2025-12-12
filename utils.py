from sklearn.metrics import roc_curve, auc

def compute_accuracy(y_test, y_pred):
    """
    Compute the classification accuracy

    Parameters
    ----------
    y_test: 
        array, True binary labels
    y_pred: 
        array, Predicted class labels

    Returns
    -------
    float
        Accuracy score
    """
    return (y_test == y_pred).mean()

def compute_auc(y_test, y_prob):
    """
    Compute the Area Under the ROC Curve

    Parameters
    ----------
    y_test: array
        True binary labels
    y_prob: array-like
        Predicted probabilities for the positive class

    Returns
    -------
    float
        The ROC AUC score, the model's ability to distinguish between positive and negative class
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    return auc(fpr, tpr)