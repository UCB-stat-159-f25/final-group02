from sklearn.metrics import roc_curve, auc

def compute_accuracy(y_test, y_pred):
    return (y_test == y_pred).mean()

def compute_auc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    return auc(fpr, tpr)