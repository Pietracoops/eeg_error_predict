import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score, confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, roc_auc_score,
                             cohen_kappa_score)

def get_kappa_score(true_labels, predicted_labels):
    """
    Calculate the Kappa score given true labels and predicted labels.

    Parameters:
    - true_labels: List or array of true labels
    - predicted_labels: List or array of predicted labels

    Returns:
    - kappa_score: The computed Kappa score
    """

    # Kappa score < 0: Indicates agreement worse than random chance.
    # 0 <= Kappa score <= 0.2: Slight agreement.
    # 0.2 < Kappa score <= 0.4: Fair agreement.
    # 0.4 < Kappa score <= 0.6: Moderate agreement.
    # 0.6 < Kappa score <= 0.8: Substantial agreement.
    # 0.8 < Kappa score <= 1: Almost perfect agreement.
    kappa_score = cohen_kappa_score(true_labels, predicted_labels)
    return kappa_score

def get_conf_indices(y_true, y_pred):
    """
    Get indices of false positives (FP) and false negatives (FN).

    Parameters:
    - y_true: true labels (ground truth)
    - y_pred: predicted labels

    Returns:
    - fp_indices: indices of false positives
    - fn_indices: indices of false negatives
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length"

    tp_indices = [i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label == 1 and pred_label == 1]
    tn_indices = [i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label == 0 and pred_label == 0]
    fp_indices = [i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label == 0 and pred_label == 1]
    fn_indices = [i for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label == 1 and pred_label == 0]

    return fp_indices, fn_indices, tp_indices, tn_indices

def get_f1_score(labels, preds, average='macro'):
    """
    Get the f1 score for the given label and prediction set
    """

    f1 = f1_score(labels, preds, average=average)

    return f1


def get_accuracy(labels, preds, raw_correct=True):
    """
    Get the accuracy of the predictions. If raw correct is true, simply return number of correct predictions.
    """

    return accuracy_score(labels, preds, normalize=raw_correct)


def get_precision_recall(labels, preds, average="macro"):
    """
    Get the precision and recall of the model predictions.
    """

    recall = recall_score(labels, preds, average=average)
    precision = precision_score(labels, preds, average=average)

    return (precision, recall)


def get_confusion_matrix_df(labels, preds, label_dict):
    """
    Generates a confusion matrix in dataframe form given the true labels and predictions.
    """
    cm_df = pd.DataFrame(confusion_matrix(labels, preds),
                         index=[i[1] for i in label_dict.items()],
                         columns=[i[1] for i in label_dict.items()])
    return cm_df


def plot_confusion_matrix_display(labels, preds, figsize=(15, 8)):
    """
    Gets the Confusion Matrix Display object from Scikit learn.
    The object takes care of all plotting.
    """

    conf_mat = ConfusionMatrixDisplay.from_predictions(labels, preds)

    return conf_mat


def plot_roc_curve(labels, probs):
    """
    Gets the ROC curve display object from scikit learn, which can be used to plot.
    """
    if isinstance(probs, list):
        roc = RocCurveDisplay.from_predictions(labels, probs)
    else:
        roc = RocCurveDisplay.from_predictions(labels, probs[:, 1])
    return roc


def get_roc_score(labels, probs):
    """
    Get the area under the ROC curve.
    """
    if isinstance(probs, list):
        score = roc_auc_score(labels, probs)
    else:
        score = roc_auc_score(labels, probs[:, 1])
    return score


def get_feature_importance(data, labels, strategy='extreme_random', r_seed=7):
    """
    Get the importances of each feature in making a correct prediction.
    """

    if strategy == 'extreme_random':
        extr = ExtraTreesClassifier(random_state=r_seed)
        extr.fit(data, labels)

        importances = extr.feature_importances_

    imp_pd = pd.DataFrame(importances, index=extr.feature_names_in_)

    return imp_pd