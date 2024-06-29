import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

THRESHOLD = 0.5


def get_report_multilabel(y_true, y_pred, target_names, output_dict):
    return classification_report(
        y_true,
        y_pred >= THRESHOLD,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def calculate_metrics_multilabel(y_true, y_pred, labels):
    len(labels)
    target_names = list(labels.values())

    auc_macro = roc_auc_score(y_true, y_pred, average="macro")
    auc_micro = roc_auc_score(y_true, y_pred, average="micro")
    auc_weighted = roc_auc_score(y_true, y_pred, average="weighted")
    aucs = roc_auc_score(y_true, y_pred, average=None)

    report_dict = get_report_multilabel(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    report_dict["macro avg"]["auc-roc"] = auc_macro
    report_dict["micro avg"]["auc-roc"] = auc_micro
    report_dict["weighted avg"]["auc-roc"] = auc_weighted

    for i, target in enumerate(target_names):
        report_dict[target]["auc-roc"] = aucs[i]

    return report_dict


def get_report_multiclass(y_true, y_pred, target_names, output_dict):
    return classification_report(
        y_true,
        np.argmax(y_pred, axis=-1),
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def calculate_metrics_multiclass(y_true, y_pred, labels):
    len(labels)
    target_names = list(labels.values())

    report_dict = get_report_multiclass(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    auc_macro = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(y_true, y_pred, multi_class="ovr", average="weighted")
    aucs = roc_auc_score(y_true, y_pred, multi_class="ovr", average=None)

    report_dict["macro avg"]["auc-roc"] = auc_macro
    report_dict["weighted avg"]["auc-roc"] = auc_weighted

    for i, target in enumerate(target_names):
        report_dict[target]["auc-roc"] = aucs[i]
    report_dict.pop("accuracy")

    return report_dict


def calculate_metrics(y_true, y_pred, labels, problem_type):
    if problem_type == "single_label_classification":
        return calculate_metrics_multiclass(y_true, y_pred, labels)
    elif problem_type == "multi_label_classification":
        return calculate_metrics_multilabel(y_true, y_pred, labels)
