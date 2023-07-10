from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)
import numpy as np

THRESHOLD = 0.5

# correct
def calculate_aucs_multilabel(y_true, y_pred, num_labels):
    return [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_labels)]


# correct
def calculate_f1_scores_multilabel(y_true, y_pred, num_labels):
    return [
        f1_score(y_true[:, i], y_pred[:, i] >= THRESHOLD) for i in range(num_labels)
    ]


# correct
def calculate_recalls_multilabel(y_true, y_pred, num_labels):
    return [
        recall_score(y_true[:, i], y_pred[:, i] >= THRESHOLD) for i in range(num_labels)
    ]


# correct
def calculate_precisions_multilabel(y_true, y_pred, num_labels):
    return [
        precision_score(y_true[:, i], y_pred[:, i] >= THRESHOLD)
        for i in range(num_labels)
    ]


# correct
def calculate_f1_score_multilabel(y_true, y_pred, average):
    return f1_score(y_true, y_pred >= THRESHOLD, average=average)


# correct
def calculate_recall_multilabel(y_true, y_pred, average):
    return recall_score(y_true, y_pred >= THRESHOLD, average=average)


# correct
def calculate_precision_multilabel(y_true, y_pred, average):
    return precision_score(y_true, y_pred >= THRESHOLD, average=average)


# correct
def get_report_multilabel(y_true, y_pred, target_names, output_dict):
    return classification_report(
        y_true,
        y_pred >= THRESHOLD,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def calculate_f1_score_old(y_true, y_pred, average, num_labels):
    f1s = [
        f1_score(y_true[:, i], y_pred[:, i] > 0.5, average=average)
        for i in range(num_labels)
    ]

    return f1s, sum(f1s) / len(f1s)


def calculate_metrics_multilabel(y_true, y_pred, labels):
    num_labels = len(labels)
    target_names = list(labels.values())

    aucs = dict(
        zip(target_names, calculate_aucs_multilabel(y_true, y_pred, num_labels))
    )
    auc_macro = roc_auc_score(y_true, y_pred)

    report_dict = get_report_multilabel(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    for key in report_dict.keys():
        if key in target_names:
            report_dict[key]["auc-roc"] = aucs[key]
        elif key == "macro avg":
            report_dict[key]["auc-roc"] = auc_macro
        else:
            report_dict[key]["auc-roc"] = None

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
    num_labels = len(labels)
    target_names = list(labels.values())

    auc_macro = roc_auc_score(y_true, y_pred, multi_class="ovr")

    report_dict = get_report_multiclass(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    report_dict["auc-roc"] = auc_macro

    return report_dict


def calculate_metrics(y_true, y_pred, labels, problem_type):
    if problem_type == "single_label_classification":
        return calculate_metrics_multiclass(y_true, y_pred, labels)
    elif problem_type == "multi_label_classification":
        return calculate_metrics_multilabel(y_true, y_pred, labels)
