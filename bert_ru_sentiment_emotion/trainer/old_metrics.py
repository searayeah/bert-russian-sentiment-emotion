from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calculate_aucs_multilabel(y_true, y_pred, num_labels):
    return [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_labels)]


def calculate_f1_scores_multilabel(y_true, y_pred, num_labels):
    return [f1_score(y_true[:, i], y_pred[:, i] >= THRESHOLD) for i in range(num_labels)]


def calculate_recalls_multilabel(y_true, y_pred, num_labels):
    return [
        recall_score(y_true[:, i], y_pred[:, i] >= THRESHOLD) for i in range(num_labels)
    ]


def calculate_precisions_multilabel(y_true, y_pred, num_labels):
    return [
        precision_score(y_true[:, i], y_pred[:, i] >= THRESHOLD)
        for i in range(num_labels)
    ]


def calculate_f1_score_multilabel(y_true, y_pred, average):
    return f1_score(y_true, y_pred >= THRESHOLD, average=average)


def calculate_recall_multilabel(y_true, y_pred, average):
    return recall_score(y_true, y_pred >= THRESHOLD, average=average)


def calculate_precision_multilabel(y_true, y_pred, average):
    return precision_score(y_true, y_pred >= THRESHOLD, average=average)
