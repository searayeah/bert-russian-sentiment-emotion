import pandas as pd

from bert_ru_sentiment_emotion.trainer.metrics import calculate_metrics
from bert_ru_sentiment_emotion.trainer.predict import predict


def eval(model, test_dataloader, labels, problem_type, return_debug=False):
    model.eval()

    test_y_true, test_y_pred, test_loss = predict(model, test_dataloader, problem_type)
    report_dict = calculate_metrics(test_y_true, test_y_pred, labels, problem_type)

    df = pd.DataFrame(report_dict)

    # if problem_type == "multi_label_classification":
    #     f1s, f1 = calculate_f1_score_old(test_y_true, test_y_pred, "micro", len(labels))
    #     df.loc["wrong f1 micro"] = f1s + [None] + [f1] + [None, None]

    #     f1s, f1 = calculate_f1_score_old(test_y_true, test_y_pred, "macro", len(labels))
    #     df.loc["wrong f1 macro"] = f1s + [None] + [f1] + [None, None]

    df = df.round(2)

    print(df)
    df.to_csv("runs/last_run.csv")

    if return_debug:
        return test_y_true, test_y_pred, df
    else:
        return df
