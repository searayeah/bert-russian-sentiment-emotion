from transformers import AutoTokenizer, BertForSequenceClassification

from bert_ru_sentiment_emotion.utils.utils import get_label2id_id2label


def get_model(model, labels, num_labels, problem_type, task):

    tokenizer = AutoTokenizer.from_pretrained(model)

    if task == "train":
        label2id, id2label = get_label2id_id2label(labels.values())
        model = BertForSequenceClassification.from_pretrained(
            model,
            num_labels=num_labels,
            problem_type=problem_type,
            label2id=label2id,
            id2label=id2label,
        )
        print("Loaded pretrained encoder")

    elif task == "eval":
        model = BertForSequenceClassification.from_pretrained(model)
        print("Loaded pretrained model")

    return tokenizer, model
