from collections import Counter

tg_reserved_chars = [
    "_",
    "*",
    "[",
    "]",
    "(",
    ")",
    "~",
    "`",
    ">",
    "#",
    "+",
    "-",
    "=",
    "|",
    "{",
    "}",
    ".",
    "!",
]


def get_label2id_id2label(class_labels):
    id2label = dict(enumerate(class_labels))
    label2id = {i: label for label, i in id2label.items()}
    return label2id, id2label


def binarize_labels_with_zero_class(labels, num_labels):
    return [int(len(labels) == 0)] + [int(i in labels) for i in range(num_labels)]


def binarize_labels(labels, num_labels):
    return [int(i in labels) for i in range(num_labels)]


def binarize_labels_class_names(labels, class_names):
    return [int(i in labels) for i in class_names]


def sentiment_preprocessor(label):
    if label >= 1:
        return "positive"
    elif label == 0:
        return "neutral"
    elif label <= -1:
        return "negative"


def sentiment_postprocessor(label):
    if label == "neutral":
        return 0
    elif label == "positive":
        return 1
    elif label == "negative":
        return 2


def my_most_common(lst):
    data = Counter(lst)
    most_common = data.most_common()
    maximum_class = most_common[0][0]
    maximum_counter = most_common[0][1]

    for i in range(1, len(most_common)):
        if most_common[i][1] >= maximum_counter:
            return False
    return maximum_class


def save_model(model, tokenizer, directory):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def push_to_hub(model, tokenizer, name):
    model.push_to_hub(name)
    tokenizer.push_to_hub(name)


def fix_text(text):
    for char in tg_reserved_chars:
        text = text.replace(char, f"\\{char}")
    return text
