import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from src.utils.utils import (
    sentiment_postprocessor,
    sentiment_preprocessor,
    my_most_common,
)
from pandarallel import pandarallel

RANDOM_STATE = 42
PANDARALLEL_WORKERS = 8


def get_linis_crowd():
    linis_crowd_2015_path = "data/linis-crowd-2015/text_rating_final.xlsx"
    linis_crowd_2016_path = "data/linis-crowd-2016/doc_comment_summary.xlsx"
    column_names = ["text", "label"]

    linis_crowd_2016 = pd.read_excel(
        linis_crowd_2016_path, header=None, names=column_names
    ).dropna()
    linis_crowd_2016 = linis_crowd_2016[
        linis_crowd_2016["label"].isin([-2, -1, 0, 1, 2])
    ].copy()

    linis_crowd_2016["label"] = linis_crowd_2016["label"].apply(sentiment_preprocessor)
    linis_crowd_2016["text"] = linis_crowd_2016["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )

    linis_crowd_2015 = pd.read_excel(
        linis_crowd_2015_path, header=None, names=column_names
    ).dropna()
    linis_crowd_2015 = linis_crowd_2015[
        linis_crowd_2015["label"].isin([-2, -1, 0, 1, 2])
    ].copy()

    linis_crowd_2015["label"] = linis_crowd_2015["label"].apply(sentiment_preprocessor)
    linis_crowd_2015["text"] = linis_crowd_2015["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )

    linis_crowd = pd.concat([linis_crowd_2015, linis_crowd_2016], ignore_index=True)

    pandarallel.initialize(progress_bar=True, nb_workers=PANDARALLEL_WORKERS)

    linis_crowd["occur"] = linis_crowd["text"].parallel_apply(
        lambda x: (linis_crowd["text"] == x).sum()
    )

    processed_linis_crowd = linis_crowd[linis_crowd["occur"] >= 3]

    unique_mapping = {}

    for index, row in processed_linis_crowd.iterrows():
        if row["text"] in unique_mapping:
            unique_mapping[row["text"]].append(row["label"])
        else:
            unique_mapping[row["text"]] = [row["label"]]

    final_df_map = {}

    for key in unique_mapping:
        most_common = my_most_common(unique_mapping[key])
        if most_common is not False:
            final_df_map[key] = most_common

    final_linis_crowd = pd.DataFrame(final_df_map.items(), columns=column_names)

    size = final_linis_crowd.shape[0]
    val_test_size = int(size * 0.1)

    linis_crowd_train_val, linis_crowd_test = train_test_split(
        final_linis_crowd,
        test_size=val_test_size,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    linis_crowd_train, linis_crowd_val = train_test_split(
        linis_crowd_train_val,
        test_size=val_test_size,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"Loaded Linis: train {linis_crowd_train.shape[0]},",
        f"val {linis_crowd_val.shape[0]},",
        f"test {linis_crowd_test.shape[0]}",
    )
    return linis_crowd_train, linis_crowd_val, linis_crowd_test


def get_rureviews():
    ru_reviews_path = "data/ru-reviews/women-clothing-accessories.3-class.balanced.csv"
    column_names = ["text", "label"]

    ru_reviews = pd.read_csv(ru_reviews_path, sep="\t", names=column_names, skiprows=1)

    ru_reviews["label"] = ru_reviews["label"].apply(
        lambda x: "neutral" if x == "neautral" else x
    )
    ru_reviews["text"] = ru_reviews["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )

    size = ru_reviews.shape[0]
    val_test_size = int(size * 0.1)

    ru_reviews_train_val, ru_reviews_test = train_test_split(
        ru_reviews, test_size=val_test_size, shuffle=True, random_state=RANDOM_STATE
    )

    ru_reviews_train, ru_reviews_val = train_test_split(
        ru_reviews_train_val,
        test_size=val_test_size,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"Loaded RuReviews: train {ru_reviews_train.shape[0]},",
        f"val {ru_reviews_val.shape[0]},",
        f"test {ru_reviews_test.shape[0]}",
    )

    return ru_reviews_train, ru_reviews_val, ru_reviews_test


def get_rusentiment():
    rusent_random_path = "data/ru-sentiment/rusentiment_random_posts.csv"
    rusent_active_path = "data/ru-sentiment/rusentiment_preselected_posts.csv"
    rusent_test_path = "data/ru-sentiment/rusentiment_test.csv"
    labels = ["positive", "negative", "neutral"]

    rusent_random = pd.read_csv(rusent_random_path)
    rusent_active = pd.read_csv(rusent_active_path)
    rusent_test = pd.read_csv(rusent_test_path)

    # remove duplicate indexes, ignore index is vital
    rusent_train_val = pd.concat([rusent_active, rusent_random], ignore_index=True)

    rusent_train_val = rusent_train_val[rusent_train_val["label"].isin(labels)]
    rusent_test = rusent_test[rusent_test["label"].isin(labels)]

    rusent_train_val["text"] = rusent_train_val["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )
    rusent_test["text"] = rusent_test["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )

    size = rusent_train_val.shape[0] + rusent_test.shape[0]
    val_size = int(size * 0.09)

    rusent_train, rusent_val = train_test_split(
        rusent_train_val, test_size=val_size, shuffle=True, random_state=RANDOM_STATE
    )

    print(
        f"Loaded RuSentiment: train {rusent_train.shape[0]},",
        f"val {rusent_val.shape[0]},",
        f"test {rusent_test.shape[0]}",
    )

    return rusent_train, rusent_val, rusent_test


def get_kaggle_russian_news():
    kaggle_path = "data/kaggle-ru-news/train.json"
    column_names = ["text", "label"]

    kaggle = pd.read_json(kaggle_path).drop("id", axis=1)
    kaggle.columns = column_names

    kaggle["text"] = kaggle["text"].apply(
        lambda x: x.replace("\n", " ").replace("\t", "")
    )

    size = kaggle.shape[0]
    val_test_size = int(size * 0.1)

    kaggle_train_val, kaggle_test = train_test_split(
        kaggle, test_size=val_test_size, shuffle=True, random_state=RANDOM_STATE
    )

    kaggle_train, kaggle_val = train_test_split(
        kaggle_train_val,
        test_size=val_test_size,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"Loaded Kaggle: train {kaggle_train.shape[0]},",
        f"val {kaggle_val.shape[0]},",
        f"test {kaggle_test.shape[0]}",
    )

    return kaggle_train, kaggle_val, kaggle_test


def get_russian_sentiment_all():
    (
        linis_crowd_2016_train,
        linis_crowd_2016_val,
        linis_crowd_2016_test,
    ) = get_linis_crowd()
    ru_reviews_train, ru_reviews_val, ru_reviews_test = get_rureviews()
    rusent_train, rusent_val, rusent_test = get_rusentiment()
    kaggle_train, kaggle_val, kaggle_test = get_kaggle_russian_news()

    russian_sentiment_train = pd.concat(
        [linis_crowd_2016_train, ru_reviews_train, rusent_train, kaggle_train],
        ignore_index=True,
    )
    russian_sentiment_val = pd.concat(
        [linis_crowd_2016_val, ru_reviews_val, rusent_val, kaggle_val],
        ignore_index=True,
    )
    russian_sentiment_test = pd.concat(
        [linis_crowd_2016_test, ru_reviews_test, rusent_test, kaggle_test],
        ignore_index=True,
    )

    print(
        f"Loaded all datasets: train {russian_sentiment_train.shape[0]},",
        f"val {russian_sentiment_val.shape[0]},",
        f"test {russian_sentiment_test.shape[0]}",
    )
    return russian_sentiment_train, russian_sentiment_val, russian_sentiment_test


def preprocess(tokenizer, max_length):
    train, val, test = get_russian_sentiment_all()

    dataset_train = Dataset.from_pandas(train, preserve_index=False)
    dataset_val = Dataset.from_pandas(val, preserve_index=False)
    dataset_test = Dataset.from_pandas(test, preserve_index=False)

    dataset_dict = DatasetDict(
        {"train": dataset_train, "val": dataset_val, "test": dataset_test}
    )

    processed_dataset = dataset_dict.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True,
    ).map(
        lambda x: {"labels": sentiment_postprocessor(x["label"])},
        batched=False,
        remove_columns=["text", "label"],
    )

    return processed_dataset


def get_dataloaders(
    tokenizer,
    max_length,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    drop_last,
):

    dataset = preprocess(tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_dataloader = DataLoader(
        dataset["val"],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(
        f"Loaded dataloaders: train {len(train_dataloader)},",
        f"val {len(val_dataloader)},",
        f"test {len(test_dataloader)}",
    )

    return train_dataloader, val_dataloader, test_dataloader
