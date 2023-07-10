from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from src.utils.utils import binarize_labels


def preprocess(tokenizer, max_length):
    num_labels = 28
    dataset = load_dataset("seara/ru-go-emotions")

    processed_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True,
    ).map(
        lambda x: {
            "label": [float(y) for y in binarize_labels(x["labels"], num_labels)]
        },
        batched=False,
        remove_columns=["text", "labels", "id", "ru_text"],
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
    data_collator = DataCollatorWithPadding(tokenizer)

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
        dataset["validation"],
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
