import wandb
from src.trainer.metrics import calculate_metrics
from src.trainer.predict import predict, train_epoch
from src.trainer.eval import eval
from tqdm.auto import tqdm
import pandas as pd


def train(
    model,
    train_dataloader,
    optimizer,
    epochs,
    val_dataloader,
    test_dataloader,
    labels,
    problem_type,
    log_wandb,
):
    tq = tqdm(range(epochs))

    for epoch in tq:
        model.train()
        train_y_true, train_y_pred, train_loss = train_epoch(
            model, train_dataloader, optimizer, problem_type
        )

        model.eval()
        val_y_true, val_y_pred, val_loss = predict(model, val_dataloader, problem_type)

        report_dict = calculate_metrics(val_y_true, val_y_pred, labels, problem_type)

        if log_wandb:
            wandb_dict = report_dict.copy()
            for label in labels.values():
                wandb_dict.pop(label)
            wandb_dict.pop("samples avg", None)
            wandb_dict.pop("weighted avg", None)
            for key in wandb_dict:
                if isinstance(wandb_dict[key], dict):
                    wandb_dict[key].pop("support", None)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, **wandb_dict})

        tq.set_description(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    df = eval(model, test_dataloader, labels, problem_type)

    if log_wandb:
        table = wandb.Table(dataframe=df.rename_axis("metric").reset_index())
        wandb.log({"table": table})
