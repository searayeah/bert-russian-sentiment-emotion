import torch
from tqdm.auto import tqdm


def predict(model, dataloader, problem_type):
    with torch.inference_mode():
        y_true = []
        y_pred = []
        val_loss = 0

        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            output = model(**batch)
            loss = output.loss

            val_loss += loss.item()
            y_true.append(batch.labels.cpu())
            y_pred.append(output.logits.cpu())

        val_loss = val_loss / len(dataloader)

    if problem_type == "single_label_classification":
        return (
            torch.cat(y_true).numpy(),
            torch.softmax(torch.cat(y_pred), dim=-1).numpy(),
            val_loss,
        )
    elif problem_type == "multi_label_classification":
        return (
            torch.cat(y_true).numpy(),
            torch.sigmoid(torch.cat(y_pred)).numpy(),
            val_loss,
        )


def train_epoch(model, train_dataloader, optimizer, problem_type):
    y_true = []
    y_pred = []
    train_loss = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        batch = batch.to(model.device)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_true.append(batch.labels.detach().cpu())
        y_pred.append(output.logits.detach().cpu())

    train_loss = train_loss / len(train_dataloader)

    if problem_type == "single_label_classification":
        return (
            torch.cat(y_true).numpy(),
            torch.softmax(torch.cat(y_pred), dim=-1).numpy(),
            train_loss,
        )
    elif problem_type == "multi_label_classification":
        return (
            torch.cat(y_true).numpy(),
            torch.sigmoid(torch.cat(y_pred)).numpy(),
            train_loss,
        )
