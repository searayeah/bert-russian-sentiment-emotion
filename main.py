import wandb
from src.trainer.train import train
from src.trainer.eval import eval
from src.model.models import get_model
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from src.utils.utils import save_model, push_to_hub
from src.utils.telegram_deploy import main as launch_telegram

# turn off bert warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# printing full errors
os.environ["HYDRA_FULL_ERROR"] = "1"

# login to services
# wandb.login()
# notebook_login()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.task == "train":

        tokenizer, model = get_model(
            cfg.model.encoder,
            cfg.dataset.labels,
            cfg.dataset.num_labels,
            cfg.trainer.problem_type,
            cfg.task,
        )

        train_dataloader, val_dataloader, test_dataloader = call(
            cfg.dataset.dataloader, tokenizer=tokenizer
        )

        if cfg.log_wandb:
            wandb.init(
                project=f"{cfg.project_name}-{cfg.model.name}-{cfg.dataset.name}",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        optimizer = instantiate(cfg.optimizer, params=model.parameters())

        model.cuda()

        train(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            epochs=cfg.trainer.num_epochs,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            labels=cfg.dataset.labels,
            problem_type=cfg.trainer.problem_type,
            log_wandb=cfg.log_wandb,
        )

        if cfg.log_wandb:
            wandb.finish()

        ask = input("Upload to hub?: ")
        if ask == "y" or ask == "yes":

            save_model(
                model,
                tokenizer,
                f"models/{cfg.model.name}-{cfg.dataset.name}-ep={cfg.trainer.num_epochs}-lr={cfg.trainer.lr}",
            )

            push_to_hub(model, tokenizer, f"{cfg.model.name}-{cfg.dataset.name}")

    elif cfg.task == "eval":

        tokenizer, model = get_model(
            f"seara/{cfg.model.name}-{cfg.dataset.name}",
            cfg.dataset.labels,
            cfg.dataset.num_labels,
            cfg.trainer.problem_type,
            cfg.task,
        )

        train_dataloader, val_dataloader, test_dataloader = call(
            cfg.dataset.dataloader, tokenizer=tokenizer
        )

        eval(
            model=model,
            test_dataloader=test_dataloader,
            labels=cfg.dataset.labels,
            problem_type=cfg.trainer.problem_type,
        )
    elif cfg.task == "telegram":
        launch_telegram()



if __name__ == "__main__":
    main()
