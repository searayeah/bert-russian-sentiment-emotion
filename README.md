# VKR BERT

## Introduction

The goal of my part of the project was to fine-tune the state-of-the-art [transformer](https://arxiv.org/abs/1706.03762) models to classify __emotions__ and __sentiment__ of input sentences in the Russian language. As there were multiple BERT models and datasets I decided to structure this DS project in a proper way. So I found multiple popular templates and made something similar. I also discovered a magnifiscent library for managing run configurations - __[Hydra](https://hydra.cc)__. Here are the list of the templates that inspired me:

- [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)
- [pytorch_tempest](https://github.com/Erlemar/pytorch_tempest/)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [pytorch-template](https://github.com/victoresque/pytorch-template)

__[WandB](https://wandb.ai)__ were used for experiment tracking and the live inference of models were provided by Telegram bot written using the __[python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)__ library.

## Models and datasets

For the Russian language I found two models, heavy and slow [ruBERT](https://huggingface.co/DeepPavlov/rubert-base-cased), and light and fast [ruBERT-tiny2](https://huggingface.co/cointegrated/rubert-tiny2). The datasets I used for sentiment analysis were taken from [Smetanin review article](https://github.com/sismetanin/sentiment-analysis-in-russian). I chose the most good looking ones (which have at least 3 classes) and unioned them into one *russian-sentiment* dataset. For the emotions dataset I used [CEDR](https://huggingface.co/datasets/cedr), which was the only one I found for the Russian language. Therefore I decided to translate English [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset using the [deep-translator](https://github.com/nidhaloff/deep-translator) Python library with Google Translate engine. The translated dataset is called [RuGoEmotions](https://huggingface.co/datasets/seara/ru-go-emotions) and is available on Hugging Face. 

## Run configuration

To start - run

```shell
python main.py <command>=<arg>
```

>If there are no options provided, the default configuration located at the `conf/config.yaml` will be executed.

#### Commands and args
```
General:
- task
  - "telegram" for deploying Telegram bot
  - "train" to train model
  - "eval" to evaluate trained model

- log_wandb
  - "True" to enable WandB
  - "False" to disable WandB

- model
  - "rubert-base-cased" for ruBERT
  - "rubert-tiny2" for ruBERT-tiny2

- dataset
  - "cedr"
  - "ru-go-emotions"
  - "russian-sentiment"

Tokenizer:
- trainer.max_length - Selecting tokenizer truncation max length

Dataloader:
- trainer.batch_size
- trainer.shuffle
- trainer.num_workers
- trainer.pin_memory
- trainer.drop_last

Optimizer
- trainer.lr
- trainer.weight_decay
- trainer.num_epochs
```

It is not necessary to provide all the parameters, the missing ones will be automatically applied according to the defaults. Default parameters for each model-dataset combination are located in the `conf` folder.

## Fine-tuned models

The fine-tuned modes for each model-dataset combination, which are automatically downloaded when `task="eval"` are located in my [Hugging Face profile](https://huggingface.co/seara).






