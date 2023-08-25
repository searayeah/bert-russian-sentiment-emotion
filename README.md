# bert-russian-sentiment-emotion

## Introduction

The aim of the project is to fine-tune the state-of-the-art [transformer](https://arxiv.org/abs/1706.03762) models for classifying __emotions__ and __sentiment__ of short input sentences in the Russian language. Since there were multiple BERT models and datasets I decided to structure this DS project in a proper way. So I found multiple popular templates and made something similar. I also discovered a magnifiscent library for managing run configurations - __[Hydra](https://hydra.cc)__. Here is the list of the templates that inspired me:

- [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)
- [pytorch_tempest](https://github.com/Erlemar/pytorch_tempest/)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [pytorch-template](https://github.com/victoresque/pytorch-template)

__[WandB](https://wandb.ai)__ were used for experiment tracking.

## Models and datasets

For the Russian language I found two models, heavy and slow __[ruBERT](https://huggingface.co/DeepPavlov/rubert-base-cased)__, and light and fast __[ruBERT-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)__. The datasets I used for sentiment analysis (multi-class classification) were taken from [Smetanin's review article](https://github.com/sismetanin/sentiment-analysis-in-russian). I chose the most good looking ones (which have at least 3 classes) and unioned them into one __russian-sentiment__ dataset. For the classification of emotions (multi-label classification), I used __[CEDR](https://huggingface.co/datasets/cedr)__ dataset, the only one I found for the Russian language. Therefore I decided to translate English __[GoEmotions](https://huggingface.co/datasets/go_emotions)__ dataset using the [deep-translator](https://github.com/nidhaloff/deep-translator) Python library with Google Translate engine. The translated dataset is called __RuGoEmotions__ and is available on [Hugging Face](https://huggingface.co/datasets/seara/ru_go_emotions) and [Github](https://github.com/searayeah/Ru-GoEmotions).

_Download links for all Russian sentiment datasets collected by Smetanin can be found in this [repository](https://github.com/searayeah/russian-sentiment-emotion-datasets)._

## Run configuration

To start - run

```shell
python main.py <command>=<arg>
```

_If there are no options provided, the default configuration located at the `conf/config.yaml` will be executed._

### Commands and args

```
General:
- task
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

_`python main.py --help` might be also useful._

It is not necessary to provide all the parameters, as the missing ones will be automatically applied according to the defaults. Default parameters for each model-dataset combination are located in the `conf` folder.

#### Examples

This will download fine-tuned rubert-tiny2 model on the default dataset (CEDR) and display popular metrics.

```shell
python main.py task="eval" model="rubert-tiny2"
```

You can explicitly specify the dataset:

```shell
python main.py task="eval" model="rubert-tiny2" dataset="ru-go-emotions"
```

_Evaluation occurs on the test set, which was not used in model's training. The train/val/test split is 80%/10%/10%._

## Metrics and fine-tuned models

The fine-tuned models for each model-dataset combination, which are automatically downloaded when `task="eval"`, and corresponding __metrics__ are located in my [Hugging Face profile](https://huggingface.co/seara).

List of all trained models:

- [rubert-base-cased-russian-sentiment](https://huggingface.co/seara/rubert-base-cased-russian-sentiment)
- [rubert-tiny2-russian-sentiment](https://huggingface.co/seara/rubert-tiny2-russian-sentiment)
- [rubert-base-cased-cedr-russian-emotion](https://huggingface.co/seara/rubert-base-cased-cedr-russian-emotion)
- [rubert-tiny2-cedr-russian-emotion](https://huggingface.co/seara/rubert-tiny2-cedr-russian-emotion)
- [rubert-base-cased-ru-go-emotions](https://huggingface.co/seara/rubert-base-cased-ru-go-emotions)
- [rubert-tiny2-ru-go-emotions](https://huggingface.co/seara/rubert-tiny2-ru-go-emotions)

## Folder structure

```
vkr-bert
├── conf                    - Hydra config files folder
│   ├── dataset             - dataset configs
│   ├── loss                - loss funtion configs
│   ├── model               - model configs
│   ├── optimizer           - optimizer configs
│   └── trainer             - trainer configs for each model-dataset combination
├── data                    - raw data folder
├── main.py                 - main execution file
├── models                  - Location of fine-tuned models
├── notebooks               - Jupyter Notebooks folder
│   ├── datasets            - analysis and visualization
│   └── error-analysis      - model errors analysis
├── requirements.txt        - Python requirements
├── src                     - source code
│   ├── data                - data download and preprocess functions
│   ├── model               - model creation functions
│   ├── trainer             - training, metrics and validation functions
│   └── utils               - some extra functions
└── strings                 - yaml strings for translating classes to the Russian
```
