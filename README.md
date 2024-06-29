# bert-russian-sentiment-emotion

## Introduction

The aim of the project is to fine-tune the state-of-the-art [transformer](https://arxiv.org/abs/1706.03762) models for classifying **emotions** and **sentiment** of short input sentences in the Russian language. Since there were multiple BERT models and datasets I decided to use a magnifiscent library for managing run configurations - **[Hydra](https://hydra.cc)**. **[WandB](https://wandb.ai)** was used for experiment tracking.

## Metrics and fine-tuned models

The fine-tuned models for each model-dataset combination, which are automatically downloaded when `task="eval"`, and corresponding **metrics** are located in my [Hugging Face profile](https://huggingface.co/seara).

Example usage:

```python
from transformers import pipeline
model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
model("Привет, ты мне нравишься!")
# [{'label': 'love', 'score': 0.5955629944801331}]
```

**List of all trained models:**

- Sentiment
  - [rubert-base-cased-russian-sentiment](https://huggingface.co/seara/rubert-base-cased-russian-sentiment)
  - [rubert-tiny2-russian-sentiment](https://huggingface.co/seara/rubert-tiny2-russian-sentiment)
- Emotion
  - [rubert-base-cased-cedr-russian-emotion](https://huggingface.co/seara/rubert-base-cased-cedr-russian-emotion)
  - [rubert-tiny2-cedr-russian-emotion](https://huggingface.co/seara/rubert-tiny2-cedr-russian-emotion)
  - [rubert-base-cased-ru-go-emotions](https://huggingface.co/seara/rubert-base-cased-ru-go-emotions)
  - [rubert-tiny2-ru-go-emotions](https://huggingface.co/seara/rubert-tiny2-ru-go-emotions)

## Models and datasets description

For the Russian language I found two models, heavy and slow **[ruBERT](https://huggingface.co/DeepPavlov/rubert-base-cased)**, and light and fast **[ruBERT-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)**. The datasets I used for sentiment analysis (multi-class classification) were taken from [Smetanin's review article](https://github.com/sismetanin/sentiment-analysis-in-russian). I chose the most good looking ones (which have at least 3 classes) and unioned them into one **russian-sentiment** dataset. For the classification of emotions (multi-label classification), I used **[CEDR](https://huggingface.co/datasets/cedr)** dataset, the only one I found for the Russian language. Therefore I decided to translate English **[GoEmotions](https://huggingface.co/datasets/go_emotions)** dataset using the [deep-translator](https://github.com/nidhaloff/deep-translator) Python library with Google Translate engine. The translated dataset is called **RuGoEmotions** and is available on [Hugging Face](https://huggingface.co/datasets/seara/ru_go_emotions) and [Github](https://github.com/searayeah/ru-goemotions).

> Download links for all Russian sentiment datasets collected by Smetanin can be found in this [repository](https://github.com/searayeah/russian-sentiment-emotion-datasets).

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

## Project structure

```
bert-russian-sentiment-emotion
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

Here is the list of the templates that inspired me:

- [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)
- [pytorch_tempest](https://github.com/Erlemar/pytorch_tempest/)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [pytorch-template](https://github.com/victoresque/pytorch-template)
