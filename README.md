# Demo-project for ML-model versioning and tracking

An example project for tracking versions of a ML-model, storing and visualising experiment results and datasets.
It trains a [JointBERT](https://github.com/monologg/JointBERT) model on a virtual dataset via a DVC pipeline and tracks all changes via MLFlow.

## Experiment description

Imagine we have a mongoDB with results of parsing a text corpus with two models: for entity extraction and sentiment analysis.
We want to train on these results a joint model, that would predict both at the same time. To do that we want to set up a DVC pipeline that would:

1. Download the latest parse results;
1. Create the markup;
1. Train the model;
1. Upload the results to our MLFlow storage.

We create a `dvc.yaml` file with the following stages: `generate` to create the dataset, `preprocess` to markup the data and `train` to train the model.

## Dataset creation

`gen_mongo_dump.py` reads documents from the provided mongoDB collection, assuming they have the target field with parse results. For every sentence BIO-markup of extracted named entities is generated, and it is attributed with the results of sentiment analysis. Finally the generated dataset is saved as a .jsonl file in ./dump directory.

## Preprocessing

`markup_mongo_dump.py` splits the generated dump in training, dev and test sets, making sure sentiment labels are evenly distributed among all three sets using `scikit-learn` library.

## Model training

`main.py` implements standard `transformers` pipeline for loading datasets, a model and a tokenizer and training the model on the tokenized and vectorized dataset. As in the experiment we want to solve two problems simultaneously, quality of both NER and SA is evaluated. Hyperparameters for training, as well as other launch arguments are taken from `parameters.yaml` file.

The `model` folder with the JointBERT code can be found in the [source repository](https://github.com/monologg/JointBERT).

## Tracking results

An MLFlow storage is used to store training artifacts (model and tokenizer), and track evaluation results.
For simplicity reasons here we assume that the tracking server is launched locally:

```sh
mlflow server -p 5100
```
