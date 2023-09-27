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

## Markup

## Model training

## Tracking results
