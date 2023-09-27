import json
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import mlflow
import numpy as np
import scipy
from datasets import ClassLabel, load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
)
from transformers.trainer_utils import set_seed

from model import JointBERT


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout for fully-connected layers"},
    )
    ignore_index: Optional[int] = field(
        default=-100,
        metadata={
            "help": "Specifies a target value that is ignored and does not \
            contribute to the input gradient",
        },
    )
    slot_loss_coef: Optional[float] = field(
        default=1.0,
        metadata={"help": "Coefficient for the slot loss."},
    )
    # CRF options
    use_crf: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use CRF"}
    )
    slot_pad_label: Optional[str] = field(
        default="PAD",
        metadata={
            "help": "Pad token for slot label pad (to ignore when calculating loss)",
        },
    )


@dataclass
class DataArguments:
    data_dir: str = field(default="./data", metadata={"help": "The input data dir"})
    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, compute_loss=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            intent_label_ids=inputs["sentiment_label"],
            slot_labels_ids=inputs["labels"],
        )
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if compute_loss else loss


def main(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    logging.set_verbosity_info()
    set_seed(training_args.seed)

    data_files = {"train": "train.jsonl", "test": "test.jsonl", "dev": "dev.jsonl"}
    dataset = load_dataset(f"{data_args.data_dir}", data_files=data_files)
    slot_label_lst = sorted(set(t for ex in dataset["train"]["labels"] for t in ex))
    intent_label_lst = sorted(set(dataset["train"]["sentiment_label"]))
    ner2id = {s: i for i, s in enumerate(slot_label_lst)}
    id2ner = {i: s for i, s in enumerate(slot_label_lst)}
    id2sent = {i: s for i, s in enumerate(intent_label_lst)}
    sent2id = {s: i for i, s in enumerate(intent_label_lst)}

    dataset = dataset.cast_column("sentiment_label", ClassLabel(names=intent_label_lst))
    dataset = dataset.map(lambda y: {"labels": [ner2id[t] for t in y["labels"]]})

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_len,
        max_len_single_sentence=data_args.max_seq_len - 2,
    )

    def preprocess(examples):
        encodings = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": [],
            "sentiment_label": [],
        }
        for text, tags, sentiment_label in zip(
            examples["text"], examples["labels"], examples["sentiment_label"]
        ):
            input_ids = [tokenizer.cls_token_id]
            token_type_ids = [0]
            attention_mask = [1]
            bio_tags = [-100]

            for token, tag in zip(text, tags):
                token_encoding = tokenizer(
                    token,
                    add_special_tokens=False,
                    truncation=True,
                )
                input_ids.extend(token_encoding["input_ids"])
                token_type_ids.extend(token_encoding["token_type_ids"])
                attention_mask.extend(token_encoding["attention_mask"])
                bio_tags.extend([tag] + [-100] * (len(token_encoding["input_ids"]) - 1))
            assert (
                len(input_ids)
                == len(token_type_ids)
                == len(attention_mask)
                == len(bio_tags)
            )
            if len(input_ids) > tokenizer.model_max_length - 1:
                input_ids = input_ids[: tokenizer.model_max_length - 1]
                token_type_ids = token_type_ids[: tokenizer.model_max_length - 1]
                attention_mask = attention_mask[: tokenizer.model_max_length - 1]
                bio_tags = bio_tags[: tokenizer.model_max_length - 1]
            input_ids += [tokenizer.sep_token_id]
            token_type_ids += [0]
            attention_mask += [1]
            bio_tags += [-100]

            encodings["input_ids"].append(input_ids)
            encodings["token_type_ids"].append(token_type_ids)
            encodings["attention_mask"].append(attention_mask)
            encodings["labels"].append(bio_tags)
            encodings["sentiment_label"].append(sentiment_label)
        return encodings

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = JointBERT.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        args=model_args,
        intent_label_lst=intent_label_lst,
        slot_label_lst=slot_label_lst,
    )
    model.config.update(
        {"ner2id": ner2id, "id2ner": id2ner, "id2sent": id2sent, "sent2id": sent2id}
    )

    tokenized_dataset = dataset.map(preprocess, batched=True)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    roc_auc = load_metric("roc_auc", "multiclass")
    seqeval = load_metric("seqeval")

    mlflow.log_param("dropout_rate", model_args.dropout_rate)
    mlflow.log_param("max_seq_len", data_args.max_seq_len)
    mlflow.log_params(training_args.to_dict())

    def compute_metrics(eval_pred):
        results = {}
        (sentiment_predictions, ner_prediction_ids), (
            sentiment_labels,
            ner_label_ids,
        ) = eval_pred
        # evaluate SA
        sentiment_predictions = scipy.special.softmax(sentiment_predictions, axis=1)
        sentiment_acc = roc_auc.compute(
            references=sentiment_labels,
            prediction_scores=sentiment_predictions,
            multi_class="ovr",
        )
        results["sentiment_roc_auc"] = sentiment_acc["roc_auc"]
        # evaluate NER
        ner_prediction_ids = np.argmax(ner_prediction_ids, axis=-1)
        ner_prediction_labels = [[] for _ in range(ner_prediction_ids.shape[0])]
        ner_labels = [[] for _ in range(ner_label_ids.shape[0])]
        for i in range(ner_label_ids.shape[0]):
            for j in range(ner_label_ids.shape[1]):
                if ner_label_ids[i, j] != model_args.ignore_index:
                    ner_prediction_labels[i].append(
                        model.config.id2ner[ner_prediction_ids[i, j]]
                    )
                    ner_labels[i].append(model.config.id2ner[ner_label_ids[i, j]])
        ner_acc = seqeval.compute(
            predictions=ner_prediction_labels, references=ner_labels
        )
        # flatten metrics dictionary
        for ner_type in ner_acc:
            if isinstance(ner_acc[ner_type], dict):
                for metric in ner_acc[ner_type]:
                    results[f"{ner_type}.{metric}"] = ner_acc[ner_type][metric]
            else:
                results[f"{ner_type}"] = ner_acc[ner_type]
        mlflow.log_metrics(results, step=trainer.state.global_step)
        return results

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.output_dir)

        mlflow.pytorch.log_model(
            trainer.model,
            artifact_path="jointBERT",
            registered_model_name="construction-JointBERT",
            code_paths=["./model"],
            pip_requirements=[
                "torch~=1.11.0",
                "transformers~=4.30.2",
                "pytorch-crf==0.7.2",
            ],
        )
        for artifact in glob(training_args.output_dir.strip("/") + "/*"):
            if os.path.isfile(artifact):
                mlflow.log_artifact(artifact, "model/")

    if training_args.do_eval:
        results = trainer.evaluate(tokenized_dataset["test"])

        # log json
        with open("metrics.json", "w") as metrics:
            metrics.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_yaml_file(
        "./params.yaml", allow_extra_keys=True
    )
    run_name = os.getenv("DVC_EXP_NAME", "undefined")
    mlflow.set_tracking_uri("http://localhost:5100")
    mlflow.set_experiment(training_args.output_dir)
    with mlflow.start_run(run_name=run_name):
        main(data_args, model_args, training_args)
