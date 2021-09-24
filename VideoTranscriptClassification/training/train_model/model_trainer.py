from functools import partial

import torch
from shared.classes import EncodedDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def get_tokenizer_and_model(  # noqa D103  # TODO: Remove this ignore
    num_classes, id2label_dict, label2id_dict
):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
        id2label=id2label_dict,
        label2id=label2id_dict,
    )
    return tokenizer, model


def compute_metrics(pred):
    """Compute metric.

    Args:
        pred (np.ndarray): Predictions of the model.

    Returns:
        dict: metric
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return_metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

class ModelTrainer:  # noqa D103  # TODO: Remove this ignore
    def __init__(  # noqa D107  # TODO: Remove this ignore
        self,
        df,
        class_lookup,
        num_epochs=15,  # total number of training epochs
        output_dir="./outputs",  # output directory
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        ddp_find_unused_parameters=False,
    ):
        self.train_labels = df[df.is_valid == False].target.values.tolist()  # noqa E712

        self.tokenizer, self.model = get_tokenizer_and_model(
            num_classes=len(set(self.train_labels)),
            id2label_dict={v: k for k, v in class_lookup.items()},
            label2id_dict=class_lookup,
        )
        self.train_dataset = EncodedDataset(
            df[df.is_valid == False], self.tokenizer  # noqa E712
        )
        self.val_dataset = EncodedDataset(
            df[df.is_valid == True], self.tokenizer  # noqa E712
        )

        self.training_args = TrainingArguments(
            output_dir=output_dir + "/logging",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            fp16=fp16,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
        )
        self.trainer = None

    def run_trainer(self):  # noqa D103  # TODO: Remove this ignore

        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=4, early_stopping_threshold=0.01
                )
            ],
            tokenizer=self.tokenizer,
        )

        print(f"is world process 0: {self.trainer.is_world_process_zero()}")
        self.trainer.train()

        return self.model, self.trainer, self.tokenizer
