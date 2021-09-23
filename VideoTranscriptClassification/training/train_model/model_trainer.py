from functools import partial

import torch
from shared.classes import EncodedDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

PROBLEM_TYPE_REGRESSION = "regression"
PROBLEM_TYPE_SINGLE_LABEL_CLASSIFICATION = "single_label_classification"
PROBLEM_TYPE_MULTI_LABEL_CLASSIFICATION = "multi_label_classification"

MODEL_NAME = "distilbert-base-uncased"


class WeightedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    """Weighted DistilBERT For Sequence Classification."""

    def __init__(self, *args, **kwargs):
        """Instantiate an instance."""
        self.pos_weight = None
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        TODO: This needs to be rewritten.

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1`
            a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1`
            a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = PROBLEM_TYPE_REGRESSION
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = PROBLEM_TYPE_SINGLE_LABEL_CLASSIFICATION
                else:
                    self.config.problem_type = PROBLEM_TYPE_MULTI_LABEL_CLASSIFICATION
                print(self.config.problem_type)

            if self.config.problem_type == PROBLEM_TYPE_REGRESSION:
                loss_fct = MSELoss()
                loss = (
                    loss_fct(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else loss_fct(logits, labels)
                )
            elif self.config.problem_type == PROBLEM_TYPE_SINGLE_LABEL_CLASSIFICATION:
                loss_fct = CrossEntropyLoss(
                    weight=torch.as_tensor(self.pos_weight, device=self.distilbert.device)
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == PROBLEM_TYPE_MULTI_LABEL_CLASSIFICATION:
                loss_fct = BCEWithLogitsLoss(
                    pos_weight=torch.as_tensor(
                        self.pos_weight, device=self.distilbert.device
                    )
                )
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


# def get_tokenizer_and_model(  # noqa D103  # TODO: Remove this ignore
#     num_classes, id2label_dict, label2id_dict
# ):
#     tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
#     model = DistilBertForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=num_classes,
#         id2label=id2label_dict,
#         label2id=label2id_dict,
#     )
#     return tokenizer, model
def get_tokenizer_and_model(num_classes, pos_weight, id2label_dict, label2id_dict):
    """Get Tokenizer and Model.

    Args:
        num_classes (int): Number of classes (labels).
        pos_weight (dict): position to weight mappings.
        id2label_dict (dict): identifier to label mappings.
        label2id_dict (dict): label to identifier mappings

    Returns:
        tuple: tokenizer and model.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = WeightedDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=id2label_dict,
        label2id=label2id_dict,
    )
    model.pos_weight = pos_weight
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

    return return_metrics


def get_weights(df):
    weights = [1.0] * len(df.target.unique())
    total_examples = len(df)
    for label, count in df.target.value_counts().items():
        weights[int(label)] = float(round(1 - (count/total_examples), 2))
    return weights


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
            pos_weight=get_weights(df),
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
