import itertools
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.classes import EncodedDataset


def create_preds_df(  # noqa D103  # TODO: Remove this ignore
    model, dl, device, label_encoder
):
    iids = []
    embeddings = []
    pred_classes = []
    pred_probs = []
    with torch.no_grad():
        for batch in tqdm(dl):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                output_attentions=True,
                output_hidden_states=True,
            )
            embeddings.append(
                output.hidden_states[-1][
                    :,
                    0,
                ].tolist()
            )
            topk = F.softmax(output.logits, dim=-1).topk(1, dim=1)
            pred_classes.append(topk.indices.T.squeeze(0).tolist())
            pred_probs.append(topk.values.T.squeeze(0).tolist())
            iids.append(batch["iid"])

    preds_df = pd.DataFrame(
        {
            "video_id": chain.from_iterable(iids),
            "predicted_label_idx": chain.from_iterable(pred_classes),
            "prediction_probability": chain.from_iterable(pred_probs),
        }
    )
    preds_df["predicted_label"] = label_encoder.inverse_transform(
        preds_df.predicted_label_idx
    )

    return preds_df, embeddings


def create_results_df(  # noqa D103  # TODO: Remove this ignore
    model, tokenizer, df, label_encoder, batch_size=32
):
    eval_dataset = EncodedDataset(df, return_iid=True, tokenizer=tokenizer)
    device = model.device

    if device == torch.device("cpu"):
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        device = (
            torch.device("cuda:0") if len(available_gpus) > 0 else torch.device("cpu")
        )
        model.to(device)

    eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)

    preds_df, embeddings = create_preds_df(
        model=model, dl=eval_loader, device=device, label_encoder=label_encoder
    )

    results_df = pd.merge(df.reset_index(), preds_df, on="video_id").set_index(
        "video_id"
    )[
        [
            "transcript",
            "target_label",
            "target",
            "predicted_label",
            "predicted_label_idx",
            "prediction_probability",
            "is_valid",
        ]
    ]

    return results_df, embeddings


def create_most_confused_df(  # noqa D103  # TODO: Remove this ignore
    results_df, embeddings
):
    doc_encodings_matrix = torch.tensor(list(chain.from_iterable(embeddings)))
    video_ids = results_df.reset_index().video_id.tolist()
    video_id_to_idx = {v: i for i, v in enumerate(video_ids)}

    most_confused = (
        results_df[results_df.target != results_df.predicted_label_idx]
        .sort_values("prediction_probability", ascending=False)
        .head(100)
    )
    most_confused["most_similar_videos"] = most_confused.apply(
        lambda row: get_most_similar_videos(
            row.name, video_ids, video_id_to_idx, doc_encodings_matrix, limit=5
        ),
        axis=1,
    )
    most_confused["most_similar_videos_labels"] = most_confused.apply(
        lambda row: results_df.loc[row["most_similar_videos"]].target_label.tolist()[1:],
        axis=1,
    )
    return most_confused


def get_most_similar_videos(  # noqa D103  # TODO: Remove this ignore
    target_video_id, video_ids, video_id_to_idx_lookup, doc_encodings_matrix, limit=10
):
    target_idx = video_id_to_idx_lookup[target_video_id]
    sim_scores = (
        torch.nn.functional.cosine_similarity(
            doc_encodings_matrix[target_idx][None], doc_encodings_matrix
        )
        .cpu()
        .tolist()
    )
    most_similar = sorted(
        zip(video_ids, sim_scores), reverse=True, key=lambda tup: tup[1]
    )[:limit]
    most_similar_ids, _ = zip(*most_similar)
    return list(most_similar_ids)


def calculate_classification_metrics(results_df):  # noqa D103  # TODO: Remove this ignore
    le = preprocessing.LabelEncoder()
    encoded_targets = le.fit_transform(results_df.target_label)
    encoded_preds = le.transform(results_df.predicted_label)
    metrics = classification_report(
        encoded_targets, encoded_preds, target_names=le.classes_, output_dict=True
    )
    return metrics


def plot_confusion_matrix(  # noqa D103  # TODO: Remove this ignore
    results_df, classes, save_fig=True, save_path="./outputs/confusion_matrix.png"
):
    cm = confusion_matrix(y_true=results_df.target, y_pred=results_df.predicted_label_idx)
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes, rotation=0)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        coeff = f"{cm[i, j]}"
        plt.text(
            j,
            i,
            coeff,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax = fig.gca()
    ax.set_ylim(len(classes) - 0.5, -0.5)

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.grid(False)
    if save_fig:
        plt.savefig(save_path)
