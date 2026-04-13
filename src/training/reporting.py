from __future__ import annotations

from typing import Iterable, Sequence

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted


def _humanize_loss_name(name: str) -> str:
    if name == "loss":
        return "Total loss"
    parts = name.replace("_", " ").split()
    return " ".join(part.capitalize() for part in parts)


def plot_training_history(
    history,
    *,
    ax=None,
    title: str = "Training history",
):
    if isinstance(history, pd.DataFrame):
        history_df = history
    elif hasattr(history, "history_"):
        check_is_fitted(history, ["history_"])
        history_df = history.history_
    else:
        raise TypeError("history must be a pandas DataFrame or a fitted estimator exposing history_")

    if history_df.empty:
        raise ValueError("history is empty")

    loss_names = sorted(
        {
            column.removeprefix("train_")
            for column in history_df.columns
            if column.startswith("train_") and column.endswith("_loss")
        },
        key=lambda name: (
            [
                "loss",
                "action_loss",
                "commutative_consistency_loss",
                "feature_alignment_loss",
                "compound_loss",
                "concentration_loss",
            ].index(name)
            if name
            in {
                "loss",
                "action_loss",
                "commutative_consistency_loss",
                "feature_alignment_loss",
                "compound_loss",
                "concentration_loss",
            }
            else 999,
            name,
        ),
    )
    if not loss_names:
        raise ValueError("history does not contain any train_*_loss columns")

    if ax is not None and len(loss_names) != 1:
        raise ValueError("ax can only be provided when plotting a single loss panel")

    if ax is not None:
        axes = np.asarray([ax], dtype=object)
        fig = ax.figure
    else:
        n_panels = len(loss_names)
        n_cols = 2 if n_panels > 1 else 1
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows), squeeze=False)
        axes = axes_grid.reshape(-1)

    for index, loss_name in enumerate(loss_names):
        panel_ax = axes[index]
        train_key = f"train_{loss_name}"
        val_key = f"val_{loss_name}"
        panel_ax.plot(history_df["epoch"], history_df[train_key], marker="o", linewidth=1.8, label="Train")
        if val_key in history_df.columns and history_df[val_key].notna().any():
            panel_ax.plot(history_df["epoch"], history_df[val_key], marker="o", linewidth=1.8, label="Val")
        panel_ax.set_xlabel("Epoch")
        panel_ax.set_ylabel("Loss")
        panel_ax.set_title(_humanize_loss_name(loss_name))
        panel_ax.grid(True, alpha=0.25)
        panel_ax.legend()

    if ax is None and len(axes) > len(loss_names):
        for empty_ax in axes[len(loss_names) :]:
            empty_ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, (axes[: len(loss_names)] if ax is None else ax)


def build_classification_reports(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    y_proba: np.ndarray | None = None,
    class_labels: Sequence[int] | None = None,
    label_map: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length")

    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    class_labels = list(class_labels)
    class_names = [label_map.get(int(label), str(label)) for label in class_labels] if label_map else [str(label) for label in class_labels]

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    per_class_df = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()
    per_class_df.index.name = "class"

    summary_metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "n_samples": int(len(y_true_arr)),
    }

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim != 2:
            raise ValueError("y_proba must have shape (n_samples, n_classes)")
        if y_proba_arr.shape[0] != len(y_true_arr):
            raise ValueError("y_proba must have the same number of rows as y_true")
        if y_proba_arr.shape[1] != len(class_labels):
            raise ValueError("y_proba column count must match the number of class labels")

        y_true_bin = label_binarize(y_true_arr, classes=class_labels)
        if len(class_labels) == 2:
            positive_scores = y_proba_arr[:, 1]
            if y_true_bin.shape[1] == 1:
                positive_targets = y_true_bin[:, 0]
            else:
                positive_targets = y_true_bin[:, 1]
            summary_metrics["roc_auc"] = float(roc_auc_score(positive_targets, positive_scores))
            summary_metrics["average_precision"] = float(
                average_precision_score(positive_targets, positive_scores)
            )
        else:
            summary_metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, y_proba_arr, multi_class="ovr", average="macro")
            )
            summary_metrics["average_precision_macro"] = float(
                average_precision_score(y_true_bin, y_proba_arr, average="macro")
            )

    summary_df = pd.DataFrame([summary_metrics]).T.rename(columns={0: "value"})
    return per_class_df, summary_df


def build_multitask_classification_reports(
    y_true: dict[str, Iterable[int]],
    y_pred: dict[str, Iterable[int]],
    *,
    y_proba: dict[str, np.ndarray] | None = None,
    class_labels: dict[str, Sequence[int]] | None = None,
    label_maps: dict[str, dict[int, str]] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target, target_y_true in y_true.items():
        if target not in y_pred:
            raise KeyError(f"Missing predictions for target {target!r}")
        target_labels = class_labels.get(target) if class_labels is not None else None
        target_label_map = label_maps.get(target) if label_maps is not None else None
        target_proba = y_proba.get(target) if y_proba is not None else None
        results[target] = build_classification_reports(
            target_y_true,
            y_pred[target],
            y_proba=target_proba,
            class_labels=target_labels,
            label_map=target_label_map,
        )
    return results


def plot_confusion_matrices(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    class_labels: Sequence[int] | None = None,
    label_map: dict[int, str] | None = None,
    axes=None,
    cmap: str = "Blues",
):
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    class_labels = list(class_labels)
    tick_labels = [label_map.get(int(label), str(label)) for label in class_labels] if label_map else [str(label) for label in class_labels]

    cm_abs = confusion_matrix(y_true_arr, y_pred_arr, labels=class_labels)
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_frac = np.divide(cm_abs, row_sums, out=np.zeros_like(cm_abs, dtype=float), where=row_sums > 0)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes[0].figure

    panels = [
        (axes[0], cm_abs, "Confusion matrix (counts)", "d"),
        (axes[1], cm_frac, "Confusion matrix (row fractions)", ".2f"),
    ]
    for ax, matrix, title, fmt in panels:
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = format(matrix[row_index, column_index], fmt)
                ax.text(column_index, row_index, value, ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, axes, cm_abs, cm_frac


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: Iterable[int],
    label_map: dict[int, str],
    *,
    title: str,
):
    labels_array = pd.Series(list(labels)).astype(int).to_numpy()
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    frame = pd.DataFrame(
        {
            "embed_x": coords[:, 0],
            "embed_y": coords[:, 1],
            "label": labels_array,
            "label_name": [label_map.get(int(label), str(int(label))) for label in labels_array],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for label_value, group_df in frame.groupby("label", sort=True):
        ax.scatter(
            group_df["embed_x"],
            group_df["embed_y"],
            s=42,
            alpha=0.82,
            label=label_map.get(int(label_value), str(int(label_value))),
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    plt.show()
    return frame


def display_multitask_reports_and_confusions(
    reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    y_true: dict[str, Iterable[int]],
    y_pred: dict[str, Iterable[int]],
    class_labels: dict[str, Sequence[int]] | None = None,
    label_maps: dict[str, dict[int, str]] | None = None,
) -> None:
    for target, (per_class_df, summary_df) in reports.items():
        print()
        print(f"## Holdout report: {target}")
        display(per_class_df)
        display(summary_df)
        plot_confusion_matrices(
            y_true[target],
            y_pred[target],
            class_labels=None if class_labels is None else class_labels.get(target),
            label_map=None if label_maps is None else label_maps.get(target),
        )
        plt.show()
