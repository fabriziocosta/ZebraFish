from __future__ import annotations

import os
from pathlib import Path
import tempfile
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


def _humanize_metric_name(name: str) -> str:
    return _humanize_loss_name(name.removeprefix("train_").removeprefix("val_"))


def _loess_smooth_1d(values: np.ndarray, frac: float = 0.25) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= 1:
        return values.copy()
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    x = np.arange(n, dtype=float)
    span = max(3, int(np.ceil(frac * n)))
    span = min(span, n)
    smoothed = np.empty(n, dtype=float)

    for i in range(n):
        distances = np.abs(x - x[i])
        bandwidth = np.partition(distances, span - 1)[span - 1]
        if bandwidth == 0:
            smoothed[i] = values[i]
            continue

        scaled = distances / bandwidth
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)
        if not np.any(weights):
            smoothed[i] = values[i]
            continue

        x_centered = x - x[i]
        design = np.column_stack([np.ones(n, dtype=float), x_centered])
        weighted_design = design * weights[:, None]
        beta, *_ = np.linalg.lstsq(weighted_design.T @ design, weighted_design.T @ values, rcond=None)
        smoothed[i] = beta[0]

    return smoothed


def _rolling_median_smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=max(1, int(window)), min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _chunk_columns(columns: list[str], max_curves_per_panel: int) -> list[list[str]]:
    if max_curves_per_panel < 1:
        raise ValueError("max_curves_per_panel must be at least 1")
    return [columns[start : start + max_curves_per_panel] for start in range(0, len(columns), max_curves_per_panel)]


def _should_use_log_axis(column: str, values: np.ndarray) -> bool:
    if "lambda" in column or column.endswith("_weight"):
        return False
    finite_values = values[np.isfinite(values)]
    return bool(finite_values.size and np.nanmax(finite_values) > 0 and np.nanmin(finite_values) > 0)


def plot_independent_axis_curves(
    ax,
    history_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    title: str,
    smoothing_window: int | None = None,
    loess_frac: float | None = None,
    show_raw: bool = True,
) -> list:
    """Plot up to a few curves with one y-axis per curve."""
    if "epoch" not in history_df.columns:
        raise ValueError("history must contain an epoch column")
    valid_columns = [
        column
        for column in columns
        if column in history_df.columns and history_df[column].notna().any()
    ]
    if not valid_columns:
        ax.set_visible(False)
        return []
    if len(valid_columns) > 4:
        raise ValueError("plot_independent_axis_curves accepts at most 4 columns")

    epoch_values = history_df["epoch"].to_numpy(dtype=float)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    axes = []
    handles = []
    labels = []
    for index, column in enumerate(valid_columns):
        curve_ax = ax if index == 0 else ax.twinx()
        axes.append(curve_ax)
        if index > 0:
            curve_ax.spines["right"].set_position(("axes", 1.0 + 0.12 * (index - 1)))
            curve_ax.spines["right"].set_visible(True)
            curve_ax.grid(False)

        series = history_df[column].astype(float)
        valid_mask = series.notna().to_numpy()
        curve_epochs = epoch_values[valid_mask]
        values = series.to_numpy(dtype=float)[valid_mask]
        color = colors[index % len(colors)] if colors else None
        label = column

        if show_raw or (smoothing_window is None and loess_frac is None):
            (raw_line,) = curve_ax.plot(
                curve_epochs,
                values,
                linewidth=1.2,
                alpha=0.25 if smoothing_window is not None or loess_frac is not None else 0.9,
                color=color,
                label=label,
            )
            color = raw_line.get_color()
        if smoothing_window is not None:
            (smooth_line,) = curve_ax.plot(
                curve_epochs,
                _rolling_median_smooth_1d(values, smoothing_window),
                linewidth=2.2,
                alpha=0.95,
                color=color,
                label=label if not show_raw else "_nolegend_",
            )
            handles.append(smooth_line if show_raw else smooth_line)
        elif loess_frac is not None:
            (smooth_line,) = curve_ax.plot(
                curve_epochs,
                _loess_smooth_1d(values, frac=loess_frac),
                linewidth=2.6,
                alpha=0.95,
                color=color,
                label=label if not show_raw else "_nolegend_",
            )
            handles.append(smooth_line if show_raw else smooth_line)
        else:
            handles.append(raw_line)
        labels.append(label)

        curve_ax.set_ylabel(_humanize_metric_name(column), color=color)
        curve_ax.tick_params(axis="y", colors=color)
        curve_ax.spines["left" if index == 0 else "right"].set_color(color)
        if _should_use_log_axis(column, values):
            curve_ax.set_yscale("log", nonpositive="clip")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(
        handles,
        labels,
        fontsize="x-small",
        ncol=min(2, len(labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=True,
    )
    return axes


def plot_grouped_independent_axis_history(
    history_df: pd.DataFrame,
    groups: Sequence[tuple[str, Sequence[str]]],
    *,
    title: str,
    max_curves_per_panel: int = 4,
    smoothing_window: int | None = None,
    loess_frac: float | None = None,
    show_raw: bool = True,
):
    panels: list[tuple[str, list[str]]] = []
    for group_title, columns in groups:
        valid_columns = [
            column
            for column in columns
            if column in history_df.columns and history_df[column].notna().any()
        ]
        chunks = _chunk_columns(valid_columns, max_curves_per_panel)
        for chunk_index, chunk in enumerate(chunks, start=1):
            suffix = f" {chunk_index}" if len(chunks) > 1 else ""
            panels.append((f"{group_title}{suffix}", chunk))
    if not panels:
        raise ValueError("history does not contain any requested curve columns")

    n_cols = 2 if len(panels) > 1 else 1
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    primary_axes = axes_grid.reshape(-1)
    for index, (panel_title, columns) in enumerate(panels):
        plot_independent_axis_curves(
            primary_axes[index],
            history_df,
            columns,
            title=panel_title,
            smoothing_window=smoothing_window,
            loess_frac=loess_frac,
            show_raw=show_raw,
        )
    for empty_ax in primary_axes[len(panels) :]:
        empty_ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=4.5, h_pad=4.0)
    return fig, primary_axes[: len(panels)]


def plot_training_history(
    history,
    *,
    ax=None,
    title: str = "Training history",
    loess_frac: float | None = None,
    show_raw: bool = True,
    excluded_loss_names: Sequence[str] | None = None,
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

    excluded_loss_name_set = set(excluded_loss_names or [])
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
                "water_vs_other_loss",
                "feature_alignment_loss",
                "compound_loss",
                "concentration_loss",
            ].index(name)
            if name
            in {
                "loss",
                "action_loss",
                "water_vs_other_loss",
                "feature_alignment_loss",
                "compound_loss",
                "concentration_loss",
            }
            else 999,
            name,
        ),
    )
    loss_names = [name for name in loss_names if name not in excluded_loss_name_set]
    if not loss_names:
        raise ValueError("history does not contain any train_*_loss columns")

    groups = []
    for loss_name in loss_names:
        columns = [f"train_{loss_name}"]
        val_key = f"val_{loss_name}"
        if val_key in history_df.columns:
            columns.append(val_key)
        groups.append((_humanize_loss_name(loss_name), columns))

    if ax is not None:
        if len(groups) != 1:
            raise ValueError("ax can only be provided when plotting a single loss panel")
        fig = ax.figure
        plot_independent_axis_curves(
            ax,
            history_df,
            groups[0][1],
            title=groups[0][0],
            loess_frac=loess_frac,
            show_raw=show_raw,
        )
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        return fig, ax

    return plot_grouped_independent_axis_history(
        history_df,
        groups,
        title=title,
        max_curves_per_panel=4,
        loess_frac=loess_frac,
        show_raw=show_raw,
    )


def save_training_history_pdf(
    history,
    output_path: str | Path,
    *,
    title: str = "Training history",
    loess_frac: float | None = None,
    show_raw: bool = True,
    excluded_loss_names: Sequence[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_training_history(
        history,
        title=title,
        loess_frac=loess_frac,
        show_raw=show_raw,
        excluded_loss_names=excluded_loss_names,
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=output_path.suffix,
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            fig.savefig(temp_path, format="pdf", bbox_inches="tight")
            os.replace(temp_path, output_path)
        finally:
            temp_path.unlink(missing_ok=True)
    finally:
        plt.close(fig)
    return output_path


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


def build_confusion_matrix_frames(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    class_labels: Sequence[int] | None = None,
    label_map: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    class_labels = list(class_labels)
    tick_labels = [label_map.get(int(label), str(label)) for label in class_labels] if label_map else [str(label) for label in class_labels]

    cm_abs = confusion_matrix(y_true_arr, y_pred_arr, labels=class_labels)
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_frac = np.divide(cm_abs, row_sums, out=np.zeros_like(cm_abs, dtype=float), where=row_sums > 0)
    counts_df = pd.DataFrame(cm_abs, index=tick_labels, columns=tick_labels)
    fractions_df = pd.DataFrame(cm_frac, index=tick_labels, columns=tick_labels)
    counts_df.index.name = "true_class"
    fractions_df.index.name = "true_class"
    return counts_df, fractions_df


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

    counts_df, fractions_df = build_confusion_matrix_frames(
        y_true_arr,
        y_pred_arr,
        class_labels=class_labels,
        label_map=label_map,
    )
    cm_abs = counts_df.to_numpy()
    cm_frac = fractions_df.to_numpy()

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
        max_value = float(np.max(matrix)) if matrix.size else 0.0
        dark_threshold = 0.5 * max_value
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                if float(matrix[row_index, column_index]) == 0.0:
                    continue
                value = format(matrix[row_index, column_index], fmt)
                text_color = "white" if float(matrix[row_index, column_index]) > dark_threshold else "black"
                ax.text(column_index, row_index, value, ha="center", va="center", color=text_color)
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
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    plt.show()
    return frame


def display_multitask_reports_and_confusions(
    reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    y_true: dict[str, Iterable[int]],
    y_pred: dict[str, Iterable[int]],
    class_labels: dict[str, Sequence[int]] | None = None,
    label_maps: dict[str, dict[int, str]] | None = None,
    title_suffix: str = "",
) -> None:
    for target, (per_class_df, summary_df) in reports.items():
        print()
        print(f"## Holdout report: {target}{title_suffix}")
        display(per_class_df)
        display(summary_df)
        plot_confusion_matrices(
            y_true[target],
            y_pred[target],
            class_labels=None if class_labels is None else class_labels.get(target),
            label_map=None if label_maps is None else label_maps.get(target),
        )
        plt.show()
