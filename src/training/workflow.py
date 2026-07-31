from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

from IPython.display import display
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from src.training.data import augment_training_tensors_with_rotations, split_labeled_tensor_dataset_by_instance
from src.training.loop import _format_eta
from src.training.reporting import (
    build_confusion_matrix_frames,
    build_multitask_classification_reports,
    display_multitask_reports_and_confusions,
    plot_embedding_projection,
)


@dataclass
class MultitaskExperimentData:
    splits: Any
    X_train: torch.Tensor
    y_train: pd.Series
    compound_train: pd.Series | None
    concentration_train: pd.Series | None
    train_metadata: pd.DataFrame | None
    y_true_holdout: dict[str, Any]
    label_maps: dict[str, dict[int, str]]
    class_labels: dict[str, list[int]]
    y_true_lockbox: dict[str, Any] = field(default_factory=dict)
    split_manifest: dict[str, Any] | None = None


@dataclass
class BinaryWaterVsOtherPretrainingData:
    X_train: torch.Tensor
    y_train: pd.Series
    X_val: torch.Tensor
    y_val: np.ndarray
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    label_map: dict[int, str]
    excluded_holdout_count: int


@dataclass
class ChunkedBinaryWaterVsOtherPretrainingResult:
    metadata: pd.DataFrame
    label_map: dict[int, str]
    excluded_holdout_count: int
    train_count: int
    val_count: int


@dataclass
class ExperimentArtifacts:
    output_dir: str
    config_path: str
    history_path: str
    summary_metrics_path: str
    per_class_dir: str
    checkpoint_path: str
    experiment_id: str | None = None
    confusion_dir: str | None = None
    predictions_dir: str | None = None
    logbook_path: str | None = None
    split_manifest_path: str | None = None
    lockbox_summary_metrics_path: str | None = None


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    run_dir: str
    loss_plot_dir: str
    figure_dir: str


@dataclass
class MultitaskEvaluationResult:
    predictions: dict[str, Any]
    probabilities: dict[str, Any]
    reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    reports_excluding_control: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None
    predictions_excluding_control: dict[str, Any] | None = None
    probabilities_excluding_control: dict[str, Any] | None = None
    y_true_excluding_control: dict[str, Any] | None = None


def _repeat_labels_for_rotation_augmentation(
    labels: torch.Tensor | np.ndarray,
    *,
    num_random_rotations: int,
) -> np.ndarray:
    values = np.asarray(labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels, dtype=int).reshape(-1)
    return np.repeat(values, int(num_random_rotations) + 1)


def _unlabeled_dataset_chunk_paths(path: str | Path) -> list[Path]:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        manifest_path = dataset_path / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return [dataset_path / chunk_name for chunk_name in payload["chunks"]]
        return sorted(dataset_path.glob("*.pt"))
    return [dataset_path]


def _to_json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_compatible(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


def _sanitize_experiment_prefix(prefix: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix).strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("experiment_prefix must contain at least one alphanumeric character")
    return sanitized


def make_experiment_id(experiment_prefix: str, *, timestamp: datetime | None = None) -> str:
    timestamp_value = timestamp or datetime.now()
    return f"{_sanitize_experiment_prefix(experiment_prefix)}_{timestamp_value:%Y%m%d_%H%M%S}"


def _append_experiments_logbook_entry(
    *,
    experiment_id: str,
    experiment_kind: str,
    artifact_dir: Path,
    key_paths: dict[str, Path | str | list[Path | str] | tuple[Path | str, ...] | None],
    config: dict[str, Any] | Any,
    analysis: str | None,
    next_round_proposal: str | None,
    logbook_path: str | Path | None,
) -> str | None:
    if logbook_path is None:
        return None

    path = Path(logbook_path)
    header = (
        "# Experiments Logbook\n\n"
        "This file records pretraining and fine-tuning runs by experiment id. "
        "Each entry links the timestamped artifacts, summarizes what happened, "
        "and states the proposed next round.\n"
    )
    if not path.exists():
        path.write_text(header, encoding="utf-8")

    compact_config = _to_json_compatible(config)
    config_lines = json.dumps(compact_config, indent=2, sort_keys=True).splitlines()
    if len(config_lines) > 28:
        config_block = "\n".join(config_lines[:28] + ["  ..."])
    else:
        config_block = "\n".join(config_lines)

    path_lines = [
        f"- `{name}`: {_format_logbook_path_value(value)}"
        for name, value in key_paths.items()
        if value is not None
    ]
    preview_lines = _logbook_preview_lines(key_paths)
    entry_lines = [
        "",
        f"## {experiment_id}",
        "",
        f"- kind: `{experiment_kind}`",
        f"- artifact_dir: {_markdown_artifact_link(artifact_dir)}",
        *path_lines,
        "",
        "### Analysis",
        "",
        (analysis or "TODO: summarize the observed losses, metrics, and failure modes."),
        "",
        "### Next Round Proposal",
        "",
        (next_round_proposal or "TODO: state the next experiment and the exact change being tested."),
        "",
        "### Config Snapshot",
        "",
        "```json",
        config_block,
        "```",
        "",
    ]
    if preview_lines:
        insert_at = entry_lines.index("### Config Snapshot")
        entry_lines[insert_at:insert_at] = ["### Plot Previews", "", *preview_lines, ""]

    entry = "\n".join(entry_lines).rstrip() + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else header
    section_pattern = re.compile(rf"(?ms)^## {re.escape(experiment_id)}\n.*?(?=^## |\Z)")
    if section_pattern.search(current):
        updated = section_pattern.sub(entry.lstrip(), current).rstrip() + "\n"
    else:
        updated = current.rstrip() + "\n\n" + entry.lstrip()
    path.write_text(updated, encoding="utf-8")
    return str(path)


def _markdown_artifact_link(path: Path | str) -> str:
    path_text = str(path)
    label = Path(path_text).name or path_text
    target = path_text.replace(" ", "%20")
    return f"[{label}]({target})"


def _format_logbook_path_value(value: Path | str | list[Path | str] | tuple[Path | str, ...]) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_markdown_artifact_link(item) for item in value)
    return _markdown_artifact_link(value)


def _logbook_preview_lines(
    key_paths: dict[str, Path | str | list[Path | str] | tuple[Path | str, ...] | None],
) -> list[str]:
    preview_candidates: list[Path] = []
    for name in ("latest_loss_pdfs", "figure_pdfs"):
        value = key_paths.get(name)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            item_path = Path(str(item))
            if item_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
                preview_candidates.append(item_path)
    return [f"![{path.name}]({str(path).replace(' ', '%20')})" for path in preview_candidates]


def create_experiment_run(output_dir: str | Path, experiment_prefix: str) -> ExperimentRun:
    experiment_id = make_experiment_id(experiment_prefix)
    run_dir = Path(output_dir) / "runs" / experiment_id
    loss_plot_dir = run_dir / "loss_plots"
    figure_dir = run_dir / "figures"
    loss_plot_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentRun(
        experiment_id=experiment_id,
        run_dir=str(run_dir),
        loss_plot_dir=str(loss_plot_dir),
        figure_dir=str(figure_dir),
    )


def _artifact_run_paths(
    output_dir: str | Path,
    *,
    experiment_prefix: str | None,
    experiment_id: str | None = None,
) -> tuple[Path, str | None, dict[str, Path]]:
    base_output_path = Path(output_dir)
    if experiment_prefix is None:
        paths = {
            "config": base_output_path / "config.json",
            "history": base_output_path / "history.csv",
            "summary": base_output_path / "summary_metrics.csv",
            "checkpoint": base_output_path / "model_state.pt",
            "per_class_dir": base_output_path / "per_class_reports",
            "confusion_dir": base_output_path / "confusion_matrices",
            "predictions_dir": base_output_path / "predictions",
        }
        return base_output_path, None, paths

    if experiment_id is None:
        experiment_id = make_experiment_id(experiment_prefix)
    else:
        experiment_id = _sanitize_experiment_prefix(experiment_id)
    run_dir = base_output_path / "runs" / experiment_id
    paths = {
        "config": run_dir / f"{experiment_id}_config.json",
        "history": run_dir / f"{experiment_id}_history.csv",
        "summary": run_dir / f"{experiment_id}_summary_metrics.csv",
        "checkpoint": run_dir / f"{experiment_id}_model_state.pt",
        "per_class_dir": run_dir / "per_class_reports",
        "confusion_dir": run_dir / "confusion_matrices",
        "predictions_dir": run_dir / "predictions",
    }
    return run_dir, experiment_id, paths


def _summary_frames_from_reports(reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    summary_frames: list[pd.DataFrame] = []
    for target, (_, summary_df) in reports.items():
        target_summary = summary_df.rename_axis("metric").reset_index()
        target_summary.insert(0, "target", target)
        summary_frames.append(target_summary)
    if not summary_frames:
        return pd.DataFrame(columns=["target", "metric", "value"])
    return pd.concat(summary_frames, ignore_index=True)


def _save_multitask_prediction_tables(
    *,
    predictions_dir: Path,
    experiment_id: str | None,
    evaluation: MultitaskEvaluationResult,
    experiment: MultitaskExperimentData,
) -> dict[str, Path]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_id}_" if experiment_id else ""
    output_paths: dict[str, Path] = {}
    for target, y_true in experiment.y_true_holdout.items():
        if target not in evaluation.predictions:
            continue
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(evaluation.predictions[target], dtype=int)
        label_map = experiment.label_maps.get(target, {})
        table = pd.DataFrame(
            {
                "true_label": y_true_arr,
                "true_name": [label_map.get(int(label), str(int(label))) for label in y_true_arr],
                "pred_label": y_pred_arr,
                "pred_name": [label_map.get(int(label), str(int(label))) for label in y_pred_arr],
            }
        )
        metadata = experiment.splits.metadata_holdout.reset_index(drop=True)
        if len(metadata) != len(table):
            raise ValueError("holdout prediction rows are not aligned with holdout metadata")
        for column in metadata.columns:
            if column not in table:
                table[column] = metadata[column].to_numpy()
        probabilities = evaluation.probabilities.get(target)
        class_labels = experiment.class_labels.get(target, [])
        if probabilities is not None:
            probabilities_arr = np.asarray(probabilities, dtype=float)
            for column_index, label in enumerate(class_labels):
                label_name = label_map.get(int(label), str(int(label)))
                safe_label_name = _sanitize_experiment_prefix(label_name)
                table[f"proba_{safe_label_name}"] = probabilities_arr[:, column_index]
        path = predictions_dir / f"{prefix}{target}_predictions.csv"
        table.to_csv(path, index=False)
        output_paths[target] = path
    return output_paths


def _save_multitask_confusion_matrices(
    *,
    confusion_dir: Path,
    experiment_id: str | None,
    evaluation: MultitaskEvaluationResult,
    experiment: MultitaskExperimentData,
) -> dict[str, Path]:
    confusion_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_id}_" if experiment_id else ""
    output_paths: dict[str, Path] = {}
    for target, y_true in experiment.y_true_holdout.items():
        if target not in evaluation.predictions:
            continue
        counts_df, fractions_df = build_confusion_matrix_frames(
            y_true,
            evaluation.predictions[target],
            class_labels=experiment.class_labels.get(target),
            label_map=experiment.label_maps.get(target),
        )
        counts_path = confusion_dir / f"{prefix}{target}_confusion_counts.csv"
        fractions_path = confusion_dir / f"{prefix}{target}_confusion_row_fractions.csv"
        counts_df.to_csv(counts_path)
        fractions_df.to_csv(fractions_path)
        output_paths[f"{target}_counts"] = counts_path
        output_paths[f"{target}_row_fractions"] = fractions_path
    return output_paths


def _copy_pdf_artifacts(
    *,
    source_dirs: list[str | Path] | None,
    destination_dir: Path,
    experiment_id: str | None,
) -> list[Path]:
    if not source_dirs:
        return []
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    prefix = f"{experiment_id}_" if experiment_id else ""
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            continue
        for pdf_path in sorted(source_path.rglob("*.pdf")):
            relative_stem = "_".join(pdf_path.relative_to(source_path).with_suffix("").parts)
            destination_path = destination_dir / f"{prefix}{_sanitize_experiment_prefix(relative_stem)}.pdf"
            shutil.copy2(pdf_path, destination_path)
            copied_paths.append(destination_path)
    return copied_paths


def _latest_loss_pdf_paths(loss_plot_dir: Path) -> list[Path]:
    if not loss_plot_dir.exists():
        return []
    return sorted(loss_plot_dir.rglob("*latest*.pdf"))


def _figure_pdf_paths(figure_dir: Path) -> list[Path]:
    if not figure_dir.exists():
        return []
    return sorted(figure_dir.rglob("*.pdf"))


def prepare_multitask_experiment_data(
    dataset: dict[str, object],
    *,
    holdout_fraction: float,
    validation_fraction_within_train: float,
    lockbox_fraction: float = 0.0,
    lockbox_evaluation: bool = False,
    train_num_random_rotations: int = 0,
    rotation_range_degrees: float = 5.0,
    random_state: int = 0,
    split_random_state: int | None = None,
) -> MultitaskExperimentData:
    splits = split_labeled_tensor_dataset_by_instance(
        dataset,
        holdout_fraction=holdout_fraction,
        validation_fraction_within_train=validation_fraction_within_train,
        lockbox_fraction=lockbox_fraction,
        split_random_state=split_random_state,
        random_state=random_state,
    )
    X_train, y_train, train_metadata = augment_training_tensors_with_rotations(
        splits.X_train_base,
        splits.y_train_base,
        metadata=splits.metadata_train_base,
        num_random_rotations=train_num_random_rotations,
        rotation_range_degrees=rotation_range_degrees,
        random_state=random_state,
    )
    compound_train = None
    concentration_train = None
    if splits.compound_train_base is not None:
        compound_train = _repeat_labels_for_rotation_augmentation(
            splits.compound_train_base,
            num_random_rotations=train_num_random_rotations,
        )
    if splits.concentration_train_base is not None:
        concentration_train = _repeat_labels_for_rotation_augmentation(
            splits.concentration_train_base,
            num_random_rotations=train_num_random_rotations,
        )

    label_maps = {
        "action": {int(k): str(v) for k, v in dataset["label_map"].items()},
        "compound": {int(k): str(v) for k, v in dataset.get("compound_label_map", {}).items()},
        "concentration": {int(k): str(v) for k, v in dataset.get("concentration_label_map", {}).items()},
    }
    class_labels = {
        "action": sorted(int(k) for k in dataset["label_map"].keys()),
        "compound": sorted(int(k) for k in dataset.get("compound_label_map", {}).keys()),
        "concentration": sorted(int(k) for k in dataset.get("concentration_label_map", {}).keys()),
    }
    y_true_holdout = {
        "action": splits.y_holdout,
        "compound": splits.compound_holdout,
        "concentration": splits.concentration_holdout,
    }
    y_true_holdout = {k: v for k, v in y_true_holdout.items() if v is not None}
    label_maps = {k: v for k, v in label_maps.items() if v}
    class_labels = {k: v for k, v in class_labels.items() if v}

    return MultitaskExperimentData(
        splits=splits,
        X_train=X_train,
        y_train=pd.Series(y_train),
        compound_train=None if compound_train is None else pd.Series(compound_train),
        concentration_train=None if concentration_train is None else pd.Series(concentration_train),
        train_metadata=train_metadata,
        y_true_holdout=y_true_holdout,
        label_maps=label_maps,
        class_labels=class_labels,
        y_true_lockbox=(
            {
                "action": splits.y_lockbox,
                "compound": splits.compound_lockbox,
                "concentration": splits.concentration_lockbox,
            }
            if lockbox_evaluation
            else {}
        ),
        split_manifest=splits.split_manifest,
    )


def prepare_water_vs_other_pretraining_data(
    unlabeled_dataset: dict[str, object],
    *,
    holdout_metadata: pd.DataFrame,
    validation_fraction: float,
    train_num_random_rotations: int = 0,
    rotation_range_degrees: float = 5.0,
    random_state: int = 0,
) -> BinaryWaterVsOtherPretrainingData:
    tensors = unlabeled_dataset["tensors"]
    metadata = unlabeled_dataset["metadata"]
    if not isinstance(tensors, torch.Tensor):
        raise TypeError("unlabeled_dataset['tensors'] must be a torch.Tensor")
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("unlabeled_dataset['metadata'] must be a pandas DataFrame")
    if "image_condition_dir" not in metadata.columns:
        raise KeyError("unlabeled dataset metadata must contain 'image_condition_dir'")
    if "condition_kind" not in metadata.columns:
        raise KeyError("unlabeled dataset metadata must contain 'condition_kind'")
    if "image_condition_dir" not in holdout_metadata.columns:
        raise KeyError("holdout_metadata must contain 'image_condition_dir'")
    if len(tensors) != len(metadata):
        raise ValueError("unlabeled tensors and metadata must have the same number of rows")

    metadata_df = metadata.reset_index(drop=True).copy()
    holdout_dirs = set(holdout_metadata["image_condition_dir"].astype(str))
    is_holdout = metadata_df["image_condition_dir"].astype(str).isin(holdout_dirs).to_numpy()
    keep_indices = np.flatnonzero(~is_holdout)
    if len(keep_indices) == 0:
        raise ValueError("No unlabeled examples remain after excluding supervised holdout image_condition_dir values")

    selected_tensors = tensors[keep_indices]
    selected_metadata = metadata_df.iloc[keep_indices].reset_index(drop=True)
    labels = np.where(selected_metadata["condition_kind"].astype(str).eq("control"), 0, 1).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Water-vs-other pretraining requires both control and treatment examples")

    indices = np.arange(len(labels))
    stratify = labels if len(np.unique(labels)) > 1 else None
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=None,
        )

    X_train, y_train, train_metadata = augment_training_tensors_with_rotations(
        selected_tensors[train_indices],
        labels[train_indices],
        metadata=selected_metadata.iloc[train_indices].reset_index(drop=True),
        num_random_rotations=train_num_random_rotations,
        rotation_range_degrees=rotation_range_degrees,
        random_state=random_state,
    )
    if train_metadata is None:
        train_metadata = pd.DataFrame()

    return BinaryWaterVsOtherPretrainingData(
        X_train=X_train,
        y_train=pd.Series(y_train),
        X_val=selected_tensors[val_indices],
        y_val=labels[val_indices],
        train_metadata=train_metadata,
        val_metadata=selected_metadata.iloc[val_indices].reset_index(drop=True),
        label_map={0: "Water", 1: "Other"},
        excluded_holdout_count=int(is_holdout.sum()),
    )


def _scan_unlabeled_chunks_for_binary_pretraining(
    unlabeled_dataset_path: str | Path,
    *,
    holdout_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if "image_condition_dir" not in holdout_metadata.columns:
        raise KeyError("holdout_metadata must contain 'image_condition_dir'")
    holdout_dirs = set(holdout_metadata["image_condition_dir"].astype(str))
    rows: list[dict[str, object]] = []
    excluded_holdout_count = 0

    for chunk_path in _unlabeled_dataset_chunk_paths(unlabeled_dataset_path):
        payload = torch.load(chunk_path, map_location="cpu")
        metadata = pd.DataFrame(payload["metadata_records"])
        if "image_condition_dir" not in metadata.columns:
            raise KeyError("unlabeled dataset metadata must contain 'image_condition_dir'")
        if "condition_kind" not in metadata.columns:
            raise KeyError("unlabeled dataset metadata must contain 'condition_kind'")

        is_holdout = metadata["image_condition_dir"].astype(str).isin(holdout_dirs).to_numpy()
        excluded_holdout_count += int(is_holdout.sum())
        kept_metadata = metadata.loc[~is_holdout].reset_index()
        for _, row in kept_metadata.iterrows():
            condition_kind = str(row["condition_kind"])
            rows.append(
                {
                    "chunk_path": str(chunk_path),
                    "local_index": int(row["index"]),
                    "image_condition_dir": str(row["image_condition_dir"]),
                    "condition_kind": condition_kind,
                    "label": 0 if condition_kind == "control" else 1,
                }
            )
        del payload

    metadata_df = pd.DataFrame(rows)
    if metadata_df.empty:
        raise ValueError("No unlabeled examples remain after excluding supervised holdout image_condition_dir values")
    if metadata_df["label"].nunique() < 2:
        raise ValueError("Water-vs-other pretraining requires both control and treatment examples")
    return metadata_df, excluded_holdout_count


def _assign_chunked_binary_train_val(
    metadata_df: pd.DataFrame,
    *,
    validation_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    indices = np.arange(len(metadata_df))
    labels = metadata_df["label"].to_numpy(dtype=int)
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=None,
        )
    result = metadata_df.copy()
    result["split"] = "train"
    result.iloc[val_indices, result.columns.get_loc("split")] = "val"
    result.iloc[train_indices, result.columns.get_loc("split")] = "train"
    return result


def _compute_chunked_train_standardization(metadata_df: pd.DataFrame) -> tuple[float, float]:
    total_sum = 0.0
    total_square_sum = 0.0
    total_count = 0
    train_df = metadata_df[metadata_df["split"] == "train"]
    for chunk_path, chunk_rows in train_df.groupby("chunk_path", sort=False):
        payload = torch.load(chunk_path, map_location="cpu")
        tensors = payload["tensors"]
        indices = torch.tensor(chunk_rows["local_index"].to_numpy(dtype=np.int64), dtype=torch.long)
        selected = tensors.index_select(0, indices).to(torch.float32)
        total_sum += float(selected.sum().item())
        total_square_sum += float(selected.square().sum().item())
        total_count += int(selected.numel())
        del selected, payload
    if total_count == 0:
        raise ValueError("No training examples selected for chunked binary pretraining")
    mean = total_sum / total_count
    variance = max(total_square_sum / total_count - mean**2, 0.0)
    std = float(np.sqrt(variance))
    return float(mean), std if std > 0 else 1.0


def fit_chunked_water_vs_other_hot_start(
    estimator,
    unlabeled_dataset_path: str | Path,
    *,
    holdout_metadata: pd.DataFrame,
    validation_fraction: float,
    epochs: int,
    random_state: int = 0,
    class_weighting: str | None = "balanced",
) -> ChunkedBinaryWaterVsOtherPretrainingResult:
    """Fit a binary water-vs-other hot-start from unlabeled chunks without concatenating them."""
    metadata_df, excluded_holdout_count = _scan_unlabeled_chunks_for_binary_pretraining(
        unlabeled_dataset_path,
        holdout_metadata=holdout_metadata,
    )
    metadata_df = _assign_chunked_binary_train_val(
        metadata_df,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    estimator.classes_ = np.array([0, 1])
    estimator.class_to_index_ = {0: 0, 1: 1}
    estimator.compound_classes_ = None
    estimator.compound_class_to_index_ = None
    estimator.concentration_classes_ = None
    estimator.concentration_class_to_index_ = None
    if estimator.standardize:
        estimator.input_mean_, estimator.input_std_ = _compute_chunked_train_standardization(metadata_df)
    else:
        estimator.input_mean_, estimator.input_std_ = 0.0, 1.0

    first_payload = torch.load(metadata_df.iloc[0]["chunk_path"], map_location="cpu")
    first_tensor = first_payload["tensors"][int(metadata_df.iloc[0]["local_index"])]
    estimator.input_shape_ = tuple(int(size) for size in first_tensor.shape)
    del first_payload

    hot_start_state = None
    if getattr(estimator, "hot_start", False) and hasattr(estimator, "model_"):
        hot_start_state = {key: value.detach().cpu() for key, value in estimator.model_.state_dict().items()}

    estimator.model_ = estimator._build_model(num_classes=2)
    estimator.device_ = estimator._device()
    estimator.model_.to(estimator.device_)
    if hasattr(estimator, "_load_pretrained_weights_into_model"):
        estimator._load_pretrained_weights_into_model(estimator.model_)
    if hot_start_state is not None:
        from src.training.loop import _load_compatible_state_dict

        loaded_keys, skipped_keys = _load_compatible_state_dict(estimator.model_, hot_start_state)
        estimator.hot_start_loaded_keys_ = loaded_keys
        estimator.hot_start_skipped_keys_ = skipped_keys

    train_labels = metadata_df.loc[metadata_df["split"] == "train", "label"].to_numpy(dtype=np.int64)
    criterion_weight = None
    if class_weighting == "balanced":
        class_counts = np.bincount(train_labels, minlength=2).astype(float)
        class_counts[class_counts == 0] = 1.0
        criterion_weight = torch.tensor(
            class_counts.sum() / (len(class_counts) * class_counts),
            dtype=torch.float32,
            device=estimator.device_,
        )
    elif class_weighting not in {None, "none"}:
        raise ValueError("class_weighting must be one of 'balanced', 'none', or None")
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = torch.optim.Adam(
        [parameter for parameter in estimator.model_.parameters() if parameter.requires_grad],
        lr=estimator.learning_rate,
        weight_decay=estimator.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=estimator.scheduler_factor,
            patience=estimator.scheduler_patience,
            min_lr=estimator.scheduler_min_lr,
        )
        if estimator.scheduler_patience is not None and estimator.scheduler_patience >= 0
        else None
    )

    rng = np.random.default_rng(random_state)
    history_rows: list[dict[str, float | int]] = []
    best_state = deepcopy(estimator.model_.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    # The binary hot-start is a separate, short phase. Reusing the later
    # multiclass warm-up can leave best_state at its initial weights.
    early_stopping_start_epoch = 1
    training_start = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        estimator.model_.train()
        train_loss_sum = 0.0
        train_count = 0
        train_correct = 0
        train_chunks = list(metadata_df[metadata_df["split"] == "train"].groupby("chunk_path", sort=False))
        rng.shuffle(train_chunks)
        for chunk_path, chunk_rows in train_chunks:
            payload = torch.load(chunk_path, map_location="cpu")
            tensors = payload["tensors"]
            local_indices = chunk_rows["local_index"].to_numpy(dtype=np.int64)
            labels = chunk_rows["label"].to_numpy(dtype=np.int64)
            order = rng.permutation(len(local_indices))
            for start in range(0, len(order), int(estimator.batch_size)):
                batch_order = order[start : start + int(estimator.batch_size)]
                index_tensor = torch.tensor(local_indices[batch_order], dtype=torch.long)
                X_batch = tensors.index_select(0, index_tensor).to(torch.float32)
                X_batch = ((X_batch - estimator.input_mean_) / estimator.input_std_).to(estimator.device_)
                y_batch = torch.from_numpy(labels[batch_order]).to(estimator.device_)
                optimizer.zero_grad(set_to_none=True)
                outputs = estimator.model_(X_batch)
                loss, _ = estimator._compute_losses(outputs, y_batch, criterion)
                loss.backward()
                optimizer.step()
                batch_size = int(X_batch.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_correct += int((outputs["logits"].argmax(dim=1) == y_batch).sum().item())
                train_count += batch_size
            del payload

        estimator.model_.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_correct = 0
        with torch.no_grad():
            val_df = metadata_df[metadata_df["split"] == "val"]
            for chunk_path, chunk_rows in val_df.groupby("chunk_path", sort=False):
                payload = torch.load(chunk_path, map_location="cpu")
                tensors = payload["tensors"]
                local_indices = chunk_rows["local_index"].to_numpy(dtype=np.int64)
                labels = chunk_rows["label"].to_numpy(dtype=np.int64)
                for start in range(0, len(local_indices), int(estimator.batch_size)):
                    batch_indices = local_indices[start : start + int(estimator.batch_size)]
                    index_tensor = torch.tensor(batch_indices, dtype=torch.long)
                    X_batch = tensors.index_select(0, index_tensor).to(torch.float32)
                    X_batch = ((X_batch - estimator.input_mean_) / estimator.input_std_).to(estimator.device_)
                    y_batch = torch.from_numpy(labels[start : start + int(estimator.batch_size)]).to(estimator.device_)
                    outputs = estimator.model_(X_batch)
                    loss, _ = estimator._compute_losses(outputs, y_batch, criterion)
                    batch_size = int(X_batch.shape[0])
                    val_loss_sum += float(loss.item()) * batch_size
                    val_correct += int((outputs["logits"].argmax(dim=1) == y_batch).sum().item())
                    val_count += batch_size
                del payload

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_action_loss": train_loss_sum / max(train_count, 1),
            "train_accuracy": train_correct / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_action_loss": val_loss_sum / max(val_count, 1),
            "val_accuracy": val_correct / max(val_count, 1),
        }
        metric = float(row["val_loss"])
        if epoch >= early_stopping_start_epoch:
            if metric < best_metric - float(estimator.early_stopping_min_delta):
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = deepcopy(estimator.model_.state_dict())
            else:
                epochs_without_improvement += 1
        if scheduler is not None and epoch >= early_stopping_start_epoch:
            scheduler.step(metric)
        history_rows.append(row)
        plot_dir = getattr(estimator, "training_plot_dir", None)
        every_n_epochs = max(1, int(getattr(estimator, "training_plot_every_n_epochs", 1) or 1))
        if plot_dir and epoch % every_n_epochs == 0:
            from src.training.reporting import save_training_history_pdf

            output_dir = Path(str(plot_dir)).expanduser()
            title = str(getattr(estimator, "training_plot_title", "Training history"))
            history_df = pd.DataFrame(history_rows)
            save_training_history_pdf(
                history_df,
                output_dir / f"epoch_{epoch:03d}.loss-curves.pdf",
                title=title,
            )
            save_training_history_pdf(
                history_df,
                output_dir / "latest.loss-curves.pdf",
                title=title,
            )
        if estimator.verbose:
            elapsed = time.perf_counter() - training_start
            eta = _format_eta((elapsed / epoch) * (int(epochs) - epoch))
            print(
                f"{epoch:03d}/{int(epochs):03d} lr={optimizer.param_groups[0]['lr']:.2e} eta={eta} "
                f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"train_acc={row['train_accuracy']:.3f} val_acc={row['val_accuracy']:.3f}"
            )
        if (
            epoch >= early_stopping_start_epoch
            and estimator.early_stopping_patience is not None
            and epochs_without_improvement >= estimator.early_stopping_patience
        ):
            stopped_early = True
            if estimator.verbose:
                print(
                    f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_metric={best_metric:.4f}"
                )
            break

    if estimator.verbose and not stopped_early:
        print(
            f"select_best epoch={len(history_rows):03d} best_epoch={best_epoch or len(history_rows):03d} "
            f"best_metric={best_metric:.4f}"
        )

    estimator.model_.load_state_dict(best_state)
    estimator.model_.eval()
    estimator.history_ = pd.DataFrame(history_rows)
    estimator.best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
    estimator.best_metric_ = float(best_metric)
    return ChunkedBinaryWaterVsOtherPretrainingResult(
        metadata=metadata_df,
        label_map={0: "Water", 1: "Other"},
        excluded_holdout_count=int(excluded_holdout_count),
        train_count=int((metadata_df["split"] == "train").sum()),
        val_count=int((metadata_df["split"] == "val").sum()),
    )


def evaluate_multitask_estimator(
    estimator,
    X: torch.Tensor,
    y_true: dict[str, Any],
    *,
    label_maps: dict[str, dict[int, str]] | None = None,
    class_labels: dict[str, list[int]] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)
    filtered_pred = {target: y_pred[target] for target in y_true}
    filtered_proba = {target: y_proba[target] for target in y_true if target in y_proba}
    return build_multitask_classification_reports(
        y_true,
        filtered_pred,
        y_proba=filtered_proba,
        label_maps=label_maps,
        class_labels=class_labels,
    )


def persist_lockbox_evaluation(
    *,
    output_dir: str | Path,
    estimator: Any,
    experiment: MultitaskExperimentData,
) -> Path:
    """Evaluate the protected split into a separate, explicitly named artifact.

    The file is never merged into ordinary validation summaries. Callers must
    invoke this only after the exploratory replicate gate has frozen selection.
    """

    if not experiment.y_true_lockbox or len(experiment.splits.X_lockbox) == 0:
        raise ValueError("lockbox evaluation requested but no protected split is available")
    reports = evaluate_multitask_estimator(
        estimator,
        experiment.splits.X_lockbox,
        experiment.y_true_lockbox,
        label_maps=experiment.label_maps,
        class_labels=experiment.class_labels,
    )
    path = Path(output_dir) / "lockbox_summary_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _summary_frames_from_reports(reports).to_csv(path, index=False)
    return path


def _renormalize_probabilities_without_control(
    probabilities: np.ndarray,
    *,
    class_labels: list[int],
    keep_class_labels: list[int],
) -> np.ndarray:
    class_to_index = {int(label): index for index, label in enumerate(class_labels)}
    keep_indices = [class_to_index[int(label)] for label in keep_class_labels]
    kept_probabilities = np.asarray(probabilities, dtype=float)[:, keep_indices]
    row_sums = kept_probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        kept_probabilities,
        row_sums,
        out=np.full_like(kept_probabilities, 1.0 / max(len(keep_indices), 1)),
        where=row_sums > 0,
    )


def build_reports_excluding_control(
    *,
    y_true: dict[str, Any],
    probabilities: dict[str, Any],
    class_labels: dict[str, list[int]],
    label_maps: dict[str, dict[int, str]],
    control_label: int = 0,
) -> tuple[
    dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    filtered_true: dict[str, Any] = {}
    filtered_pred: dict[str, Any] = {}
    filtered_probabilities: dict[str, Any] = {}
    filtered_class_labels: dict[str, list[int]] = {}
    filtered_label_maps: dict[str, dict[int, str]] = {}

    for target, target_y_true in y_true.items():
        target_class_labels = [int(label) for label in class_labels.get(target, [])]
        if control_label not in target_class_labels or target not in probabilities:
            continue

        keep_class_labels = [label for label in target_class_labels if label != int(control_label)]
        if not keep_class_labels:
            continue

        y_true_arr = np.asarray(target_y_true, dtype=int)
        row_mask = y_true_arr != int(control_label)
        if not row_mask.any():
            continue

        renormalized_proba = _renormalize_probabilities_without_control(
            np.asarray(probabilities[target])[row_mask],
            class_labels=target_class_labels,
            keep_class_labels=keep_class_labels,
        )
        predicted_indices = renormalized_proba.argmax(axis=1)
        predicted_labels = np.asarray([keep_class_labels[index] for index in predicted_indices], dtype=int)

        filtered_true[target] = y_true_arr[row_mask]
        filtered_pred[target] = predicted_labels
        filtered_probabilities[target] = renormalized_proba
        filtered_class_labels[target] = keep_class_labels
        filtered_label_maps[target] = {
            label: label_maps.get(target, {}).get(label, str(label))
            for label in keep_class_labels
        }

    reports = build_multitask_classification_reports(
        filtered_true,
        filtered_pred,
        y_proba=filtered_probabilities,
        label_maps=filtered_label_maps,
        class_labels=filtered_class_labels,
    )
    return reports, filtered_true, filtered_pred, filtered_probabilities


def fit_estimator_on_experiment(estimator, experiment: MultitaskExperimentData):
    splits = experiment.splits
    estimator.fit(
        experiment.X_train,
        experiment.y_train.to_numpy(),
        validation_data=(splits.X_val, splits.y_val),
        compound_y=None if experiment.compound_train is None else experiment.compound_train.to_numpy(),
        concentration_y=None if experiment.concentration_train is None else experiment.concentration_train.to_numpy(),
        validation_compound_y=splits.compound_val,
        validation_concentration_y=splits.concentration_val,
    )
    return estimator


def display_experiment_summary(experiment: MultitaskExperimentData, *, top_n: int = 20) -> None:
    splits = experiment.splits
    summary_df = pd.DataFrame(
        [
            {"split": "train_augmented", "n_samples": int(len(experiment.X_train))},
            {"split": "train_base", "n_samples": int(len(splits.X_train_base))},
            {"split": "val", "n_samples": int(len(splits.X_val))},
            {"split": "holdout", "n_samples": int(len(splits.X_holdout))},
        ]
    )
    display(summary_df)
    if experiment.train_metadata is not None:
        display(
            experiment.train_metadata[["mechanism_of_action", "compound", "concentration_band"]]
            .value_counts()
            .rename("n_samples")
            .reset_index()
            .head(top_n)
        )


def display_holdout_evaluation(
    estimator,
    experiment: MultitaskExperimentData,
) -> MultitaskEvaluationResult:
    predictions = estimator.predict(experiment.splits.X_holdout)
    probabilities = estimator.predict_proba(experiment.splits.X_holdout)
    reports = evaluate_multitask_estimator(
        estimator,
        experiment.splits.X_holdout,
        experiment.y_true_holdout,
        label_maps=experiment.label_maps,
        class_labels=experiment.class_labels,
    )
    display_multitask_reports_and_confusions(
        reports,
        y_true=experiment.y_true_holdout,
        y_pred=predictions,
        class_labels=experiment.class_labels,
        label_maps=experiment.label_maps,
        title_suffix=" (including control)",
    )
    (
        reports_excluding_control,
        y_true_excluding_control,
        predictions_excluding_control,
        probabilities_excluding_control,
    ) = build_reports_excluding_control(
        y_true=experiment.y_true_holdout,
        probabilities=probabilities,
        class_labels=experiment.class_labels,
        label_maps=experiment.label_maps,
    )
    display_multitask_reports_and_confusions(
        reports_excluding_control,
        y_true=y_true_excluding_control,
        y_pred=predictions_excluding_control,
        class_labels={
            target: [label for label in labels if int(label) != 0]
            for target, labels in experiment.class_labels.items()
            if target in reports_excluding_control
        },
        label_maps={
            target: {label: name for label, name in label_map.items() if int(label) != 0}
            for target, label_map in experiment.label_maps.items()
            if target in reports_excluding_control
        },
        title_suffix=" (excluding control; probabilities renormalized)",
    )
    return MultitaskEvaluationResult(
        predictions=predictions,
        probabilities=probabilities,
        reports=reports,
        reports_excluding_control=reports_excluding_control,
        predictions_excluding_control=predictions_excluding_control,
        probabilities_excluding_control=probabilities_excluding_control,
        y_true_excluding_control=y_true_excluding_control,
    )


def plot_holdout_embedding_projection(
    estimator,
    experiment: MultitaskExperimentData,
    *,
    target: str = "action",
    title: str = "Holdout embedding projection",
) -> pd.DataFrame:
    return plot_embedding_projection(
        estimator.transform(experiment.splits.X_holdout),
        experiment.y_true_holdout[target],
        experiment.label_maps[target],
        title=title,
    )


def plot_holdout_branch_embedding_projections(
    estimator,
    experiment: MultitaskExperimentData,
    *,
    target: str = "action",
) -> dict[str, pd.DataFrame]:
    branch_embeddings = estimator.transform_branches(experiment.splits.X_holdout)
    projections: dict[str, pd.DataFrame] = {}
    for key in ["st_embedding", "ts_embedding", "embedding"]:
        projections[key] = plot_embedding_projection(
            branch_embeddings[key],
            experiment.y_true_holdout[target],
            experiment.label_maps[target],
            title=f"Holdout {key} projection by {target}",
        )
    return projections


def persist_experiment_artifacts(
    *,
    output_dir: str | Path,
    estimator,
    reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: dict[str, Any],
    experiment_prefix: str | None = None,
    experiment_id: str | None = None,
    evaluation: MultitaskEvaluationResult | None = None,
    experiment: MultitaskExperimentData | None = None,
    analysis: str | None = None,
    next_round_proposal: str | None = None,
    logbook_path: str | Path | None = "EXPERIMENTS_LOGBOOK.md",
    loss_plot_dirs: list[str | Path] | None = None,
) -> ExperimentArtifacts:
    output_path, resolved_experiment_id, paths = _artifact_run_paths(
        output_dir,
        experiment_prefix=experiment_prefix,
        experiment_id=experiment_id,
    )
    per_class_dir = paths["per_class_dir"]
    confusion_dir = paths["confusion_dir"]
    predictions_dir = paths["predictions_dir"]
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    per_class_dir.mkdir(parents=True, exist_ok=True)

    config_path = paths["config"]
    history_path = paths["history"]
    summary_path = paths["summary"]
    checkpoint_path = paths["checkpoint"]

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_compatible(config), handle, indent=2, sort_keys=True)

    estimator.history_.to_csv(history_path, index=False)
    for target, (per_class_df, summary_df) in reports.items():
        prefix = f"{resolved_experiment_id}_" if resolved_experiment_id else ""
        per_class_df.to_csv(per_class_dir / f"{prefix}{target}_per_class.csv")
    _summary_frames_from_reports(reports).to_csv(summary_path, index=False)

    if evaluation is not None and experiment is not None:
        _save_multitask_confusion_matrices(
            confusion_dir=confusion_dir,
            experiment_id=resolved_experiment_id,
            evaluation=evaluation,
            experiment=experiment,
        )
        _save_multitask_prediction_tables(
            predictions_dir=predictions_dir,
            experiment_id=resolved_experiment_id,
            evaluation=evaluation,
            experiment=experiment,
        )

    copied_loss_pdfs = _copy_pdf_artifacts(
        source_dirs=loss_plot_dirs,
        destination_dir=output_path / "loss_plots",
        experiment_id=resolved_experiment_id,
    )
    torch.save(estimator.model_.state_dict(), checkpoint_path)

    logbook_output_path = _append_experiments_logbook_entry(
        experiment_id=resolved_experiment_id or Path(output_path).name,
        experiment_kind="fine_tuning",
        artifact_dir=output_path,
        key_paths={
            "config": config_path,
            "history": history_path,
            "summary_metrics": summary_path,
            "checkpoint": checkpoint_path,
            "per_class_reports": per_class_dir,
            "confusion_matrices": confusion_dir if evaluation is not None and experiment is not None else None,
            "predictions": predictions_dir if evaluation is not None and experiment is not None else None,
            "loss_pdfs": output_path / "loss_plots" if copied_loss_pdfs else None,
            "latest_loss_pdfs": _latest_loss_pdf_paths(output_path / "loss_plots") or None,
            "figures": figure_dir if figure_dir.exists() else None,
            "figure_pdfs": _figure_pdf_paths(figure_dir) or None,
        },
        config=config,
        analysis=analysis,
        next_round_proposal=next_round_proposal,
        logbook_path=logbook_path if resolved_experiment_id else None,
    )

    return ExperimentArtifacts(
        output_dir=str(output_path),
        config_path=str(config_path),
        history_path=str(history_path),
        summary_metrics_path=str(summary_path),
        per_class_dir=str(per_class_dir),
        checkpoint_path=str(checkpoint_path),
        experiment_id=resolved_experiment_id,
        confusion_dir=str(confusion_dir) if evaluation is not None and experiment is not None else None,
        predictions_dir=str(predictions_dir) if evaluation is not None and experiment is not None else None,
        logbook_path=logbook_output_path,
    )


def _build_pretraining_summary(estimator) -> pd.DataFrame:
    history_df = pd.DataFrame(getattr(estimator, "pretrain_history_", pd.DataFrame()))
    rows: list[dict[str, object]] = []
    if hasattr(estimator, "pretrain_best_epoch_"):
        rows.append({"metric": "best_epoch", "value": int(estimator.pretrain_best_epoch_)})
    if hasattr(estimator, "pretrain_best_metric_"):
        rows.append({"metric": "best_metric", "value": float(estimator.pretrain_best_metric_)})
    if not history_df.empty:
        rows.append({"metric": "n_epochs_recorded", "value": int(len(history_df))})
        for column in history_df.columns:
            if column == "epoch" or not pd.api.types.is_numeric_dtype(history_df[column]):
                continue
            values = history_df[column].dropna()
            if values.empty:
                continue
            final_value = float(values.iloc[-1])
            min_index = int(values.idxmin())
            rows.append({"metric": f"final_{column}", "value": final_value})
            rows.append({"metric": f"min_{column}", "value": float(values.loc[min_index])})
            rows.append({"metric": f"min_{column}_epoch", "value": int(history_df.loc[min_index, "epoch"])})
    return pd.DataFrame(rows, columns=["metric", "value"])


def persist_pretraining_artifacts(
    *,
    output_dir: str | Path,
    estimator,
    config: dict[str, Any] | Any,
    experiment_prefix: str,
    experiment_id: str | None = None,
    pretrained_encoder_path: str | Path | None = None,
    analysis: str | None = None,
    next_round_proposal: str | None = None,
    logbook_path: str | Path | None = "EXPERIMENTS_LOGBOOK.md",
    loss_plot_dirs: list[str | Path] | None = None,
) -> ExperimentArtifacts:
    output_path, resolved_experiment_id, paths = _artifact_run_paths(
        output_dir,
        experiment_prefix=experiment_prefix,
        experiment_id=experiment_id,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = paths["config"]
    history_path = paths["history"]
    summary_path = paths["summary"]
    checkpoint_path = output_path / f"{resolved_experiment_id}_encoder_state.pt"

    config_payload = _to_json_compatible(config)
    if pretrained_encoder_path is not None:
        config_payload = dict(config_payload)
        config_payload["pretrained_encoder_path"] = str(pretrained_encoder_path)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, sort_keys=True)

    history_df = pd.DataFrame(getattr(estimator, "pretrain_history_", pd.DataFrame()))
    history_df.to_csv(history_path, index=False)
    _build_pretraining_summary(estimator).to_csv(summary_path, index=False)

    if hasattr(estimator, "save_pretrained_encoder"):
        estimator.save_pretrained_encoder(checkpoint_path)
    elif pretrained_encoder_path is not None:
        shutil.copy2(pretrained_encoder_path, checkpoint_path)

    copied_loss_pdfs = _copy_pdf_artifacts(
        source_dirs=loss_plot_dirs,
        destination_dir=output_path / "loss_plots",
        experiment_id=resolved_experiment_id,
    )

    logbook_output_path = _append_experiments_logbook_entry(
        experiment_id=resolved_experiment_id or Path(output_path).name,
        experiment_kind="pretraining",
        artifact_dir=output_path,
        key_paths={
            "config": config_path,
            "history": history_path,
            "summary_metrics": summary_path,
            "checkpoint": checkpoint_path,
            "loss_pdfs": output_path / "loss_plots" if copied_loss_pdfs else None,
            "latest_loss_pdfs": _latest_loss_pdf_paths(output_path / "loss_plots") or None,
            "latest_encoder_pointer": pretrained_encoder_path,
        },
        config=config_payload,
        analysis=analysis,
        next_round_proposal=next_round_proposal,
        logbook_path=logbook_path,
    )

    return ExperimentArtifacts(
        output_dir=str(output_path),
        config_path=str(config_path),
        history_path=str(history_path),
        summary_metrics_path=str(summary_path),
        per_class_dir="",
        checkpoint_path=str(checkpoint_path),
        experiment_id=resolved_experiment_id,
        logbook_path=logbook_output_path,
    )
