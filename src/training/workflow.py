from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from IPython.display import display
import pandas as pd
import torch

from src.training.data import augment_training_tensors_with_rotations, split_labeled_tensor_dataset_by_instance
from src.training.reporting import (
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


@dataclass
class ExperimentArtifacts:
    output_dir: str
    config_path: str
    history_path: str
    summary_metrics_path: str
    per_class_dir: str
    checkpoint_path: str


@dataclass
class MultitaskEvaluationResult:
    predictions: dict[str, Any]
    probabilities: dict[str, Any]
    reports: dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


def prepare_multitask_experiment_data(
    dataset: dict[str, object],
    *,
    holdout_fraction: float,
    validation_fraction_within_train: float,
    train_num_random_rotations: int = 0,
    rotation_range_degrees: float = 5.0,
    random_state: int = 0,
) -> MultitaskExperimentData:
    splits = split_labeled_tensor_dataset_by_instance(
        dataset,
        holdout_fraction=holdout_fraction,
        validation_fraction_within_train=validation_fraction_within_train,
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
        _, compound_train, _ = augment_training_tensors_with_rotations(
            splits.X_train_base,
            splits.compound_train_base,
            metadata=None,
            num_random_rotations=train_num_random_rotations,
            rotation_range_degrees=rotation_range_degrees,
            random_state=random_state,
        )
    if splits.concentration_train_base is not None:
        _, concentration_train, _ = augment_training_tensors_with_rotations(
            splits.X_train_base,
            splits.concentration_train_base,
            metadata=None,
            num_random_rotations=train_num_random_rotations,
            rotation_range_degrees=rotation_range_degrees,
            random_state=random_state,
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
    )
    return MultitaskEvaluationResult(
        predictions=predictions,
        probabilities=probabilities,
        reports=reports,
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
) -> ExperimentArtifacts:
    output_path = Path(output_dir)
    per_class_dir = output_path / "per_class_reports"
    output_path.mkdir(parents=True, exist_ok=True)
    per_class_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.json"
    history_path = output_path / "history.csv"
    summary_path = output_path / "summary_metrics.csv"
    checkpoint_path = output_path / "model_state.pt"

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_compatible(config), handle, indent=2, sort_keys=True)

    estimator.history_.to_csv(history_path, index=False)
    summary_frames: list[pd.DataFrame] = []
    for target, (per_class_df, summary_df) in reports.items():
        per_class_df.to_csv(per_class_dir / f"{target}.csv")
        target_summary = summary_df.rename_axis("metric").reset_index()
        target_summary.insert(0, "target", target)
        summary_frames.append(target_summary)
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    torch.save(estimator.model_.state_dict(), checkpoint_path)

    return ExperimentArtifacts(
        output_dir=str(output_path),
        config_path=str(config_path),
        history_path=str(history_path),
        summary_metrics_path=str(summary_path),
        per_class_dir=str(per_class_dir),
        checkpoint_path=str(checkpoint_path),
    )
