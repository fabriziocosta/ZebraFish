from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import _PreparedData, _ensure_labels_1d, _ensure_tensor_5d


def _format_eta(seconds: float) -> str:
    remaining = max(int(round(seconds)), 0)
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_loss_components_for_log(row: dict[str, float | int], *, prefix: str) -> str:
    ordered_names = [
        "loss",
        "action_loss",
        "commutative_consistency_loss",
        "feature_alignment_loss",
        "compound_loss",
        "concentration_loss",
    ]
    parts: list[str] = []
    for name in ordered_names:
        key = f"{prefix}{name}"
        if key in row:
            parts.append(f"{name}={float(row[key]):.4f}")
    return " ".join(parts)


def _fit_multitask_estimator(estimator, prepared: _PreparedData):
    estimator.model_ = estimator._build_model_from_prepared(prepared)
    estimator.device_ = estimator._device()
    estimator.model_.to(estimator.device_)
    estimator.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])

    train_tensors: list[torch.Tensor] = [prepared.X_train, prepared.y_train]
    if prepared.compound_train is not None:
        train_tensors.append(prepared.compound_train)
    if prepared.concentration_train is not None:
        train_tensors.append(prepared.concentration_train)
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=estimator.batch_size, shuffle=True)

    val_loader = None
    if prepared.X_val is not None and prepared.y_val is not None:
        val_tensors = [prepared.X_val, prepared.y_val]
        if prepared.compound_val is not None:
            val_tensors.append(prepared.compound_val)
        if prepared.concentration_val is not None:
            val_tensors.append(prepared.concentration_val)
        val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=estimator.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        estimator.model_.parameters(),
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

    history_rows: list[dict[str, float | int]] = []
    best_state = deepcopy(estimator.model_.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    for epoch in range(1, estimator.epochs + 1):
        estimator.model_.train()
        train_total_loss_sum = 0.0
        train_component_sums: dict[str, float] = {}
        train_count = 0

        for batch in train_loader:
            X_batch = batch[0].to(estimator.device_, non_blocking=True)
            y_batch = batch[1].to(estimator.device_, non_blocking=True)
            compound_batch = batch[2].to(estimator.device_, non_blocking=True) if prepared.compound_train is not None else None
            concentration_batch = (
                batch[3 if prepared.compound_train is not None else 2].to(estimator.device_, non_blocking=True)
                if prepared.concentration_train is not None
                else None
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = estimator.model_(X_batch)
            loss, loss_components = estimator._compute_losses(
                outputs,
                y_batch,
                criterion,
                compound_targets=compound_batch,
                concentration_targets=concentration_batch,
            )
            loss.backward()
            optimizer.step()

            batch_size = int(X_batch.shape[0])
            train_total_loss_sum += float(loss.item()) * batch_size
            for key, value in loss_components.items():
                train_component_sums[key] = train_component_sums.get(key, 0.0) + value * batch_size
            train_count += batch_size

        row: dict[str, float | int] = {"epoch": epoch, "train_loss": train_total_loss_sum / max(train_count, 1)}
        for key, value in train_component_sums.items():
            row[f"train_{key}"] = value / max(train_count, 1)

        if val_loader is not None:
            estimator.model_.eval()
            val_total_loss_sum = 0.0
            val_component_sums: dict[str, float] = {}
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch[0].to(estimator.device_, non_blocking=True)
                    y_batch = batch[1].to(estimator.device_, non_blocking=True)
                    compound_batch = batch[2].to(estimator.device_, non_blocking=True) if prepared.compound_val is not None else None
                    concentration_batch = (
                        batch[3 if prepared.compound_val is not None else 2].to(estimator.device_, non_blocking=True)
                        if prepared.concentration_val is not None
                        else None
                    )
                    outputs = estimator.model_(X_batch)
                    loss, loss_components = estimator._compute_losses(
                        outputs,
                        y_batch,
                        criterion,
                        compound_targets=compound_batch,
                        concentration_targets=concentration_batch,
                    )
                    batch_size = int(X_batch.shape[0])
                    val_total_loss_sum += float(loss.item()) * batch_size
                    for key, value in loss_components.items():
                        val_component_sums[key] = val_component_sums.get(key, 0.0) + value * batch_size
                    val_count += batch_size
            row["val_loss"] = val_total_loss_sum / max(val_count, 1)
            for key, value in val_component_sums.items():
                row[f"val_{key}"] = value / max(val_count, 1)
            metric = float(row["val_loss"])
        else:
            metric = float(row["train_loss"])

        improved = metric < (best_metric - float(estimator.early_stopping_min_delta))
        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = deepcopy(estimator.model_.state_dict())
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(metric)

        history_rows.append(row)
        if estimator.verbose:
            elapsed = time.perf_counter() - training_start
            avg_epoch_seconds = elapsed / epoch
            eta = _format_eta(avg_epoch_seconds * (estimator.epochs - epoch))
            current_lr = optimizer.param_groups[0]["lr"]
            if "val_loss" in row:
                train_loss_parts = _format_loss_components_for_log(row, prefix="train_")
                val_loss_parts = _format_loss_components_for_log(row, prefix="val_")
                print(
                    f"epoch {epoch:03d}/{estimator.epochs:03d} "
                    f"train[{train_loss_parts}] "
                    f"val[{val_loss_parts}] "
                    f"lr={current_lr:.2e} eta={eta}"
                )
            else:
                train_loss_parts = _format_loss_components_for_log(row, prefix="train_")
                print(
                    f"epoch {epoch:03d}/{estimator.epochs:03d} train[{train_loss_parts}] "
                    f"lr={current_lr:.2e} eta={eta}"
                )

        if estimator.early_stopping_patience is not None and epochs_without_improvement >= estimator.early_stopping_patience:
            if estimator.verbose:
                print(
                    f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_metric={best_metric:.4f}"
                )
            break

    estimator.model_.load_state_dict(best_state)
    estimator.model_.eval()
    estimator.history_ = pd.DataFrame(history_rows)
    estimator.best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
    estimator.best_metric_ = float(best_metric)
    return estimator


def _collect_output_batches(estimator, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    check_is_fitted(estimator, ["model_", "classes_", "input_mean_", "input_std_"])
    X_tensor = estimator._standardize_apply(_ensure_tensor_5d(X))
    loader = DataLoader(TensorDataset(X_tensor), batch_size=estimator.batch_size, shuffle=False)
    collected: dict[str, list[np.ndarray]] = {}
    estimator.model_.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(estimator.device_, non_blocking=True)
            outputs = estimator.model_(X_batch)
            for key, value in outputs.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _predict_proba_from_estimator(estimator, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    outputs = _collect_output_batches(estimator, X)
    result = {"action": torch.softmax(torch.from_numpy(outputs["logits"]), dim=1).numpy()}
    if "compound_logits" in outputs:
        result["compound"] = torch.softmax(torch.from_numpy(outputs["compound_logits"]), dim=1).numpy()
    if "concentration_logits" in outputs:
        result["concentration"] = torch.softmax(torch.from_numpy(outputs["concentration_logits"]), dim=1).numpy()
    return result


def _predict_from_estimator(estimator, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    probabilities = _predict_proba_from_estimator(estimator, X)
    predictions = {"action": estimator.classes_[probabilities["action"].argmax(axis=1)]}
    if "compound" in probabilities:
        predictions["compound"] = estimator.compound_classes_[probabilities["compound"].argmax(axis=1)]
    if "concentration" in probabilities:
        predictions["concentration"] = estimator.concentration_classes_[probabilities["concentration"].argmax(axis=1)]
    return predictions


def _transform_from_estimator(estimator, X: torch.Tensor | np.ndarray) -> np.ndarray:
    outputs = _collect_output_batches(estimator, X)
    return outputs["embedding"]


def _evaluate_loss_components_from_estimator(
    estimator,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray | list[int],
) -> dict[str, float]:
    check_is_fitted(estimator, ["model_", "classes_", "input_mean_", "input_std_"])
    X_tensor = estimator._standardize_apply(_ensure_tensor_5d(X))
    y_values = _ensure_labels_1d(y)
    if len(X_tensor) != len(y_values):
        raise ValueError("X and y must have the same number of samples")
    y_tensor = estimator._encode_labels(y_values)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=estimator.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss_sum = 0.0
    component_sums: dict[str, float] = {}
    count = 0
    estimator.model_.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(estimator.device_, non_blocking=True)
            y_batch = y_batch.to(estimator.device_, non_blocking=True)
            outputs = estimator.model_(X_batch)
            loss, loss_components = estimator._compute_losses(outputs, y_batch, criterion)
            batch_size = int(X_batch.shape[0])
            total_loss_sum += float(loss.item()) * batch_size
            for key, value in loss_components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
            count += batch_size
    result = {"loss": total_loss_sum / max(count, 1)}
    for key, value in component_sums.items():
        result[key] = value / max(count, 1)
    return result


def _score_from_estimator(
    estimator,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray | list[int],
) -> float:
    y_true = _ensure_labels_1d(y)
    y_pred = _predict_from_estimator(estimator, X)["action"]
    return float(accuracy_score(y_true, y_pred))
