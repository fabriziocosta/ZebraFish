from __future__ import annotations

from copy import deepcopy
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import _ensure_tensor_5d
from src.training.losses import commutative_consistency_loss
from src.training.loop import _format_eta


def _compute_commutative_pretraining_loss(estimator, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    consistency = commutative_consistency_loss(
        outputs["st_prototypes"],
        outputs["ts_prototypes"],
        temperature=float(estimator.prototype_temperature),
    )
    feature_alignment = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
    total_loss = (
        float(estimator.consistency_weight) * consistency
        + float(estimator.feature_weight) * feature_alignment
    )
    return total_loss, {
        "commutative_consistency_loss": float(consistency.item()),
        "feature_alignment_loss": float(feature_alignment.item()),
    }


def _pretrain_commutative_estimator(
    estimator,
    X: torch.Tensor,
    *,
    validation_data: torch.Tensor | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
):
    X_train = _ensure_tensor_5d(X)
    X_val = _ensure_tensor_5d(validation_data) if validation_data is not None else None

    estimator._standardize_fit(X_train)
    X_train = estimator._standardize_apply(X_train)
    if X_val is not None:
        X_val = estimator._standardize_apply(X_val)

    estimator.classes_ = torch.tensor([0]).numpy()
    estimator.compound_classes_ = None
    estimator.concentration_classes_ = None
    estimator.model_ = estimator._build_model(num_classes=1)
    estimator.device_ = estimator._device()
    estimator.input_shape_ = tuple(int(size) for size in X_train.shape[1:])
    estimator.model_.to(estimator.device_)

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=int(batch_size or estimator.batch_size),
        shuffle=True,
    )
    val_loader = (
        DataLoader(
            TensorDataset(X_val),
            batch_size=int(batch_size or estimator.batch_size),
            shuffle=False,
        )
        if X_val is not None
        else None
    )

    optimizer = torch.optim.Adam(
        estimator.model_.parameters(),
        lr=float(learning_rate or estimator.learning_rate),
        weight_decay=float(weight_decay if weight_decay is not None else estimator.weight_decay),
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

    n_epochs = int(epochs or estimator.epochs)
    history_rows: list[dict[str, float | int]] = []
    best_state = deepcopy(estimator.model_.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    if estimator.verbose:
        print("cols:\n    ep=epoch\n    lr=learning_rate\n    eta=estimated_time_remaining\n    trL=train_loss\n    trCC=train_commutative_consistency_loss\n    trFA=train_feature_alignment_loss")
        if val_loader is None:
            print(f"{'ep':>7} {'lr':>8} {'eta':>9} | {'trL':>8} {'trCC':>8} {'trFA':>8}")
        else:
            print(
                f"{'ep':>7} {'lr':>8} {'eta':>9} | {'trL':>8} {'trCC':>8} {'trFA':>8} | "
                f"{'vaL':>8} {'vaCC':>8} {'vaFA':>8}"
            )

    for epoch in range(1, n_epochs + 1):
        estimator.model_.train()
        train_loss_sum = 0.0
        train_component_sums: dict[str, float] = {}
        train_count = 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(estimator.device_, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = estimator.model_(X_batch)
            loss, components = _compute_commutative_pretraining_loss(estimator, outputs)
            loss.backward()
            optimizer.step()

            batch_size_value = int(X_batch.shape[0])
            train_loss_sum += float(loss.item()) * batch_size_value
            for key, value in components.items():
                train_component_sums[key] = train_component_sums.get(key, 0.0) + value * batch_size_value
            train_count += batch_size_value

        row: dict[str, float | int] = {"epoch": epoch, "train_loss": train_loss_sum / max(train_count, 1)}
        for key, value in train_component_sums.items():
            row[f"train_{key}"] = value / max(train_count, 1)

        if val_loader is not None:
            estimator.model_.eval()
            val_loss_sum = 0.0
            val_component_sums: dict[str, float] = {}
            val_count = 0
            with torch.no_grad():
                for (X_batch,) in val_loader:
                    X_batch = X_batch.to(estimator.device_, non_blocking=True)
                    outputs = estimator.model_(X_batch)
                    loss, components = _compute_commutative_pretraining_loss(estimator, outputs)
                    batch_size_value = int(X_batch.shape[0])
                    val_loss_sum += float(loss.item()) * batch_size_value
                    for key, value in components.items():
                        val_component_sums[key] = val_component_sums.get(key, 0.0) + value * batch_size_value
                    val_count += batch_size_value
            row["val_loss"] = val_loss_sum / max(val_count, 1)
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
            eta = _format_eta(avg_epoch_seconds * (n_epochs - epoch))
            current_lr = optimizer.param_groups[0]["lr"]
            train_parts = (
                f"{float(row['train_loss']):8.4f} "
                f"{float(row.get('train_commutative_consistency_loss', 0.0)):8.4f} "
                f"{float(row.get('train_feature_alignment_loss', 0.0)):8.4f}"
            )
            if "val_loss" in row:
                val_parts = (
                    f"{float(row['val_loss']):8.4f} "
                    f"{float(row.get('val_commutative_consistency_loss', 0.0)):8.4f} "
                    f"{float(row.get('val_feature_alignment_loss', 0.0)):8.4f}"
                )
                print(f"{epoch:03d}/{n_epochs:03d} {current_lr:8.2e} {eta:>9} | {train_parts} | {val_parts}")
            else:
                print(f"{epoch:03d}/{n_epochs:03d} {current_lr:8.2e} {eta:>9} | {train_parts}")

        if estimator.early_stopping_patience is not None and epochs_without_improvement >= estimator.early_stopping_patience:
            if estimator.verbose:
                print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} best_metric={best_metric:.4f}")
            break

    estimator.model_.load_state_dict(best_state)
    estimator.model_.eval()
    estimator.pretrain_history_ = pd.DataFrame(history_rows)
    estimator.pretrain_best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
    estimator.pretrain_best_metric_ = float(best_metric)
    estimator.pretrained_encoder_state_dict_ = estimator._extract_transfer_state_dict(estimator.model_)
    return estimator
