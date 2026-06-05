from __future__ import annotations

from copy import deepcopy
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import _ensure_tensor_5d
from src.models.probes import PROBE_TYPES, build_probe_masks, build_probe_targets, masked_probe_loss
from src.training.loop import _format_eta


def _probe_alpha_weights(estimator) -> dict[str, float]:
    return {
        "local": float(getattr(estimator, "probe_alpha_local", 1.0)),
        "region_time": float(getattr(estimator, "probe_alpha_region_time", 1.0)),
        "derivative": float(getattr(estimator, "probe_alpha_derivative", 1.0)),
        "frequency": float(getattr(estimator, "probe_alpha_frequency", 1.0)),
        "correlation": float(getattr(estimator, "probe_alpha_correlation", 1.0)),
    }


def _cross_weight_for_epoch(estimator, epoch: int) -> float:
    target = float(getattr(estimator, "lambda_cross", 1.0))
    warmup_epochs = int(getattr(estimator, "cross_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return target
    return target * min(max((int(epoch) - 1) / warmup_epochs, 0.0), 1.0)


def _compute_commutative_pretraining_loss(
    estimator,
    X: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    *,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    probe_targets = build_probe_targets(X, getattr(estimator.model_, "probe_spec"))
    probe_masks = build_probe_masks(
        probe_targets,
        observe_probability=float(getattr(estimator, "probe_mask_probability", 0.25)),
    )

    alpha_weights = _probe_alpha_weights(estimator)
    loss_A_self, per_A_self = masked_probe_loss(
        outputs["pred_A_self"],
        probe_targets,
        probe_masks,
        alpha_weights=alpha_weights,
    )
    loss_B_self, per_B_self = masked_probe_loss(
        outputs["pred_B_self"],
        probe_targets,
        probe_masks,
        alpha_weights=alpha_weights,
    )
    self_loss = loss_A_self + loss_B_self

    teacher_student_warmup_epochs = int(getattr(estimator, "teacher_student_warmup_epochs", 0))
    if teacher_student_warmup_epochs > 0 and int(epoch) <= teacher_student_warmup_epochs:
        cross_targets_A_to_B = {key: value.detach() for key, value in outputs["pred_B_self"].items()}
        cross_targets_B_to_A = {key: value.detach() for key, value in outputs["pred_A_self"].items()}
    else:
        cross_targets_A_to_B = probe_targets
        cross_targets_B_to_A = probe_targets

    loss_A_to_B, per_A_to_B = masked_probe_loss(
        outputs["pred_A_to_B"],
        cross_targets_A_to_B,
        probe_masks,
        alpha_weights=alpha_weights,
    )
    loss_B_to_A, per_B_to_A = masked_probe_loss(
        outputs["pred_B_to_A"],
        cross_targets_B_to_A,
        probe_masks,
        alpha_weights=alpha_weights,
    )
    cross_loss = loss_A_to_B + loss_B_to_A
    lambda_cross = _cross_weight_for_epoch(estimator, epoch)

    alignment_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
    total_loss = self_loss + lambda_cross * cross_loss + float(getattr(estimator, "lambda_align", 0.0)) * alignment_loss

    components = {
        "self_probe_loss": float(self_loss.item()),
        "cross_probe_loss": float(cross_loss.item()),
        "lambda_cross": float(lambda_cross),
        "feature_alignment_loss": float(alignment_loss.item()),
    }
    for probe_type in PROBE_TYPES:
        components[f"self_probe_{probe_type}_loss"] = float((per_A_self[probe_type] + per_B_self[probe_type]).item())
        components[f"cross_probe_{probe_type}_loss"] = float((per_A_to_B[probe_type] + per_B_to_A[probe_type]).item())
    return total_loss, components


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
        print(
            "cols:\n"
            "    ep=epoch\n"
            "    lr=learning_rate\n"
            "    eta=estimated_time_remaining\n"
            "    trL=train_loss\n"
            "    trS=train_self_probe_loss\n"
            "    trX=train_cross_probe_loss\n"
            "    trFA=train_feature_alignment_loss"
        )
        if val_loader is None:
            print(f"{'ep':>7} {'lr':>8} {'eta':>9} | {'trL':>8} {'trS':>8} {'trX':>8} {'trFA':>8}")
        else:
            print(
                f"{'ep':>7} {'lr':>8} {'eta':>9} | {'trL':>8} {'trS':>8} {'trX':>8} {'trFA':>8} | "
                f"{'vaL':>8} {'vaS':>8} {'vaX':>8} {'vaFA':>8}"
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
            loss, components = _compute_commutative_pretraining_loss(estimator, X_batch, outputs, epoch=epoch)
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
                    loss, components = _compute_commutative_pretraining_loss(estimator, X_batch, outputs, epoch=epoch)
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
                f"{float(row.get('train_self_probe_loss', 0.0)):8.4f} "
                f"{float(row.get('train_cross_probe_loss', 0.0)):8.4f} "
                f"{float(row.get('train_feature_alignment_loss', 0.0)):8.4f}"
            )
            if "val_loss" in row:
                val_parts = (
                    f"{float(row['val_loss']):8.4f} "
                    f"{float(row.get('val_self_probe_loss', 0.0)):8.4f} "
                    f"{float(row.get('val_cross_probe_loss', 0.0)):8.4f} "
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
