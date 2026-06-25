from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import _ensure_tensor_5d
from src.models.probes import PROBE_TYPES, build_probe_masks, build_probe_targets, masked_probe_loss
from src.training.losses import prototype_consistency_loss
from src.training.loop import _format_eta
from src.training.reporting import plot_grouped_independent_axis_history


def _probe_alpha_weights(estimator) -> dict[str, float]:
    return {
        "local": float(getattr(estimator, "probe_alpha_local", 1.0)),
        "region_time": float(getattr(estimator, "probe_alpha_region_time", 1.0)),
        "derivative": float(getattr(estimator, "probe_alpha_derivative", 1.0)),
        "frequency": float(getattr(estimator, "probe_alpha_frequency", 1.0)),
        "correlation": float(getattr(estimator, "probe_alpha_correlation", 1.0)),
    }


def linear_ramp(epoch: int, target_weight: float, warmup_epochs: int, ramp_epochs: int) -> float:
    target = float(target_weight)
    warmup = int(warmup_epochs)
    ramp = int(ramp_epochs)
    if int(epoch) <= warmup:
        return 0.0
    if ramp <= 0:
        return target
    ramp_step = int(epoch) - warmup
    return target * min(max(ramp_step / ramp, 0.0), 1.0)


def _cross_weight_for_epoch(estimator, epoch: int) -> float:
    return linear_ramp(
        epoch=epoch,
        target_weight=float(getattr(estimator, "lambda_cross", 1.0)),
        warmup_epochs=int(getattr(estimator, "cross_warmup_epochs", 0)),
        ramp_epochs=int(getattr(estimator, "cross_ramp_epochs", 0)),
    )


def _prototype_weight_for_epoch(estimator, epoch: int) -> float:
    return linear_ramp(
        epoch=epoch,
        target_weight=float(getattr(estimator, "prototype_alignment_weight", 1.0)),
        warmup_epochs=int(getattr(estimator, "prototype_warmup_epochs", 0)),
        ramp_epochs=int(getattr(estimator, "prototype_ramp_epochs", 0)),
    )


def _latent_alignment_weight(estimator) -> float:
    weight = float(getattr(estimator, "latent_alignment_weight", 0.0))
    if weight != 0.0:
        return weight
    return float(getattr(estimator, "lambda_align", 0.0))


def _cross_probe_teacher_targets(
    outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cross_targets_A_to_B = {
        key: value.detach()
        for key, value in outputs["pred_B_self"].items()
    }
    cross_targets_B_to_A = {
        key: value.detach()
        for key, value in outputs["pred_A_self"].items()
    }
    return cross_targets_A_to_B, cross_targets_B_to_A


def _early_stopping_start_epoch_for_pretraining(estimator) -> int:
    configured_start = getattr(estimator, "early_stopping_start_epoch", None)
    if configured_start is not None:
        return max(1, int(configured_start))
    if float(getattr(estimator, "lambda_cross", 0.0)) <= 0.0:
        return 1
    warmup_epochs = max(0, int(getattr(estimator, "cross_warmup_epochs", 0)))
    ramp_epochs = max(0, int(getattr(estimator, "cross_ramp_epochs", 0)))
    return max(1, warmup_epochs + max(1, ramp_epochs // 2))


def _monitor_key_for_row(estimator, row: dict[str, float | int]) -> str:
    monitor = str(getattr(estimator, "early_stopping_monitor", "loss") or "loss")
    candidates = [monitor]
    if not monitor.startswith(("train_", "val_")):
        candidates = [f"val_{monitor}", f"train_{monitor}", monitor]
    for candidate in candidates:
        if candidate in row:
            return candidate
    if "val_loss" in row:
        return "val_loss"
    return "train_loss"


def _smoothed_monitor_value(
    estimator,
    history_rows: list[dict[str, float | int]],
    row: dict[str, float | int],
    monitor_key: str,
) -> tuple[float, float]:
    raw_value = float(row[monitor_key])
    smoothing = str(getattr(estimator, "early_stopping_smoothing", "none") or "none").lower()
    window = max(1, int(getattr(estimator, "early_stopping_smoothing_window", 1) or 1))
    if smoothing in {"none", "raw"} or window <= 1:
        return raw_value, raw_value

    values = [
        float(history_row[monitor_key])
        for history_row in history_rows
        if monitor_key in history_row and np.isfinite(float(history_row[monitor_key]))
    ]
    values.append(raw_value)
    recent = np.asarray(values[-window:], dtype=float)
    if smoothing == "median":
        return raw_value, float(np.median(recent))
    if smoothing == "mean":
        return raw_value, float(np.mean(recent))
    raise ValueError(
        "early_stopping_smoothing must be one of 'none', 'median', or 'mean', "
        f"got {smoothing!r}"
    )


def _rolling_median(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(1, int(window)), min_periods=1).median()


def _plot_loss_group(ax, history_df: pd.DataFrame, columns: list[str], *, title: str, smoothing_window: int) -> None:
    if not columns:
        ax.set_visible(False)
        return
    epochs = history_df["epoch"].to_numpy(dtype=float)
    plotted_values = []
    for column in columns:
        if column not in history_df.columns:
            continue
        values = history_df[column].astype(float)
        if not values.notna().any():
            continue
        plotted_values.append(values.to_numpy())
        (line,) = ax.plot(epochs, values.to_numpy(), linewidth=1.0, alpha=0.22, label=column)
        smoothed = _rolling_median(values, smoothing_window)
        ax.plot(epochs, smoothed.to_numpy(), linewidth=2.0, alpha=0.9, color=line.get_color())
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if plotted_values and np.nanmax(np.concatenate(plotted_values)) > 0:
        ax.set_yscale("log", nonpositive="clip")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(
        fontsize="x-small",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=True,
    )


def _save_pretraining_loss_pdf(
    history_rows: list[dict[str, float | int]],
    output_path: Path,
    *,
    smoothing_window: int,
    estimator=None,
) -> Path:
    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        return output_path

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cross_active = float(getattr(estimator, "lambda_cross", 1.0) if estimator is not None else history_df.get("train_lambda_cross", pd.Series([0.0])).max()) > 0.0
    proto_active = float(getattr(estimator, "prototype_alignment_weight", 1.0) if estimator is not None else history_df.get("train_lambda_proto", pd.Series([0.0])).max()) > 0.0
    latent_active = float(
        getattr(estimator, "latent_alignment_weight", 0.0) if estimator is not None else 0.0
    ) > 0.0 or float(getattr(estimator, "lambda_align", 0.0) if estimator is not None else 0.0) > 0.0

    groups = [
        ("Total Loss", ["train_loss", "val_loss", "monitor_metric"]),
        ("Self-Probe Loss", ["train_self_probe_loss", "val_self_probe_loss"]),
    ]
    if cross_active:
        groups.append(("Cross-Probe Loss", ["train_cross_probe_loss", "val_cross_probe_loss", "train_lambda_cross"]))
    for probe_type in PROBE_TYPES:
        groups.append(
            (
                f"{probe_type.replace('_', ' ').title()} Self-Probe Loss",
                [f"train_self_probe_{probe_type}_loss", f"val_self_probe_{probe_type}_loss"],
            )
        )
        if cross_active:
            groups.append(
                (
                    f"{probe_type.replace('_', ' ').title()} Cross-Probe Loss",
                    [f"train_cross_probe_{probe_type}_loss", f"val_cross_probe_{probe_type}_loss"],
                )
            )

    alignment_columns = []
    if proto_active:
        alignment_columns.extend(["train_prototype_alignment_loss", "val_prototype_alignment_loss", "train_lambda_proto"])
    if latent_active:
        alignment_columns.extend(["train_latent_alignment_loss", "val_latent_alignment_loss", "train_feature_alignment_loss", "val_feature_alignment_loss"])
    if alignment_columns:
        groups.append(("Active Alignment And Weights", alignment_columns))
    try:
        fig, _ = plot_grouped_independent_axis_history(
            history_df,
            groups,
            title="Commutative Pretraining Loss Curves",
            max_curves_per_panel=4,
            smoothing_window=smoothing_window,
            show_raw=True,
        )
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


def _maybe_save_pretraining_loss_pdfs(estimator, history_rows: list[dict[str, float | int]], epoch: int) -> None:
    plot_dir = getattr(estimator, "training_plot_dir", None)
    if not plot_dir:
        return
    every_n_epochs = max(1, int(getattr(estimator, "training_plot_every_n_epochs", 1) or 1))
    if int(epoch) % every_n_epochs != 0:
        return
    output_dir = Path(str(plot_dir)).expanduser()
    smoothing_window = max(1, int(getattr(estimator, "training_plot_smoothing_window", 5) or 5))
    epoch_path = output_dir / f"epoch_{int(epoch):03d}.loss-curves.pdf"
    latest_path = output_dir / "latest.loss-curves.pdf"
    history_df = pd.DataFrame(history_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_dir / f"epoch_{int(epoch):03d}.history.csv", index=False)
    history_df.to_csv(output_dir / "latest.history.csv", index=False)
    _save_pretraining_loss_pdf(history_rows, epoch_path, smoothing_window=smoothing_window, estimator=estimator)
    _save_pretraining_loss_pdf(history_rows, latest_path, smoothing_window=smoothing_window, estimator=estimator)


def _compute_commutative_pretraining_loss(
    estimator,
    X: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    *,
    epoch: int,
    full_probe_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    probe_targets = build_probe_targets(X, getattr(estimator.model_, "probe_spec"))
    probe_masks = build_probe_masks(
        probe_targets,
        observe_probability=float(getattr(estimator, "probe_mask_probability", 0.25)),
        full=full_probe_mask,
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

    cross_targets_A_to_B, cross_targets_B_to_A = _cross_probe_teacher_targets(outputs)

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

    lambda_proto = _prototype_weight_for_epoch(estimator, epoch)
    prototype_alignment_loss = prototype_consistency_loss(
        outputs["st_prototypes"],
        outputs["ts_prototypes"],
        temperature=float(getattr(estimator, "prototype_temperature", 0.1)),
    )
    latent_alignment_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
    latent_alignment_weight = _latent_alignment_weight(estimator)
    total_loss = (
        self_loss
        + lambda_cross * cross_loss
        + lambda_proto * prototype_alignment_loss
        + latent_alignment_weight * latent_alignment_loss
    )

    components = {
        "self_probe_loss": float(self_loss.item()),
        "cross_probe_loss": float(cross_loss.item()),
        "lambda_cross": float(lambda_cross),
        "lambda_proto": float(lambda_proto),
        "prototype_alignment_loss": float(prototype_alignment_loss.item()),
        "latent_alignment_loss": float(latent_alignment_loss.item()),
        "latent_alignment_weight": float(latent_alignment_weight),
        "feature_alignment_loss": float(latent_alignment_loss.item()),
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
    if hasattr(estimator, "_load_pretrained_weights_into_model"):
        estimator._load_pretrained_weights_into_model(estimator.model_)

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
    early_stopping_start_epoch = _early_stopping_start_epoch_for_pretraining(estimator)
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
            "    trP=train_prototype_alignment_loss\n"
            "    trLA=train_latent_alignment_loss"
        )
        if val_loader is None:
            print(
                f"{'ep':>7} {'lr':>8} {'eta':>9} | "
                f"{'trL':>8} {'trS':>8} {'trX':>8} {'trP':>8} {'trLA':>8}"
            )
        else:
            print(
                f"{'ep':>7} {'lr':>8} {'eta':>9} | "
                f"{'trL':>8} {'trS':>8} {'trX':>8} {'trP':>8} {'trLA':>8} | "
                f"{'vaL':>8} {'vaS':>8} {'vaX':>8} {'vaP':>8} {'vaLA':>8}"
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
                    loss, components = _compute_commutative_pretraining_loss(
                        estimator,
                        X_batch,
                        outputs,
                        epoch=epoch,
                        full_probe_mask=True,
                    )
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

        monitor_key = _monitor_key_for_row(estimator, row)
        monitor_raw, monitor_metric = _smoothed_monitor_value(estimator, history_rows, row, monitor_key)
        row["monitor_metric_raw"] = monitor_raw
        row["monitor_metric"] = monitor_metric
        row["monitor_key"] = monitor_key

        should_monitor = epoch >= early_stopping_start_epoch
        improved = False
        if should_monitor:
            improved = monitor_metric < (best_metric - float(estimator.early_stopping_min_delta))
            if improved:
                best_metric = monitor_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = deepcopy(estimator.model_.state_dict())
            else:
                epochs_without_improvement += 1

        if scheduler is not None and should_monitor:
            scheduler.step(monitor_metric)

        history_rows.append(row)
        _maybe_save_pretraining_loss_pdfs(estimator, history_rows, epoch)
        if estimator.verbose:
            elapsed = time.perf_counter() - training_start
            avg_epoch_seconds = elapsed / epoch
            eta = _format_eta(avg_epoch_seconds * (n_epochs - epoch))
            current_lr = optimizer.param_groups[0]["lr"]
            train_parts = (
                f"{float(row['train_loss']):8.4f} "
                f"{float(row.get('train_self_probe_loss', 0.0)):8.4f} "
                f"{float(row.get('train_cross_probe_loss', 0.0)):8.4f} "
                f"{float(row.get('train_prototype_alignment_loss', 0.0)):8.4f} "
                f"{float(row.get('train_latent_alignment_loss', 0.0)):8.4f}"
            )
            if "val_loss" in row:
                val_parts = (
                    f"{float(row['val_loss']):8.4f} "
                    f"{float(row.get('val_self_probe_loss', 0.0)):8.4f} "
                    f"{float(row.get('val_cross_probe_loss', 0.0)):8.4f} "
                    f"{float(row.get('val_prototype_alignment_loss', 0.0)):8.4f} "
                    f"{float(row.get('val_latent_alignment_loss', 0.0)):8.4f}"
                )
                print(f"{epoch:03d}/{n_epochs:03d} {current_lr:8.2e} {eta:>9} | {train_parts} | {val_parts}")
            else:
                print(f"{epoch:03d}/{n_epochs:03d} {current_lr:8.2e} {eta:>9} | {train_parts}")

        if (
            should_monitor
            and estimator.early_stopping_patience is not None
            and epochs_without_improvement >= estimator.early_stopping_patience
        ):
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
