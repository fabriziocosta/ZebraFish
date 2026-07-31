from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import _PreparedData, _ensure_labels_1d, _ensure_tensor_5d
from src.training.checkpointing import (
    TrainingSuspended,
    load_training_resume_checkpoint,
    save_training_resume_checkpoint,
    should_suspend_training,
)


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
        "water_vs_other_loss",
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


def _balanced_cross_entropy(labels: torch.Tensor, *, num_classes: int, device: torch.device) -> nn.CrossEntropyLoss:
    labels_cpu = labels.detach().to("cpu", dtype=torch.long)
    counts = torch.bincount(labels_cpu, minlength=num_classes).to(torch.float32)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = float(labels_cpu.numel()) / (float(num_classes) * counts[present])
    return nn.CrossEntropyLoss(weight=weights.to(device))


def _build_supervised_criteria(estimator, prepared: _PreparedData) -> nn.Module | dict[str, nn.Module]:
    class_weighting = getattr(estimator, "class_weighting", None)
    if class_weighting in {None, "none"}:
        return nn.CrossEntropyLoss()
    if class_weighting != "balanced":
        raise ValueError("class_weighting must be one of 'balanced', 'none', or None")

    criteria: dict[str, nn.Module] = {
        "action": _balanced_cross_entropy(
            prepared.y_train,
            num_classes=len(estimator.classes_),
            device=estimator.device_,
        )
    }
    if prepared.compound_train is not None and estimator.compound_classes_ is not None:
        criteria["compound"] = _balanced_cross_entropy(
            prepared.compound_train,
            num_classes=len(estimator.compound_classes_),
            device=estimator.device_,
        )
    if prepared.concentration_train is not None and estimator.concentration_classes_ is not None:
        criteria["concentration"] = _balanced_cross_entropy(
            prepared.concentration_train,
            num_classes=len(estimator.concentration_classes_),
            device=estimator.device_,
        )
    return criteria


def _loss_acronym(name: str) -> str:
    return {
        "loss": "L",
        "action_loss": "A",
        "water_vs_other_loss": "WO",
        "feature_alignment_loss": "FA",
        "compound_loss": "Co",
        "concentration_loss": "Cn",
    }.get(name, name)


def _build_epoch_log_layout(*, include_val: bool) -> tuple[list[tuple[str, str, str]], str, str]:
    ordered_names = [
        "loss",
        "action_loss",
        "water_vs_other_loss",
        "feature_alignment_loss",
        "compound_loss",
        "concentration_loss",
    ]
    run_columns: list[tuple[str, str, str]] = [
        ("epoch", "ep", "epoch"),
        ("lr", "lr", "learning_rate"),
        ("eta", "eta", "estimated_time_remaining"),
    ]
    train_columns = [(f"train_{name}", f"tr{_loss_acronym(name)}", f"train_{name}") for name in ordered_names]
    val_columns = [(f"val_{name}", f"va{_loss_acronym(name)}", f"val_{name}") for name in ordered_names] if include_val else []
    columns = run_columns + train_columns + val_columns
    legend_items: list[tuple[str, str]] = [
        ("ep", "epoch"),
        ("lr", "learning_rate"),
        ("eta", "estimated_time_remaining"),
        ("trL", "train_loss"),
        ("trA", "train_action_loss"),
        ("trWO", "train_water_vs_other_loss"),
        ("trFA", "train_feature_alignment_loss"),
        ("trCo", "train_compound_loss"),
        ("trCn", "train_concentration_loss"),
    ]
    if include_val:
        legend_items.extend(
            [
                ("vaL", "val_loss"),
                ("vaA", "val_action_loss"),
                ("vaWO", "val_water_vs_other_loss"),
                ("vaFA", "val_feature_alignment_loss"),
                ("vaCo", "val_compound_loss"),
                ("vaCn", "val_concentration_loss"),
            ]
        )
    legend = "cols:\n" + "\n".join(f"    {acronym}={description}" for acronym, description in legend_items)
    sections = [
        [acronym for _, acronym, _ in run_columns],
        [acronym for _, acronym, _ in train_columns],
    ]
    if include_val:
        sections.append([acronym for _, acronym, _ in val_columns])
    header_parts: list[str] = []
    for section_index, section in enumerate(sections):
        if section_index > 0:
            header_parts.append("|")
        for acronym in section:
            header_parts.append(
                f"{acronym:>8}" if acronym not in {"ep", "eta"} else (f"{acronym:>9}" if acronym == "eta" else f"{acronym:>7}")
            )
    header = " ".join(header_parts)
    return columns, legend, header


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


def _format_epoch_log_row(
    row: dict[str, float | int | str],
    *,
    epochs: int,
    current_lr: float,
    eta: str,
    include_val: bool,
) -> str:
    columns, _, _ = _build_epoch_log_layout(include_val=include_val)
    run_keys = {"epoch", "lr", "eta"}
    train_keys = {
        "train_loss",
        "train_action_loss",
        "train_water_vs_other_loss",
        "train_feature_alignment_loss",
        "train_compound_loss",
        "train_concentration_loss",
    }
    values: dict[str, float | int | str] = dict(row)
    values["lr"] = current_lr
    values["eta"] = eta
    run_parts: list[str] = []
    train_parts: list[str] = []
    val_parts: list[str] = []
    for key, acronym, _ in columns:
        if key == "epoch":
            rendered = f"{int(values[key]):03d}/{epochs:03d}".rjust(7)
        elif key == "eta":
            rendered = str(values[key]).rjust(9)
        elif key == "lr":
            rendered = f"{float(values[key]):8.2e}"
        else:
            value = values.get(key)
            rendered = f"{float(value):8.4f}" if value is not None else f"{'-':>8}"
        if key in run_keys:
            run_parts.append(rendered)
        elif key in train_keys:
            train_parts.append(rendered)
        else:
            val_parts.append(rendered)
    parts = [" ".join(run_parts), "|", " ".join(train_parts)]
    if include_val:
        parts.extend(["|", " ".join(val_parts)])
    return " ".join(parts)


def _load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    current_state = model.state_dict()
    compatible_state = {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    }
    skipped_keys = sorted(key for key in state_dict if key not in compatible_state)
    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)
    return sorted(compatible_state), skipped_keys


def _maybe_save_training_history_pdfs(
    estimator,
    history_rows: list[dict[str, float | int]],
    epoch: int,
) -> None:
    plot_dir = getattr(estimator, "training_plot_dir", None)
    if not plot_dir:
        return
    every_n_epochs = max(1, int(getattr(estimator, "training_plot_every_n_epochs", 1) or 1))
    if int(epoch) % every_n_epochs != 0:
        return

    from pathlib import Path

    from src.training.reporting import save_training_history_pdf

    output_dir = Path(str(plot_dir)).expanduser()
    title = str(getattr(estimator, "training_plot_title", "Training history"))
    latest_path = output_dir / "latest.loss-curves.pdf"
    history_df = pd.DataFrame(history_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_dir / "latest.history.csv", index=False)
    excluded_loss_names = []
    for loss_name, weight_name in [
        ("action_loss", "action_weight"),
        ("water_vs_other_loss", "water_vs_other_weight"),
        ("compound_loss", "compound_weight"),
        ("concentration_loss", "concentration_weight"),
    ]:
        if float(getattr(estimator, weight_name, 1.0) or 0.0) == 0.0:
            excluded_loss_names.append(loss_name)
    if float(getattr(estimator, "latent_alignment_weight", 0.0) or getattr(estimator, "lambda_align", 0.0) or 0.0) == 0.0:
        excluded_loss_names.append("feature_alignment_loss")
    save_training_history_pdf(history_df, latest_path, title=title, excluded_loss_names=excluded_loss_names)


def _maybe_save_live_best_checkpoint(
    estimator,
    state_dict: dict[str, torch.Tensor],
    *,
    epoch: int,
    metric: float,
    monitor_key: str,
) -> None:
    checkpoint_path = getattr(estimator, "live_checkpoint_path", None)
    if not checkpoint_path:
        return

    output_path = Path(str(checkpoint_path)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
        "best_epoch": int(epoch),
        "best_metric": float(metric),
        "monitor_key": str(monitor_key),
    }
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(output_path)
    estimator.live_checkpoint_path_ = str(output_path)


def _maybe_run_epoch_evaluation_callback(estimator, *, epoch: int, row: dict[str, object]) -> None:
    callback = getattr(estimator, "epoch_evaluation_callback", None)
    if not callable(callback):
        return
    every = max(1, int(getattr(estimator, "epoch_evaluation_every_n_epochs", 1) or 1))
    minimum = max(1, int(getattr(estimator, "epoch_evaluation_minimum_epoch", 1) or 1))
    if int(epoch) < minimum or int(epoch) % every != 0:
        return
    try:
        callback(estimator=estimator, epoch=int(epoch), history_row=dict(row))
    except Exception as exc:  # diagnostics must not invalidate otherwise healthy training
        errors = list(getattr(estimator, "epoch_evaluation_errors_", []))
        errors.append({"epoch": int(epoch), "error": f"{type(exc).__name__}: {exc}"})
        estimator.epoch_evaluation_errors_ = errors[-20:]
        if getattr(estimator, "verbose", False):
            print(
                f"live domain diagnostic failed at epoch={int(epoch):03d}: {type(exc).__name__}: {exc}",
                flush=True,
            )


def _fit_multitask_estimator(estimator, prepared: _PreparedData):
    hot_start_state = None
    if getattr(estimator, "hot_start", False) and hasattr(estimator, "model_"):
        hot_start_state = {
            key: value.detach().cpu()
            for key, value in estimator.model_.state_dict().items()
        }

    estimator.model_ = estimator._build_model_from_prepared(prepared)
    estimator.device_ = estimator._device()
    estimator.model_.to(estimator.device_)
    estimator.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])
    if hasattr(estimator, "_load_pretrained_weights_into_model"):
        estimator._load_pretrained_weights_into_model(estimator.model_)
    if hot_start_state is not None:
        loaded_keys, skipped_keys = _load_compatible_state_dict(estimator.model_, hot_start_state)
        estimator.hot_start_loaded_keys_ = loaded_keys
        estimator.hot_start_skipped_keys_ = skipped_keys
    elif getattr(estimator, "hot_start", False):
        estimator.hot_start_loaded_keys_ = []
        estimator.hot_start_skipped_keys_ = []
    if getattr(estimator, "freeze_backbone", False) and hasattr(estimator, "_set_encoder_trainable"):
        estimator._set_encoder_trainable(estimator.model_, trainable=False)

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

    criterion = _build_supervised_criteria(estimator, prepared)
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

    history_rows: list[dict[str, float | int | str]] = []
    best_state = deepcopy(estimator.model_.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    early_stopping_start_epoch = max(1, int(getattr(estimator, "early_stopping_start_epoch", None) or 1))
    start_epoch = 1
    resume_state = load_training_resume_checkpoint(
        estimator,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_stage="supervised",
    )
    if resume_state is not None:
        start_epoch = int(resume_state["start_epoch"])
        history_rows = resume_state["history_rows"]
        best_state = resume_state["best_state"]
        best_metric = float(resume_state["best_metric"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
    training_start = time.perf_counter()
    if estimator.verbose:
        _, legend, header = _build_epoch_log_layout(include_val=val_loader is not None)
        print(legend)
        print(header)
        if start_epoch > 1:
            print(f"resuming supervised training from epoch {start_epoch:03d}/{estimator.epochs:03d}")

    for epoch in range(start_epoch, estimator.epochs + 1):
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
                _maybe_save_live_best_checkpoint(
                    estimator,
                    best_state,
                    epoch=best_epoch,
                    metric=best_metric,
                    monitor_key=monitor_key,
                )
            else:
                epochs_without_improvement += 1

        if scheduler is not None and should_monitor:
            scheduler.step(monitor_metric)

        history_rows.append(row)
        resume_checkpoint_path = save_training_resume_checkpoint(
            estimator,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            history_rows=history_rows,
            best_state=best_state,
            best_metric=best_metric,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            stage="supervised",
        )
        _maybe_save_training_history_pdfs(estimator, history_rows, epoch)
        _maybe_run_epoch_evaluation_callback(estimator, epoch=epoch, row=row)
        if estimator.verbose:
            elapsed = time.perf_counter() - training_start
            completed_since_resume = max(1, epoch - start_epoch + 1)
            avg_epoch_seconds = elapsed / completed_since_resume
            eta = _format_eta(avg_epoch_seconds * (estimator.epochs - epoch))
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                _format_epoch_log_row(
                    row,
                    epochs=estimator.epochs,
                    current_lr=current_lr,
                    eta=eta,
                    include_val="val_loss" in row,
                )
            )

        if should_suspend_training(estimator):
            if estimator.verbose:
                print(f"suspend requested after epoch={epoch:03d}; resume_checkpoint={resume_checkpoint_path}", flush=True)
            if resume_checkpoint_path is None:
                raise RuntimeError("Training suspend requested but resume_checkpoint_path is not configured")
            raise TrainingSuspended(resume_checkpoint_path)

        if (
            should_monitor
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
                if not isinstance(value, torch.Tensor):
                    continue
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _predict_proba_from_estimator(estimator, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    outputs = _collect_output_batches(estimator, X)
    action_logits = torch.from_numpy(outputs["logits"])
    use_hierarchical_action = (
        "water_logits" in outputs
        and float(getattr(estimator, "water_vs_other_weight", 0.0)) > 0.0
        and action_logits.shape[1] > 2
        and 0 in getattr(estimator, "class_to_index_", {})
    )
    if use_hierarchical_action:
        water_index = int(estimator.class_to_index_[0])
        drug_indices = [index for index in range(action_logits.shape[1]) if index != water_index]
        water_proba = torch.softmax(torch.from_numpy(outputs["water_logits"]), dim=1)
        drug_action_proba = torch.softmax(action_logits[:, drug_indices], dim=1)
        action_proba = torch.zeros_like(action_logits)
        action_proba[:, water_index] = water_proba[:, 0]
        action_proba[:, drug_indices] = water_proba[:, 1:2] * drug_action_proba
        result = {"action": action_proba.numpy()}
    else:
        result = {"action": torch.softmax(action_logits, dim=1).numpy()}
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
