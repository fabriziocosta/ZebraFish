from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import signal
from typing import Any

import torch

_SUSPEND_REQUESTED = False


class TrainingSuspended(RuntimeError):
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        super().__init__(f"Training suspended after saving resume checkpoint: {self.checkpoint_path}")


def _path_attr(estimator: Any, name: str) -> Path | None:
    value = getattr(estimator, name, None)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def resume_checkpoint_path(estimator: Any) -> Path | None:
    return _path_attr(estimator, "resume_checkpoint_path")


def suspend_marker_path(estimator: Any) -> Path | None:
    return _path_attr(estimator, "suspend_marker_path")


def should_suspend_training(estimator: Any) -> bool:
    marker_path = suspend_marker_path(estimator)
    return _SUSPEND_REQUESTED or (marker_path is not None and marker_path.exists())


def install_training_signal_handlers(estimator: Any) -> None:
    marker_path = suspend_marker_path(estimator)

    def _request_suspend(signum: int, _frame: Any) -> None:
        global _SUSPEND_REQUESTED
        _SUSPEND_REQUESTED = True
        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(f"signal={signal.Signals(signum).name}\n", encoding="utf-8")

    signal.signal(signal.SIGINT, _request_suspend)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_suspend)


def save_training_resume_checkpoint(
    estimator: Any,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    history_rows: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor],
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    checkpoint_path = resume_checkpoint_path(estimator)
    if checkpoint_path is None:
        return None

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "zebrafish_training_resume_checkpoint_v1",
        "stage": stage,
        "epoch": int(epoch),
        "history_rows": list(history_rows),
        "model_state_dict": {key: value.detach().cpu() for key, value in estimator.model_.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "extra": dict(extra or {}),
    }
    temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(checkpoint_path)
    estimator.resume_checkpoint_path_ = str(checkpoint_path)
    estimator.resume_checkpoint_epoch_ = int(epoch)
    return checkpoint_path


def load_training_resume_checkpoint(
    estimator: Any,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_stage: str,
) -> dict[str, Any] | None:
    checkpoint_path = resume_checkpoint_path(estimator)
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Resume checkpoint must contain a mapping: {checkpoint_path}")
    stage = payload.get("stage")
    if stage != expected_stage:
        raise ValueError(f"Resume checkpoint stage mismatch: expected {expected_stage!r}, found {stage!r}")

    estimator.model_.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    device = getattr(estimator, "device_", None)
    if device is not None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    best_state = {
        str(key): value.detach().cpu()
        for key, value in dict(payload["best_state_dict"]).items()
    }
    history_rows = [dict(row) for row in payload.get("history_rows", [])]
    epoch = int(payload.get("epoch", 0) or 0)
    estimator.resumed_from_checkpoint_ = str(checkpoint_path)
    estimator.resumed_from_epoch_ = epoch
    return {
        "start_epoch": epoch + 1,
        "history_rows": history_rows,
        "best_state": deepcopy(best_state),
        "best_metric": float(payload.get("best_metric", float("inf"))),
        "best_epoch": int(payload.get("best_epoch", 0) or 0),
        "epochs_without_improvement": int(payload.get("epochs_without_improvement", 0) or 0),
    }
