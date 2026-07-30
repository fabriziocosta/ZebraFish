"""Cheap deterministic process/artifact watchdog for campaign supervisors.

The watchdog never calls an LLM and never launches or terminates a trial.  It
records only operational facts and trigger observations for the reconciliation
controller to consume.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
from typing import Any

from src.observation_engine import read_history
from src.scientific_state import state_transaction


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) < 3 or fields[2] != "Z"
    except OSError:
        return True


def _process_identity(pid: Any) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "running": False}
    proc = Path(f"/proc/{pid}")
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        stat = (proc / "stat").read_text(encoding="utf-8").split()
        return {
            "pid": pid,
            "running": True,
            "command_hash": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "process_start_ticks": stat[21] if len(stat) > 21 else None,
        }
    except (OSError, IndexError):
        return {"pid": pid, "running": _running(pid)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _age_seconds(path: Path, now: datetime) -> float | None:
    try:
        return max(0.0, now.timestamp() - path.stat().st_mtime)
    except OSError:
        return None


def inspect_campaign(
    campaign_config: dict[str, Any],
    *,
    campaign_state_path: str | Path,
    stale_after_seconds: float = 180.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a bounded health snapshot from live process and artifact facts."""

    timestamp = now or _now()
    campaign_state = _read_json(Path(campaign_state_path))
    launch = campaign_state.get("active_launch_state", {})
    pid = launch.get("pid") if isinstance(launch, dict) else None
    run_dir = None
    if isinstance(launch, dict):
        run_dir = launch.get("run_dir")
    run_dir = run_dir or campaign_state.get("current_run_dir")
    history_paths = sorted(Path(str(run_dir)).rglob("*history*.csv")) if run_dir and Path(str(run_dir)).exists() else []
    checkpoint = launch.get("checkpoint_path") if isinstance(launch, dict) else None
    if not checkpoint:
        checkpoint = campaign_state.get("resume_checkpoint_path")
    fresh_paths = [path for path in history_paths if _age_seconds(path, timestamp) is not None]
    newest_age = min((_age_seconds(path, timestamp) for path in fresh_paths), default=None)
    process_running = _running(pid)
    identity = _process_identity(pid)
    expected_hash = launch.get("command_hash") if isinstance(launch, dict) else None
    expected_ticks = launch.get("process_start_ticks") if isinstance(launch, dict) else None
    identity_mismatch = bool(process_running and ((expected_hash and identity.get("command_hash") != expected_hash) or (expected_ticks and identity.get("process_start_ticks") != expected_ticks)))
    triggers: list[dict[str, Any]] = []
    if process_running and newest_age is not None and newest_age > stale_after_seconds:
        triggers.append({"type": "artifact_stale", "age_seconds": newest_age, "threshold_seconds": stale_after_seconds})
    if campaign_state.get("status") == "running" and not process_running:
        triggers.append({"type": "dead_process", "pid": pid})
    if identity_mismatch:
        triggers.append({"type": "process_identity_mismatch", "expected_command_hash": expected_hash, "observed_command_hash": identity.get("command_hash"), "expected_start_ticks": expected_ticks, "observed_start_ticks": identity.get("process_start_ticks")})
    if checkpoint and not Path(str(checkpoint)).exists() and campaign_state.get("status") in {"running", "suspended"}:
        triggers.append({"type": "checkpoint_missing", "path": str(checkpoint)})
    non_finite: list[str] = []
    for path in history_paths:
        for row in read_history(path):
            for key, value in row.items():
                if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
                    non_finite.append(key)
    if non_finite:
        triggers.append({"type": "non_finite_metric", "metrics": sorted(set(non_finite))})
    status = "running" if process_running else "dead_process" if campaign_state.get("status") == "running" else "inactive"
    return {
        "status": status,
        "pid": pid,
        "process_running": process_running,
        "process_identity": identity,
        "process_identity_mismatch": identity_mismatch,
        "run_dir": str(run_dir) if run_dir else None,
        "latest_history_age_seconds": newest_age,
        "checkpoint": {"path": str(checkpoint) if checkpoint else None, "available": bool(checkpoint and Path(str(checkpoint)).exists())},
        "triggers": triggers,
        "checked_at": timestamp.isoformat(timespec="seconds"),
        "detector_version": "watchdog-v1",
    }


def run_watchdog_once(
    campaign_config: dict[str, Any],
    *,
    state_path: str | Path,
    campaign_state_path: str | Path,
    stale_after_seconds: float = 180.0,
) -> dict[str, Any]:
    """Inspect once and persist only operational watchdog metadata."""

    snapshot = inspect_campaign(
        campaign_config,
        campaign_state_path=campaign_state_path,
        stale_after_seconds=stale_after_seconds,
    )
    with state_transaction(state_path) as state:
        state.setdefault("controller_state", {})["watchdog"] = snapshot
        state.setdefault("audit_log", []).append(
            {
                "id": f"watchdog_{snapshot['checked_at'].replace(':', '').replace('-', '')}",
                "created_at": snapshot["checked_at"],
                "actor": "campaign_watchdog",
                "operation": {"operation": "watchdog_observation", "trigger_count": len(snapshot["triggers"])},
            }
        )
    return snapshot
