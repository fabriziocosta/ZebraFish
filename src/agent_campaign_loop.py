from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, TextIO
import uuid

from src import agent_experiment_loop as experiment_loop
from src.experiment_runner_shared import merge_dicts, read_yaml_mapping, write_yaml_mapping

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ALLOWED_CAMPAIGN_DECISIONS = {
    "no_action",
    "propose_trial",
    "update_logbook",
    "stop_campaign",
}

CREDIT_EXHAUSTION_CODES = {
    "billing_hard_limit_reached",
    "billing_not_active",
    "insufficient_quota",
    "quota_exceeded",
}

RESTARTABLE_TERMINAL_STATUSES = {
    "terminated",
    "termination_race_stopped",
}


@dataclass(frozen=True)
class CampaignDecision:
    decision: str
    reason: str
    trial_patch: dict[str, Any] | None = None
    logbook_markdown: str | None = None


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state_error": f"Could not parse {path}"}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{target} must contain a mapping")
    return payload


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = _load_mapping(target)
    payload.setdefault("campaign", {})
    campaign = payload["campaign"]
    campaign.setdefault("id", target.stem)
    campaign.setdefault("loop_config", "configs/agent_experiment_loop.yaml")
    campaign.setdefault("poll_seconds", 3600)
    campaign.setdefault("trial_budget", 20)
    campaign.setdefault("max_patch_leaf_count", 2)
    campaign.setdefault("stages", [])
    if not campaign["stages"]:
        raise ValueError(f"{target} must define campaign.stages")

    root = Path(payload.get("artifacts", {}).get("root", f"artifacts/campaigns/{campaign['id']}"))
    payload.setdefault("artifacts", {})
    payload["artifacts"].setdefault("root", str(root))
    payload["artifacts"].setdefault("state_path", str(root / "campaign_state.json"))
    payload["artifacts"].setdefault("trials_csv", str(root / "trials.csv"))
    payload["artifacts"].setdefault("trials_jsonl", str(root / "trials.jsonl"))
    payload["artifacts"].setdefault("leaderboard_csv", str(root / "leaderboard.csv"))

    payload.setdefault("objective", {})
    objective = payload["objective"]
    objective.setdefault("target", "compound")
    objective.setdefault("primary_metric", "macro_f1")
    objective.setdefault("required_primary_metric", False)
    objective.setdefault("fallback_metrics", ["balanced_accuracy", "accuracy"])
    objective.setdefault("tie_breaker_metrics", ["roc_auc_ovr_macro"])
    objective.setdefault("maximize", True)
    objective.setdefault("minimums", {"action.accuracy": 0.0})

    payload.setdefault("logbook", {})
    payload["logbook"].setdefault("path", "EXPERIMENTS_LOGBOOK.md")
    payload.setdefault("prompts", {})
    payload["prompts"].setdefault(
        "decision_schema",
        (
            "Return one JSON object with decision no_action|propose_trial|update_logbook|stop_campaign, "
            "a reason, optional logbook_markdown, and optional trial_patch keyed by experiment id."
        ),
    )
    return payload


def _campaign_state_path(config: dict[str, Any]) -> Path:
    return Path(config["artifacts"]["state_path"])


def _campaign_root(config: dict[str, Any]) -> Path:
    return Path(config["artifacts"]["root"])


def _campaign_lock_path(config: dict[str, Any]) -> Path:
    return _campaign_root(config) / "campaign.lock"


def _campaign_terminate_lock_path(config: dict[str, Any]) -> Path:
    return _campaign_root(config) / "campaign.terminate.lock"


def _acquire_campaign_lock(config: dict[str, Any]) -> Path:
    path = _campaign_lock_path(config)
    return _acquire_pid_lock(path, label="Campaign")


def _acquire_terminate_lock(config: dict[str, Any]) -> Path:
    path = _campaign_terminate_lock_path(config)
    return _acquire_pid_lock(path, label="Campaign termination")


def _acquire_pid_lock(path: Path, *, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            existing_pid = _read_lock_pid(path)
            if attempt == 0 and existing_pid is not None and not experiment_loop._is_process_running(existing_pid):
                path.unlink(missing_ok=True)
                continue
            owner = f" pid={existing_pid}" if existing_pid is not None else ""
            raise RuntimeError(f"{label} lock already exists at {path}.{owner}")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": pid, "created_at": _now_iso()}, indent=2))
        return path
    raise RuntimeError(f"Could not acquire {label.lower()} lock at {path}")


def _read_lock_pid(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return int(payload.get("pid"))
    except (TypeError, ValueError, AttributeError):
        return None


def _release_campaign_lock(path: Path) -> None:
    existing_pid = _read_lock_pid(path)
    if existing_pid == os.getpid():
        path.unlink(missing_ok=True)


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _load_loop_config(config: dict[str, Any]) -> dict[str, Any]:
    return experiment_loop.load_loop_config(config["campaign"]["loop_config"])


def _stage_state_path(trial_dir: Path, stage: str) -> Path:
    return trial_dir / "stage_state" / f"{stage}.json"


def _stage_log_dir(trial_dir: Path, stage: str) -> Path:
    return trial_dir / "logs" / stage


def _trial_id(campaign_id: str, start_trial: str | None = None) -> str:
    if start_trial:
        return start_trial
    return f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _trial_dir(config: dict[str, Any], trial_id: str) -> Path:
    return _campaign_root(config) / trial_id


def _stage_output_root(trial_dir: Path, stage: str) -> Path:
    return trial_dir / "outputs" / stage


def _latest_file_tails(dirs: list[str | Path], *, limit: int = 4, max_lines: int = 40) -> list[dict[str, str]]:
    candidates: list[Path] = []
    for directory in dirs:
        root = Path(directory)
        if root.exists():
            candidates.extend(path for path in root.glob("*.log") if path.is_file())
    latest = sorted(candidates, key=lambda path: path.stat().st_mtime)[-limit:]
    return [{"path": str(path), "tail": _tail_text(path, max_lines=max_lines)} for path in latest]


def collect_init_snapshot(campaign_config: dict[str, Any], loop_config: dict[str, Any]) -> dict[str, Any]:
    stage_statuses: dict[str, Any] = {}
    for stage in campaign_config["campaign"]["stages"]:
        status = experiment_loop.collect_status(loop_config, state_override={"active_experiment": stage})
        controller_status, controller_reason = experiment_loop.classify_controller_status(loop_config, status)
        latest_artifacts = dict(status.get("artifacts", {}))
        if latest_artifacts.get("run_dir_source") == "latest_fallback":
            latest_artifacts["evidence_reliability"] = "low"
            latest_artifacts["evidence_warning"] = (
                "No precise state file or runner status identified this run; "
                "artifacts came from the newest folder fallback."
            )
        stage_statuses[stage] = {
            "controller_status": controller_status,
            "controller_reason": controller_reason,
            "latest_artifacts": latest_artifacts,
            "run_status": status.get("run_status", {}),
            "log_tail": status.get("log_tail", ""),
        }
    logbook_path = Path(campaign_config["logbook"]["path"])
    state_dir = Path(loop_config.get("state", {}).get("log_dir", "artifacts/agent_experiment_loop/logs"))
    return {
        "checked_at": _now_iso(),
        "campaign_id": campaign_config["campaign"]["id"],
        "stages": list(campaign_config["campaign"]["stages"]),
        "objective": campaign_config.get("objective", {}),
        "stage_statuses": stage_statuses,
        "logbook_tail": _tail_text(logbook_path, max_lines=180),
        "recent_logs": _latest_file_tails([state_dir, "artifacts/notebook_run_logs"], limit=6, max_lines=50),
        "existing_trials_csv": _tail_text(Path(campaign_config["artifacts"]["trials_csv"]), max_lines=40),
        "existing_leaderboard_csv": _tail_text(Path(campaign_config["artifacts"]["leaderboard_csv"]), max_lines=40),
    }


def _copy_stage_configs(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    trial_dir: Path,
    *,
    trial_patch: dict[str, Any] | None = None,
) -> dict[str, str]:
    trial_patch = trial_patch or {}
    config_dir = trial_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for stage in campaign_config["campaign"]["stages"]:
        source = Path(loop_config["experiments"][stage]["params_yaml"])
        target = config_dir / f"{stage}_{source.name}"
        if stage in trial_patch:
            _validate_trial_patch(loop_config, {stage: trial_patch[stage]})
            patched = merge_dicts(read_yaml_mapping(source), trial_patch[stage])
        else:
            patched = read_yaml_mapping(source)
        patched["experiment_output_dir"] = str(_stage_output_root(trial_dir, stage))
        write_yaml_mapping(target, patched)
        copied[stage] = str(target)
    return copied


def _trial_count(campaign_config: dict[str, Any]) -> int:
    trial_ids: set[str] = set()
    trials_csv = Path(campaign_config["artifacts"]["trials_csv"])
    if trials_csv.exists():
        with trials_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                trial_id = str(row.get("trial_id") or "").strip()
                if trial_id:
                    trial_ids.add(trial_id)

    for path in _iter_campaign_trial_dirs(campaign_config):
        manifest_path = path / "trial_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest_path.exists() and _manifest_counts_against_budget(manifest):
            trial_ids.add(path.name)
        elif not manifest_path.exists() and _trial_dir_without_manifest_counts(path):
            trial_ids.add(path.name)

    state = _read_json(_campaign_state_path(campaign_config))
    current_trial_id = str(state.get("current_trial_id") or "").strip()
    if current_trial_id and _state_counts_against_budget(state):
        trial_ids.add(current_trial_id)
    return len(trial_ids)


def _iter_campaign_trial_dirs(campaign_config: dict[str, Any]) -> list[Path]:
    root = _campaign_root(campaign_config)
    if not root.exists():
        return []
    ignored_names = {
        "trials",
        "campaign.lock",
        "campaign.terminate.lock",
        "campaign_state.json",
        "trials.csv",
        "trials.jsonl",
        "leaderboard.csv",
    }
    return [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name not in ignored_names and _trial_dir_without_manifest_counts(path)
    ]


def _manifest_counts_against_budget(manifest: dict[str, Any]) -> bool:
    if not manifest:
        return False
    return bool(manifest.get("trial_configs") or manifest.get("stage_runs"))


def _trial_dir_without_manifest_counts(path: Path) -> bool:
    return any((path / name).exists() for name in ("configs", "stage_state", "logs", "trial_summary.json"))


def _state_counts_against_budget(state: dict[str, Any]) -> bool:
    if not state:
        return False
    if state.get("phase") == "initializing" and not state.get("trial_configs"):
        return False
    return bool(state.get("trial_configs") or state.get("stage_runs") or state.get("active_launch_state"))


def _active_stage_status_if_available(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], str, str] | None:
    required_keys = {"current_stage", "current_trial_dir", "trial_configs", "stage_state_path"}
    if not required_keys.issubset(state):
        return None
    try:
        return _collect_stage_status(campaign_config, loop_config, state)
    except Exception:
        return None


def _active_campaign_pid(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None, str]:
    active_status = _active_stage_status_if_available(campaign_config, loop_config, state)
    if active_status is not None:
        status, _controller_status, _controller_reason = active_status
        pid = status.get("pid")
        if isinstance(pid, int) and status.get("process_running"):
            return pid, status, "stage_state"
    launch_pid = state.get("active_launch_state", {}).get("pid")
    if isinstance(launch_pid, int) and experiment_loop._is_process_running(launch_pid):
        return launch_pid, active_status[0] if active_status is not None else None, "campaign_state"
    return None, active_status[0] if active_status is not None else None, "none"


def campaign_live_status(campaign_config: dict[str, Any]) -> dict[str, Any]:
    loop_config = _load_loop_config(campaign_config)
    state = _read_json(_campaign_state_path(campaign_config))
    if not state:
        return {
            "campaign_id": campaign_config["campaign"]["id"],
            "running": False,
            "reason": "state missing",
        }
    pid, status, pid_source = _active_campaign_pid(campaign_config, loop_config, state)
    return {
        "campaign_id": campaign_config["campaign"]["id"],
        "running": pid is not None,
        "pid": pid,
        "pid_source": pid_source,
        "state": state,
        "stage_status": status,
        "updated_at": state.get("updated_at"),
        "state_mtime": _campaign_state_path(campaign_config).stat().st_mtime if _campaign_state_path(campaign_config).exists() else 0,
    }


def _assert_trial_budget_available(campaign_config: dict[str, Any]) -> None:
    budget = int(campaign_config["campaign"].get("trial_budget", 0) or 0)
    if budget <= 0:
        return
    used = _trial_count(campaign_config)
    if used >= budget:
        raise RuntimeError(f"Campaign trial budget exhausted: used {used} of {budget}")


def _loop_config_for_stage(
    base_loop_config: dict[str, Any],
    trial_dir: Path,
    stage: str,
    trial_configs: dict[str, str],
) -> dict[str, Any]:
    loop_config = json.loads(json.dumps(base_loop_config))
    loop_config["state"]["path"] = str(_stage_state_path(trial_dir, stage))
    loop_config["state"]["log_dir"] = str(_stage_log_dir(trial_dir, stage))
    for experiment, params_path in trial_configs.items():
        loop_config["experiments"][experiment]["params_yaml"] = params_path
        try:
            params = read_yaml_mapping(params_path)
        except FileNotFoundError:
            params = {}
        experiment_output_dir = params.get("experiment_output_dir")
        if experiment_output_dir:
            loop_config["experiments"][experiment]["artifact_root"] = str(experiment_output_dir)
    return loop_config


def _validate_chain(campaign_config: dict[str, Any], loop_config: dict[str, Any]) -> None:
    stages = list(campaign_config["campaign"]["stages"])
    for stage in stages:
        if stage not in loop_config["experiments"]:
            raise ValueError(f"Campaign references unknown experiment {stage!r}")
    for left, right in zip(stages, stages[1:]):
        configured_next = loop_config["experiments"][left].get("next")
        if configured_next != right:
            raise ValueError(f"Campaign stage {left!r} expects next {right!r}, but loop config has {configured_next!r}")


def _patch_leaf_paths(patch: dict[str, Any], *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            paths.extend(_patch_leaf_paths(value, prefix=path))
        else:
            paths.append(path)
    return paths


def _max_patch_leaf_count(campaign_config: dict[str, Any]) -> int | None:
    raw = campaign_config.get("campaign", {}).get("max_patch_leaf_count")
    if raw in (None, "", 0, "0"):
        return None
    return int(raw)


def _validate_trial_patch(
    loop_config: dict[str, Any],
    trial_patch: dict[str, Any],
    *,
    max_leaf_count: int | None = None,
) -> None:
    all_leaf_paths: list[str] = []
    for experiment, patch in trial_patch.items():
        if experiment not in loop_config["experiments"]:
            raise ValueError(f"trial_patch references unknown experiment {experiment!r}")
        if not isinstance(patch, dict):
            raise ValueError(f"trial_patch for {experiment} must be a mapping")
        leaf_paths = _patch_leaf_paths(patch)
        all_leaf_paths.extend(f"{experiment}.{path}" for path in leaf_paths)
        allowed = [str(path) for path in loop_config["experiments"][experiment].get("allowed_patch_paths", [])]
        rejected = [
            path
            for path in leaf_paths
            if not any(path == allowed_path or path.startswith(f"{allowed_path}.") for allowed_path in allowed)
        ]
        if rejected:
            raise ValueError(
                f"trial_patch for {experiment} contains non-allowlisted path(s): "
                + ", ".join(sorted(rejected))
            )
    if max_leaf_count is not None and len(all_leaf_paths) > max_leaf_count:
        raise ValueError(
            f"trial_patch changes {len(all_leaf_paths)} leaf value(s), exceeding max_patch_leaf_count={max_leaf_count}: "
            + ", ".join(sorted(all_leaf_paths))
        )


def start_trial(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    *,
    start_trial: str | None = None,
    trial_patch: dict[str, Any] | None = None,
    dry_run: bool = False,
    allow_existing_trial_dir: bool = False,
) -> dict[str, Any]:
    _validate_chain(campaign_config, loop_config)
    _validate_trial_patch(
        loop_config,
        trial_patch or {},
        max_leaf_count=_max_patch_leaf_count(campaign_config),
    )
    if not dry_run:
        _assert_trial_budget_available(campaign_config)
    campaign_id = str(campaign_config["campaign"]["id"])
    trial_id = _trial_id(campaign_id, start_trial)
    trial_dir = _trial_dir(campaign_config, trial_id)
    if not dry_run and trial_dir.exists() and not allow_existing_trial_dir:
        raise FileExistsError(f"Campaign trial folder already exists: {trial_dir}")
    stages = list(campaign_config["campaign"]["stages"])
    if dry_run:
        trial_configs = {
            stage: str(trial_dir / "configs" / f"{stage}_{Path(loop_config['experiments'][stage]['params_yaml']).name}")
            for stage in stages
        }
        launched = experiment_loop.launch_experiment(
            _loop_config_for_stage(loop_config, trial_dir, stages[0], trial_configs),
            stages[0],
            dry_run=True,
        )
    else:
        trial_configs = _copy_stage_configs(campaign_config, loop_config, trial_dir, trial_patch=trial_patch)
    state = {
        "campaign_id": campaign_id,
        "status": "dry_run" if dry_run else "launching",
        "phase": "running",
        "current_trial_id": trial_id,
        "current_trial_dir": str(trial_dir),
        "current_stage_index": 0,
        "current_stage": stages[0],
        "stage_state_path": str(_stage_state_path(trial_dir, stages[0])),
        "trial_configs": trial_configs,
        "trial_patch": trial_patch or {},
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "trials": [],
    }
    if dry_run:
        state["active_launch_state"] = launched
        state["status"] = "running"
        return state

    _write_json(_campaign_state_path(campaign_config), state)
    _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))
    try:
        launched = experiment_loop.launch_experiment(
            _loop_config_for_stage(loop_config, trial_dir, stages[0], trial_configs),
            stages[0],
            dry_run=False,
        )
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failure_reason": f"launch failed: {type(exc).__name__}: {exc}",
                "updated_at": _now_iso(),
            }
        )
        _persist_campaign_state(campaign_config, state)
        raise
    state["active_launch_state"] = launched
    state["status"] = "running"
    state["updated_at"] = _now_iso()
    if not dry_run:
        _write_json(_campaign_state_path(campaign_config), state)
        _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))
    return state


def _trial_manifest(campaign_config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": campaign_config["campaign"]["id"],
        "trial_id": state.get("current_trial_id"),
        "trial_dir": state.get("current_trial_dir"),
        "stages": campaign_config["campaign"]["stages"],
        "trial_configs": state.get("trial_configs", {}),
        "stage_runs": state.get("stage_runs", {}),
        "trial_patch": state.get("trial_patch", {}),
        "updated_at": _now_iso(),
    }


def _find_pretraining_config(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_config.yaml"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_completed_stage_config(status: dict[str, Any]) -> Path | None:
    candidates = status.get("artifacts", {}).get("latest_config_yamls") or []
    for candidate in reversed(candidates):
        path = Path(candidate)
        if path.exists():
            return path
    run_dir = status.get("artifacts", {}).get("run_dir")
    return _find_pretraining_config(Path(run_dir)) if run_dir else None


def wire_pretrain_checkpoint_into_finetune_config(
    finetune_config_path: str | Path,
    pretrain_run_dir: str | Path,
    *,
    pretrain_config_path: str | Path | None = None,
) -> Path:
    pretrain_config = Path(pretrain_config_path) if pretrain_config_path else _find_pretraining_config(Path(pretrain_run_dir))
    if pretrain_config is None:
        raise FileNotFoundError(f"No *_config.yaml found in pretrain run folder {pretrain_run_dir}")
    if not pretrain_config.exists():
        raise FileNotFoundError(f"Pretraining config artifact does not exist: {pretrain_config}")
    finetune_path = Path(finetune_config_path)
    raw = read_yaml_mapping(finetune_path)
    raw["pretraining_config_path"] = str(pretrain_config)
    write_yaml_mapping(finetune_path, raw)
    return pretrain_config


def _collect_stage_status(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    stage = str(state["current_stage"])
    trial_dir = Path(state["current_trial_dir"])
    stage_loop_config = _loop_config_for_stage(loop_config, trial_dir, stage, state["trial_configs"])
    status = experiment_loop.collect_status(stage_loop_config, state_path=Path(state["stage_state_path"]))
    controller_status, controller_reason = experiment_loop.classify_controller_status(stage_loop_config, status)
    status["controller_status"] = controller_status
    status["controller_reason"] = controller_reason
    return status, controller_status, controller_reason


def _advance_stage_or_complete_trial(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    status: dict[str, Any],
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], str]:
    stages = list(campaign_config["campaign"]["stages"])
    stage = str(state["current_stage"])
    stage_runs = dict(state.get("stage_runs", {}))
    run_dir = status.get("artifacts", {}).get("run_dir")
    if run_dir:
        stage_runs[stage] = run_dir
    state["stage_runs"] = stage_runs

    next_index = int(state["current_stage_index"]) + 1
    if next_index >= len(stages):
        summary = summarize_trial(campaign_config, state)
        state["status"] = "trial_completed"
        state["completed_at"] = _now_iso()
        state["score"] = summary.get("score")
        state["ranking_score"] = summary.get("ranking_score")
        state["objective_eligible"] = summary.get("objective_eligible")
        state["objective_metric_path"] = summary.get("metrics_path")
        state["updated_at"] = _now_iso()
        if not dry_run:
            _persist_campaign_state(campaign_config, state)
            _write_trial_outputs(campaign_config, state, summary)
        return state, "trial_completed"

    next_stage = stages[next_index]
    if stage_runs.get(stage):
        pretrain_config = wire_pretrain_checkpoint_into_finetune_config(
            state["trial_configs"][next_stage],
            stage_runs[stage],
            pretrain_config_path=_find_completed_stage_config(status),
        )
        state.setdefault("stage_config_artifacts", {})[stage] = str(pretrain_config)
    trial_dir = Path(state["current_trial_dir"])
    next_loop_config = _loop_config_for_stage(loop_config, trial_dir, next_stage, state["trial_configs"])
    state.update(
        {
            "status": "dry_run" if dry_run else "launching",
            "current_stage_index": next_index,
            "current_stage": next_stage,
            "stage_state_path": str(_stage_state_path(trial_dir, next_stage)),
            "updated_at": _now_iso(),
        }
    )
    if not dry_run:
        _persist_campaign_state(campaign_config, state)
    try:
        launched = experiment_loop.launch_experiment(next_loop_config, next_stage, dry_run=dry_run)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failure_reason": f"launch failed: {type(exc).__name__}: {exc}",
                "updated_at": _now_iso(),
            }
        )
        if not dry_run:
            _persist_campaign_state(campaign_config, state)
        raise
    state["active_launch_state"] = launched
    state["status"] = "running"
    state["updated_at"] = _now_iso()
    if not dry_run:
        _persist_campaign_state(campaign_config, state)
    return state, f"launched_{next_stage}"


def _metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_latest_summary_metrics(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("*summary_metrics*.csv"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _metric_lookup(rows: list[dict[str, str]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        target = row.get("target")
        metric = row.get("metric")
        value = row.get("value")
        if not target or not metric or value is None:
            continue
        try:
            metrics[f"{target}.{metric}"] = float(value)
        except ValueError:
            continue
    return metrics


def score_metrics(metrics_path: str | Path, objective: dict[str, Any]) -> dict[str, Any]:
    rows = _metric_rows(Path(metrics_path))
    metrics = _metric_lookup(rows)
    target = str(objective.get("target", "compound"))
    primary_metric = str(objective.get("primary_metric", "macro_f1"))
    required_primary = bool(objective.get("required_primary_metric", False))
    candidates = [primary_metric]
    if not required_primary:
        candidates.extend(str(metric) for metric in objective.get("fallback_metrics", []))
    selected_metric = None
    selected_value = None
    for metric in candidates:
        key = f"{target}.{metric}"
        if key in metrics:
            selected_metric = key
            selected_value = metrics[key]
            break
    tie_breaker_metrics = [str(metric) for metric in objective.get("tie_breaker_metrics", [])]
    ranking_metric_order = []
    if selected_metric is not None:
        ranking_metric_order.append(selected_metric)
    ranking_metric_order.extend(
        f"{target}.{metric}"
        for metric in tie_breaker_metrics
        if f"{target}.{metric}" != selected_metric
    )
    ranking_values = [metrics.get(key) for key in ranking_metric_order]
    minimums = {str(key): float(value) for key, value in objective.get("minimums", {}).items()}
    guardrail_failures = {
        key: {"minimum": minimum, "actual": metrics.get(key)}
        for key, minimum in minimums.items()
        if metrics.get(key) is None or float(metrics[key]) < minimum
    }
    return {
        "score": selected_value,
        "ranking_score": selected_value if selected_value is not None and not guardrail_failures else None,
        "ranking_values": ranking_values if selected_value is not None and not guardrail_failures else [],
        "ranking_metric_order": ranking_metric_order if selected_value is not None and not guardrail_failures else [],
        "selected_metric": selected_metric,
        "guardrail_passed": not guardrail_failures,
        "guardrail_failures": guardrail_failures,
        "metrics": metrics,
    }


def summarize_trial(campaign_config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    stages = list(campaign_config["campaign"]["stages"])
    final_stage = stages[-1]
    final_run_value = state.get("stage_runs", {}).get(final_stage)
    final_run_dir = Path(final_run_value) if final_run_value else None
    metrics_path = _find_latest_summary_metrics(final_run_dir) if final_run_dir and final_run_dir.exists() else None
    scored = score_metrics(metrics_path, campaign_config["objective"]) if metrics_path else {
        "score": None,
        "ranking_score": None,
        "ranking_values": [],
        "ranking_metric_order": [],
        "selected_metric": None,
        "guardrail_passed": False,
        "guardrail_failures": {"metrics": {"minimum": "present", "actual": None}},
        "metrics": {},
    }
    return {
        "trial_id": state.get("current_trial_id"),
        "trial_dir": state.get("current_trial_dir"),
        "status": state.get("status"),
        "score": scored["score"],
        "ranking_score": scored["ranking_score"],
        "ranking_values": scored["ranking_values"],
        "ranking_metric_order": scored["ranking_metric_order"],
        "objective_eligible": bool(scored["guardrail_passed"] and scored["ranking_score"] is not None),
        "selected_metric": scored["selected_metric"],
        "guardrail_passed": scored["guardrail_passed"],
        "guardrail_failures": scored["guardrail_failures"],
        "metrics_path": str(metrics_path) if metrics_path else None,
        "stage_runs": state.get("stage_runs", {}),
        "trial_configs": state.get("trial_configs", {}),
        "metrics": scored["metrics"],
    }


def _persist_campaign_state(campaign_config: dict[str, Any], state: dict[str, Any]) -> None:
    _write_json(_campaign_state_path(campaign_config), state)
    trial_dir = Path(state["current_trial_dir"])
    _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))


def _write_trial_outputs(campaign_config: dict[str, Any], state: dict[str, Any], summary: dict[str, Any]) -> None:
    trial_dir = Path(state["current_trial_dir"])
    _write_json(trial_dir / "trial_summary.json", summary)
    trial_records = _trial_records(campaign_config, state, summary)
    _write_csv(Path(campaign_config["artifacts"]["trials_csv"]), trial_records)
    _write_jsonl(Path(campaign_config["artifacts"]["trials_jsonl"]), trial_records)
    maximize = bool(campaign_config["objective"].get("maximize", True))
    leaderboard = sorted(
        [
            record
            for record in trial_records
            if record.get("objective_eligible") in {True, "True", "true"}
            and record.get("ranking_score") not in ("", None)
        ],
        key=lambda record: _leaderboard_sort_key(record, maximize=maximize),
        reverse=maximize,
    )
    _write_csv(Path(campaign_config["artifacts"]["leaderboard_csv"]), leaderboard)
    _upsert_campaign_logbook(campaign_config, summary, model_markdown=None)


def _trial_records(campaign_config: dict[str, Any], state: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    previous = []
    trials_csv = Path(campaign_config["artifacts"]["trials_csv"])
    if trials_csv.exists():
        with trials_csv.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
    current = {
        "campaign_id": campaign_config["campaign"]["id"],
        "trial_id": summary["trial_id"],
        "status": state.get("status"),
        "score": "" if summary.get("score") is None else summary.get("score"),
        "ranking_score": "" if summary.get("ranking_score") is None else summary.get("ranking_score"),
        "ranking_values": json.dumps(summary.get("ranking_values") or []),
        "ranking_metric_order": json.dumps(summary.get("ranking_metric_order") or []),
        "objective_eligible": bool(summary.get("objective_eligible")),
        "selected_metric": summary.get("selected_metric") or "",
        "guardrail_passed": summary.get("guardrail_passed"),
        "trial_dir": summary.get("trial_dir"),
        "metrics_path": summary.get("metrics_path") or "",
        "updated_at": _now_iso(),
    }
    by_trial = {str(record.get("trial_id")): record for record in previous}
    by_trial[str(current["trial_id"])] = current
    return list(by_trial.values())


def _leaderboard_sort_key(record: dict[str, Any], *, maximize: bool) -> tuple[float, ...]:
    raw = record.get("ranking_values")
    values: list[Any]
    if isinstance(raw, str) and raw.strip():
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = []
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    missing = float("-inf") if maximize else float("inf")
    key: list[float] = []
    for value in values:
        try:
            key.append(float(value))
        except (TypeError, ValueError):
            key.append(missing)
    if not key:
        try:
            key.append(float(record["ranking_score"]))
        except (KeyError, TypeError, ValueError):
            key.append(missing)
    return tuple(key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "campaign_id",
        "trial_id",
        "status",
        "score",
        "ranking_score",
        "ranking_values",
        "ranking_metric_order",
        "objective_eligible",
        "selected_metric",
        "guardrail_passed",
        "trial_dir",
        "metrics_path",
        "updated_at",
    ]
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp_path.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _rel(path: str | Path) -> str:
    target = Path(path)
    try:
        return str(target.relative_to(Path.cwd()))
    except ValueError:
        return str(target)


def _latest_pdfs(run_dir: str | Path, *, limit: int = 4) -> list[str]:
    if not run_dir:
        return []
    root = Path(run_dir)
    if not root.exists():
        return []
    return [_rel(path) for path in sorted(root.rglob("*.pdf"), key=lambda item: item.stat().st_mtime)[-limit:]]


def _render_pdf_preview_png(pdf_path: str | Path, preview_dir: str | Path) -> Path | None:
    source = Path(pdf_path)
    if not source.exists():
        return None
    try:
        import fitz
    except ModuleNotFoundError:
        return None
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    target_dir = Path(preview_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}_{digest}.png"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    try:
        document = fitz.open(source)
        try:
            if document.page_count == 0:
                return None
            page = document.load_page(0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(target)
        finally:
            document.close()
    except Exception:
        return None
    return target


def _print_analysis_block(
    stream: TextIO,
    *,
    title: str,
    markdown: str,
    trial_patch: dict[str, Any] | None = None,
) -> None:
    print(f"\n{title}", file=stream, flush=True)
    print("-" * len(title), file=stream, flush=True)
    print(markdown.strip() or "No analysis text returned.", file=stream, flush=True)
    if trial_patch:
        print("\nproposed trial patch:", file=stream, flush=True)
        print(json.dumps(trial_patch, indent=2, sort_keys=True), file=stream, flush=True)
    print("", file=stream, flush=True)


def _upsert_campaign_logbook(
    campaign_config: dict[str, Any],
    summary: dict[str, Any],
    *,
    model_markdown: str | None,
) -> None:
    trial_id = str(summary.get("trial_id"))
    lines = [
        f"## Campaign Trial: {trial_id}",
        "",
        f"- Campaign: `{campaign_config['campaign']['id']}`",
        f"- Trial folder: [{_rel(summary['trial_dir'])}]({_rel(summary['trial_dir'])})",
        f"- Score: `{summary.get('score')}` via `{summary.get('selected_metric')}`",
        f"- Metrics: [{_rel(summary['metrics_path'])}]({_rel(summary['metrics_path'])})" if summary.get("metrics_path") else "- Metrics: missing",
    ]
    preview_dir = Path(str(summary["trial_dir"])) / "plot_previews" if summary.get("trial_dir") else None
    for stage, run_dir in summary.get("stage_runs", {}).items():
        lines.append(f"- {stage} run: [{_rel(run_dir)}]({_rel(run_dir)})")
        for pdf in _latest_pdfs(run_dir, limit=2):
            lines.append(f"  - PDF: [{pdf}]({pdf})")
            preview = _render_pdf_preview_png(pdf, preview_dir) if preview_dir is not None else None
            if preview is not None:
                preview_link = _rel(preview)
                lines.append(f"    ![{Path(pdf).stem}]({preview_link})")
            else:
                lines.append("    - Preview: unavailable; inspect the linked PDF directly.")
    for stage, config_path in summary.get("trial_configs", {}).items():
        lines.append(f"- {stage} config: [{_rel(config_path)}]({_rel(config_path)})")
    if model_markdown:
        lines.extend(["", model_markdown.strip()])
    marker = f"campaign:{campaign_config['campaign']['id']}:{trial_id}"
    experiment_loop._upsert_marked_block(Path(campaign_config["logbook"]["path"]), marker, "\n".join(lines))


def _upsert_init_logbook(
    campaign_config: dict[str, Any],
    *,
    trial_id: str,
    decision: CampaignDecision,
    trial_patch: dict[str, Any],
) -> None:
    trial_dir = _trial_dir(campaign_config, trial_id)
    lines = [
        f"## Campaign Start: {trial_id}",
        "",
        f"- Campaign: `{campaign_config['campaign']['id']}`",
        f"- Trial folder: [{_rel(trial_dir)}]({_rel(trial_dir)})",
        f"- Stages: `{ ' -> '.join(str(stage) for stage in campaign_config['campaign']['stages']) }`",
        f"- Objective: `{campaign_config['objective'].get('target')}.{campaign_config['objective'].get('primary_metric')}`",
        "",
        "### Previous Results And Next Plan",
        "",
        decision.logbook_markdown or decision.reason or "Start the configured baseline trial and evaluate downstream fine-tune metrics.",
        "",
        "### Initial Trial Patch",
        "",
        "```json",
        json.dumps(trial_patch, indent=2, sort_keys=True),
        "```",
    ]
    marker = f"campaign:{campaign_config['campaign']['id']}:init:{trial_id}"
    experiment_loop._upsert_marked_block(Path(campaign_config["logbook"]["path"]), marker, "\n".join(lines))


def parse_campaign_decision(text: str) -> CampaignDecision:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Campaign response must be a JSON object")
    decision = str(payload.get("decision", ""))
    if decision not in ALLOWED_CAMPAIGN_DECISIONS:
        raise ValueError(f"Unsupported campaign decision: {decision!r}")
    trial_patch = payload.get("trial_patch")
    if isinstance(trial_patch, str):
        trial_patch = json.loads(trial_patch) if trial_patch.strip() else {}
    if trial_patch is not None and not isinstance(trial_patch, dict):
        raise ValueError("trial_patch must be a mapping when provided")
    reason = str(payload.get("reason", "")).strip()
    logbook_markdown = payload.get("logbook_markdown")
    if not reason and not (isinstance(logbook_markdown, str) and logbook_markdown.strip()):
        raise ValueError("Campaign decision must include non-empty reason or logbook_markdown")
    return CampaignDecision(
        decision=decision,
        reason=reason,
        trial_patch=trial_patch,
        logbook_markdown=logbook_markdown,
    )


def _campaign_decision_text_format(stages: list[str]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "campaign_decision",
            "description": "A single campaign orchestration decision.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["decision", "reason", "logbook_markdown", "trial_patch"],
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": sorted(ALLOWED_CAMPAIGN_DECISIONS),
                    },
                    "reason": {"type": "string"},
                    "logbook_markdown": {"type": "string"},
                    "trial_patch": {
                        "type": "string",
                        "description": (
                            "A JSON-encoded object keyed by experiment id, for example "
                            "{\"10C\":{\"optimization_config\":{\"epochs\":3}}}. Use {} when no patch is proposed. "
                            f"Allowed experiment ids: {', '.join(str(stage) for stage in stages)}."
                        ),
                    },
                },
            },
        }
    }


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for content_item in content:
                    text = getattr(content_item, "text", None)
                    if isinstance(text, str):
                        text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    raise ValueError("OpenAI response did not contain output_text")


def _build_campaign_prompt(config: dict[str, Any], state: dict[str, Any], summary: dict[str, Any]) -> str:
    prompts = config.get("prompts", {})
    return "\n\n".join(
        [
            "You are coordinating ZebraFish experiment campaigns.",
            "A campaign trial is the full pretrain-to-finetune chain. Judge success by downstream finetune metrics.",
            f"Objective:\n{json.dumps(config.get('objective', {}), indent=2, sort_keys=True)}",
            f"Campaign policy:\n{prompts.get('analysis_policy', '')}",
            f"Logbook rule:\n{prompts.get('update_logbook', '')}",
            f"Parameter patching rule:\n{prompts.get('patch_parameters', '')}",
            f"Decision schema:\n{prompts.get('decision_schema', '')}",
            "Campaign state JSON:",
            json.dumps(state, indent=2, sort_keys=True),
            "Latest trial summary JSON:",
            json.dumps(summary, indent=2, sort_keys=True),
        ]
    )


def _build_init_prompt(config: dict[str, Any], snapshot: dict[str, Any], trial_id: str) -> str:
    prompts = config.get("prompts", {})
    return "\n\n".join(
        [
            "You are initializing a ZebraFish experiment campaign.",
            "Inspect the current status, recent logs, old experiment artifacts, existing ledgers, and logbook tail before the first stage launches.",
            (
                "In logbook_markdown, write concise human-readable markdown with these headings: "
                "Previous results reviewed, Next experiment to run, Why this should help, Patch to apply, Monitoring plan. "
                "Mention the concrete old evidence you used; if evidence is weak or from latest-folder fallback, say so."
            ),
            "The controller will launch the first campaign stage after this decision unless you return stop_campaign.",
            f"Proposed trial id: {trial_id}",
            f"Objective:\n{json.dumps(config.get('objective', {}), indent=2, sort_keys=True)}",
            f"Campaign policy:\n{prompts.get('analysis_policy', '')}",
            f"Logbook rule:\n{prompts.get('update_logbook', '')}",
            f"Parameter patching rule:\n{prompts.get('patch_parameters', '')}",
            f"Decision schema:\n{prompts.get('decision_schema', '')}",
            (
                "Return propose_trial when you have a concrete first trial patch. "
                "Return update_logbook or no_action only if the baseline config should be launched unchanged. "
                "Keep logbook_markdown under 1200 words and trial_patch compact."
            ),
            "Initialization snapshot JSON:",
            json.dumps(snapshot, indent=2, sort_keys=True),
        ]
    )


def _mark_agent_decision_failed(
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    exc: Exception,
    *,
    stream: TextIO,
) -> None:
    state["status"] = "agent_decision_failed"
    state["agent_decision_error"] = f"{type(exc).__name__}: {exc}"
    state["updated_at"] = _now_iso()
    _persist_campaign_state(campaign_config, state)
    print(
        f"campaign agent decision failed trial={state.get('current_trial_id')} "
        f"error={state['agent_decision_error']}",
        file=stream,
        flush=True,
    )


def _is_openai_credits_exhausted(exc: Exception) -> bool:
    structured_values: list[str] = []
    for attr in ("code", "type"):
        value = getattr(exc, attr, None)
        if value:
            structured_values.append(str(value).lower())
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type", "message"):
                value = error.get(key)
                if value:
                    structured_values.append(str(value).lower())
    if any(value in CREDIT_EXHAUSTION_CODES for value in structured_values):
        return True

    message = f"{type(exc).__name__}: {exc}".lower()
    credit_markers = [
        "insufficient_quota",
        "exceeded your current quota",
        "billing hard limit",
        "billing_not_active",
        "quota_exceeded",
        "credits exhausted",
        "credit balance",
    ]
    return any(marker in message for marker in credit_markers)


def _mark_openai_credits_exhausted(
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    exc: Exception,
    *,
    stream: TextIO,
) -> None:
    state["status"] = "openai_credits_exhausted"
    state["openai_error"] = f"{type(exc).__name__}: {exc}"
    state["updated_at"] = _now_iso()
    if state.get("current_trial_dir"):
        _persist_campaign_state(campaign_config, state)
    else:
        _write_json(_campaign_state_path(campaign_config), state)
    print(
        "OpenAI API credits appear to be exhausted; terminating campaign loop. "
        f"error={state['openai_error']}",
        file=stream,
        flush=True,
    )


def request_campaign_decision(
    config: dict[str, Any],
    state: dict[str, Any],
    summary: dict[str, Any],
    *,
    client: Any | None = None,
) -> CampaignDecision:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    agent = config.get("agent", {})
    response = client.responses.create(
        model=agent.get("model", "gpt-5.3-codex"),
        reasoning={"effort": agent.get("reasoning_effort", "medium")},
        max_output_tokens=int(agent.get("max_output_tokens", 2000)),
        text=_campaign_decision_text_format(list(config["campaign"]["stages"])),
        input=_build_campaign_prompt(config, state, summary),
    )
    return parse_campaign_decision(_response_output_text(response))


def request_init_decision(
    config: dict[str, Any],
    snapshot: dict[str, Any],
    trial_id: str,
    *,
    client: Any | None = None,
) -> CampaignDecision:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    agent = config.get("agent", {})
    response = client.responses.create(
        model=agent.get("model", "gpt-5.3-codex"),
        reasoning={"effort": agent.get("reasoning_effort", "medium")},
        max_output_tokens=int(agent.get("max_output_tokens", 2000)),
        text=_campaign_decision_text_format(list(config["campaign"]["stages"])),
        input=_build_init_prompt(config, snapshot, trial_id),
    )
    return parse_campaign_decision(_response_output_text(response))


def initialize_campaign_trial(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    *,
    start_trial_id: str | None = None,
    client: Any | None = None,
    stream: TextIO = sys.stdout,
) -> dict[str, Any]:
    campaign_id = str(campaign_config["campaign"]["id"])
    trial_id = _trial_id(campaign_id, start_trial_id)
    snapshot = collect_init_snapshot(campaign_config, loop_config)
    trial_dir = _trial_dir(campaign_config, trial_id)
    snapshot_path = trial_dir / "init_snapshot.json"
    _write_json(snapshot_path, snapshot)
    try:
        decision = request_init_decision(campaign_config, snapshot, trial_id, client=client)
    except Exception as exc:
        state = _init_failure_state(campaign_config, trial_id=trial_id, trial_dir=trial_dir, snapshot_path=snapshot_path, exc=exc)
        if _is_openai_credits_exhausted(exc):
            _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))
            _mark_openai_credits_exhausted(campaign_config, state, exc, stream=stream)
        else:
            _write_json(_campaign_state_path(campaign_config), state)
            _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))
            print(
                f"campaign initialization decision failed trial={trial_id} error={state['agent_decision_error']}",
                file=stream,
                flush=True,
            )
        return state
    try:
        return _apply_init_decision(
            campaign_config,
            loop_config,
            trial_id=trial_id,
            decision=decision,
            stream=stream,
            allow_existing_trial_dir=True,
        )
    except ValueError as exc:
        state = _init_failure_state(campaign_config, trial_id=trial_id, trial_dir=trial_dir, snapshot_path=snapshot_path, exc=exc)
        _write_json(_campaign_state_path(campaign_config), state)
        _write_json(trial_dir / "trial_manifest.json", _trial_manifest(campaign_config, state))
        print(
            f"campaign initialization decision rejected trial={trial_id} error={state['agent_decision_error']}",
            file=stream,
            flush=True,
        )
        return state


def _init_failure_state(
    campaign_config: dict[str, Any],
    *,
    trial_id: str,
    trial_dir: Path,
    snapshot_path: Path,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_config["campaign"]["id"],
        "status": "agent_decision_failed",
        "phase": "initializing",
        "current_trial_id": trial_id,
        "current_trial_dir": str(trial_dir),
        "current_stage_index": 0,
        "current_stage": campaign_config["campaign"]["stages"][0],
        "stage_state_path": str(_stage_state_path(trial_dir, campaign_config["campaign"]["stages"][0])),
        "trial_configs": {},
        "init_snapshot_path": str(snapshot_path),
        "agent_decision_error": f"{type(exc).__name__}: {exc}",
        "updated_at": _now_iso(),
    }


def _apply_init_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    *,
    trial_id: str,
    decision: CampaignDecision,
    stream: TextIO,
    allow_existing_trial_dir: bool,
) -> dict[str, Any]:
    campaign_id = str(campaign_config["campaign"]["id"])
    if decision.decision == "stop_campaign":
        state = {
            "campaign_id": campaign_id,
            "status": "campaign_completed",
            "phase": "completed",
            "stop_reason": decision.reason,
            "current_trial_id": trial_id,
            "updated_at": _now_iso(),
        }
        _write_json(_campaign_state_path(campaign_config), state)
        _upsert_init_logbook(campaign_config, trial_id=trial_id, decision=decision, trial_patch={})
        _print_analysis_block(
            stream,
            title=f"campaign initialization analysis: {trial_id}",
            markdown=decision.logbook_markdown or decision.reason,
            trial_patch={},
        )
        return state
    trial_patch = decision.trial_patch if decision.decision == "propose_trial" else {}
    _validate_trial_patch(
        loop_config,
        trial_patch or {},
        max_leaf_count=_max_patch_leaf_count(campaign_config),
    )
    _upsert_init_logbook(campaign_config, trial_id=trial_id, decision=decision, trial_patch=trial_patch or {})
    _print_analysis_block(
        stream,
        title=f"campaign initialization analysis: {trial_id}",
        markdown=decision.logbook_markdown or decision.reason,
        trial_patch=trial_patch or {},
    )
    return start_trial(
        campaign_config,
        loop_config,
        start_trial=trial_id,
        trial_patch=trial_patch or {},
        dry_run=False,
        allow_existing_trial_dir=allow_existing_trial_dir,
    )


def retry_initialization_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    *,
    client: Any | None,
    stream: TextIO,
) -> dict[str, Any]:
    trial_id = str(state["current_trial_id"])
    snapshot_value = state.get("init_snapshot_path")
    snapshot_path = Path(str(snapshot_value)) if snapshot_value else None
    snapshot = _read_json(snapshot_path) if snapshot_path and snapshot_path.exists() else collect_init_snapshot(campaign_config, loop_config)
    try:
        decision = request_init_decision(campaign_config, snapshot, trial_id, client=client)
    except Exception as exc:
        if _is_openai_credits_exhausted(exc):
            _mark_openai_credits_exhausted(campaign_config, state, exc, stream=stream)
        else:
            _mark_agent_decision_failed(campaign_config, state, exc, stream=stream)
        return state
    try:
        return _apply_init_decision(
            campaign_config,
            loop_config,
            trial_id=trial_id,
            decision=decision,
            stream=stream,
            allow_existing_trial_dir=True,
        )
    except ValueError as exc:
        state["agent_decision_error"] = f"{type(exc).__name__}: {exc}"
        _mark_agent_decision_failed(campaign_config, state, exc, stream=stream)
        return state


def apply_campaign_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    decision: CampaignDecision,
    *,
    dry_run: bool = False,
    stream: TextIO = sys.stdout,
) -> dict[str, Any]:
    if dry_run:
        return {"applied": False, "reason": "dry_run"}
    summary = summarize_trial(campaign_config, state)
    if decision.decision == "no_action":
        markdown = decision.logbook_markdown or decision.reason or "No further action recommended for this campaign decision."
        _upsert_campaign_logbook(campaign_config, summary, model_markdown=markdown)
        _print_analysis_block(
            stream,
            title=f"campaign result analysis: {summary.get('trial_id')}",
            markdown=markdown,
        )
        state["status"] = "analysis_completed"
        state["analysis_completed_at"] = _now_iso()
        state["updated_at"] = _now_iso()
        _persist_campaign_state(campaign_config, state)
        return {"applied": True, "updated": campaign_config["logbook"]["path"], "reason": "no_action_logged"}
    if decision.decision == "update_logbook":
        markdown = decision.logbook_markdown or decision.reason
        _upsert_campaign_logbook(campaign_config, summary, model_markdown=markdown)
        _print_analysis_block(
            stream,
            title=f"campaign result analysis: {summary.get('trial_id')}",
            markdown=markdown,
        )
        state["status"] = "analysis_completed"
        state["analysis_completed_at"] = _now_iso()
        state["updated_at"] = _now_iso()
        _persist_campaign_state(campaign_config, state)
        return {"applied": True, "updated": campaign_config["logbook"]["path"]}
    if decision.decision == "stop_campaign":
        markdown = decision.logbook_markdown or decision.reason
        _upsert_campaign_logbook(campaign_config, summary, model_markdown=markdown)
        _print_analysis_block(
            stream,
            title=f"campaign result analysis: {summary.get('trial_id')}",
            markdown=markdown,
        )
        state["status"] = "campaign_completed"
        state["stop_reason"] = decision.reason
        state["updated_at"] = _now_iso()
        _persist_campaign_state(campaign_config, state)
        return {"applied": True, "status": "campaign_completed"}
    if decision.decision == "propose_trial":
        _assert_trial_budget_available(campaign_config)
        _validate_trial_patch(
            loop_config,
            decision.trial_patch or {},
            max_leaf_count=_max_patch_leaf_count(campaign_config),
        )
        if state.get("launch_blocked_reason"):
            state["status"] = "launch_blocked"
            state["updated_at"] = _now_iso()
            _persist_campaign_state(campaign_config, state)
            print(
                f"campaign launch blocked trial={state.get('current_trial_id')} "
                f"reason={state['launch_blocked_reason']}",
                file=stream,
                flush=True,
            )
            return {"applied": False, "reason": state["launch_blocked_reason"]}
        markdown = decision.logbook_markdown or decision.reason
        _upsert_campaign_logbook(campaign_config, summary, model_markdown=markdown)
        _print_analysis_block(
            stream,
            title=f"campaign result analysis: {summary.get('trial_id')}",
            markdown=markdown,
            trial_patch=decision.trial_patch or {},
        )
        next_state = start_trial(
            campaign_config,
            loop_config,
            trial_patch=decision.trial_patch or {},
            dry_run=False,
        )
        return {"applied": True, "state": next_state}
    return {"applied": False, "reason": f"no local action for {decision.decision}"}


def _request_and_apply_campaign_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    summary: dict[str, Any],
    *,
    client: Any | None,
    stream: TextIO,
) -> tuple[dict[str, Any], str, str]:
    try:
        decision = request_campaign_decision(campaign_config, state, summary, client=client)
        result = apply_campaign_decision(
            campaign_config,
            loop_config,
            state,
            decision,
            dry_run=False,
            stream=stream,
        )
    except Exception as exc:
        if _is_openai_credits_exhausted(exc):
            _mark_openai_credits_exhausted(campaign_config, state, exc, stream=stream)
            return state, "openai_credits_exhausted", f"{type(exc).__name__}: {exc}"
        else:
            _mark_agent_decision_failed(campaign_config, state, exc, stream=stream)
            return state, "agent_decision_failed", f"{type(exc).__name__}: {exc}"

    if isinstance(result.get("state"), dict):
        state = result["state"]
    return state, decision.decision, decision.reason or "campaign decision applied"


def _print_status_line(
    stream: TextIO,
    *,
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    action: str,
    reason: str,
    next_poll: datetime | None,
) -> None:
    print(
        f"[{_now_iso()}] campaign poll campaign={campaign_config['campaign']['id']} "
        f"trial={state.get('current_trial_id')} stage={state.get('current_stage')} "
        f"status={state.get('status')} action={action} reason={reason}",
        file=stream,
        flush=True,
    )
    if next_poll is not None:
        print(f"next poll: {next_poll.isoformat(timespec='seconds')}", file=stream, flush=True)


def run_campaign(
    campaign_config: dict[str, Any],
    *,
    once: bool = False,
    dry_run: bool = False,
    start_trial_id: str | None = None,
    new_trial: bool = False,
    terminate_child_on_exit: bool = False,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stream: TextIO = sys.stdout,
) -> int:
    loop_config = _load_loop_config(campaign_config)
    campaign_config.setdefault("agent", dict(loop_config.get("agent", {})))
    campaign_config["agent"] = {**dict(loop_config.get("agent", {})), **dict(campaign_config.get("agent", {}))}
    _validate_chain(campaign_config, loop_config)
    if not dry_run:
        experiment_loop.validate_api_key({"agent": campaign_config["agent"]})
    lock_path = _acquire_campaign_lock(campaign_config) if not dry_run else None

    try:
        poll_seconds = int(campaign_config["campaign"].get("poll_seconds", campaign_config["agent"].get("poll_seconds", 3600)))
        state_path = _campaign_state_path(campaign_config)
        state = _read_json(state_path)
        if state and not new_trial and not dry_run and state.get("status") in RESTARTABLE_TERMINAL_STATUSES:
            print(
                f"previous campaign state is {state.get('status')}; starting a new campaign trial",
                file=stream,
                flush=True,
            )
            new_trial = True
        if new_trial and state and not dry_run:
            active_status = _active_stage_status_if_available(campaign_config, loop_config, state)
            if active_status is not None and active_status[0].get("process_running"):
                print(
                    f"refusing to start a new campaign trial because trial={state.get('current_trial_id')} "
                    f"stage={state.get('current_stage')} still has a running process",
                    file=stream,
                    flush=True,
                )
                return 1
        if new_trial or not state:
            if dry_run:
                state = start_trial(
                    campaign_config,
                    loop_config,
                    start_trial=start_trial_id,
                    dry_run=True,
                )
            else:
                state = initialize_campaign_trial(
                    campaign_config,
                    loop_config,
                    start_trial_id=start_trial_id,
                    client=client,
                    stream=stream,
                )
            if state.get("status") == "openai_credits_exhausted":
                return 1

        next_poll = datetime.now() if once else datetime.now() + timedelta(seconds=poll_seconds)
        print(
            f"campaign loop started campaign={campaign_config['campaign']['id']} poll_seconds={poll_seconds} state={state_path}",
            file=stream,
            flush=True,
        )
        print(f"next poll: {next_poll.isoformat(timespec='seconds')}", file=stream, flush=True)

        try:
            first_poll = True
            while True:
                if not once and not first_poll:
                    sleep_fn(float(poll_seconds))
                first_poll = False
                if not dry_run:
                    latest_state = _read_json(state_path)
                    if latest_state:
                        state = latest_state

                if state.get("status") == "openai_credits_exhausted":
                    action = "openai_credits_exhausted"
                    reason = "OpenAI API credits exhausted; terminating campaign loop."
                elif state.get("status") in {"campaign_completed", "analysis_completed", "terminated", "termination_race_stopped"}:
                    action = "no_action"
                    reason = f"campaign state is {state.get('status')}"
                elif state.get("status") == "terminating":
                    action = "wait_terminating"
                    reason = "campaign termination is in progress"
                elif state.get("status") == "agent_decision_failed":
                    if not dry_run and state.get("phase") == "initializing":
                        state = retry_initialization_decision(
                            campaign_config,
                            loop_config,
                            state,
                            client=client,
                            stream=stream,
                        )
                        action = "retry_init_decision"
                        reason = str(state.get("agent_decision_error") or "initialization decision retried")
                    elif not dry_run:
                        summary = summarize_trial(campaign_config, state)
                        state, action, reason = _request_and_apply_campaign_decision(
                            campaign_config,
                            loop_config,
                            state,
                            summary,
                            client=client,
                            stream=stream,
                        )
                    else:
                        action = "agent_decision_failed"
                        reason = state.get("agent_decision_error", "previous agent decision failed")
                else:
                    status, controller_status, controller_reason = _collect_stage_status(campaign_config, loop_config, state)
                    if not dry_run:
                        stage_loop_config = _loop_config_for_stage(
                            loop_config,
                            Path(state["current_trial_dir"]),
                            str(state["current_stage"]),
                            state["trial_configs"],
                        )
                        experiment_loop.update_controller_state(
                            stage_loop_config,
                            status,
                            controller_status,
                            controller_reason,
                            state_path=Path(state["stage_state_path"]),
                        )

                    if controller_status in {"running_progress", "running_wait"}:
                        action = "wait"
                        reason = controller_reason
                    elif controller_status == "completed":
                        state, action = _advance_stage_or_complete_trial(
                            campaign_config,
                            loop_config,
                            state,
                            status,
                            dry_run=dry_run,
                        )
                        reason = controller_reason
                        if state.get("status") == "trial_completed" and not dry_run:
                            summary = summarize_trial(campaign_config, state)
                            state, action, reason = _request_and_apply_campaign_decision(
                                campaign_config,
                                loop_config,
                                state,
                                summary,
                                client=client,
                                stream=stream,
                            )
                    elif controller_status == "running_stale":
                        state["status"] = "running_stale"
                        state["stale_reason"] = controller_reason
                        state["updated_at"] = _now_iso()
                        if not dry_run:
                            _persist_campaign_state(campaign_config, state)
                        action = "wait_stale"
                        reason = f"{controller_reason}; active process is still running, so new trial launch is blocked"
                    elif controller_status == "failed":
                        state["status"] = controller_status
                        state["failure_reason"] = controller_reason
                        state["updated_at"] = _now_iso()
                        if not dry_run:
                            _persist_campaign_state(campaign_config, state)
                            summary = summarize_trial(campaign_config, state)
                            state, action, reason = _request_and_apply_campaign_decision(
                                campaign_config,
                                loop_config,
                                state,
                                summary,
                                client=client,
                                stream=stream,
                            )
                        else:
                            action = controller_status
                            reason = controller_reason
                    else:
                        action = "no_action"
                        reason = controller_reason

                next_poll = None if once else datetime.now() + timedelta(seconds=poll_seconds)
                _print_status_line(
                    stream,
                    campaign_config=campaign_config,
                    state=state,
                    action=action,
                    reason=reason,
                    next_poll=next_poll,
                )
                if state.get("status") == "openai_credits_exhausted":
                    return 1
                if state.get("status") in {"campaign_completed", "analysis_completed", "terminated", "termination_race_stopped"}:
                    return 0
                if once:
                    return 0
        except KeyboardInterrupt:
            child_pid = state.get("active_launch_state", {}).get("pid")
            if terminate_child_on_exit and isinstance(child_pid, int) and experiment_loop._is_process_running(child_pid):
                os.killpg(child_pid, signal.SIGTERM)
            print(
                f"campaign loop stopped campaign={campaign_config['campaign']['id']} "
                f"trial={state.get('current_trial_id')} stage={state.get('current_stage')} state={state_path}",
                file=stream,
                flush=True,
            )
            return 130
    finally:
        if lock_path is not None:
            _release_campaign_lock(lock_path)


def status_command(campaign_config: dict[str, Any], *, stream: TextIO = sys.stdout) -> int:
    state = _read_json(_campaign_state_path(campaign_config))
    if not state:
        print(f"campaign status campaign={campaign_config['campaign']['id']} state=missing", file=stream, flush=True)
        return 0
    _print_status_line(
        stream,
        campaign_config=campaign_config,
        state=state,
        action="status",
        reason="status only",
        next_poll=None,
    )
    return 0


def _mark_stage_process_state(
    state: dict[str, Any],
    *,
    status: str,
    pid: int,
    reason: str,
    signal_name: str | None,
) -> None:
    stage_state_path = state.get("stage_state_path")
    if not stage_state_path:
        return
    path = Path(str(stage_state_path))
    stage_state = _read_json(path)
    timestamp_key = f"{status}_at"
    stage_state.update(
        {
            "status": status,
            timestamp_key: _now_iso(),
            f"{status}_pid": pid,
            "termination_reason": reason,
        }
    )
    if signal_name is not None:
        stage_state["termination_signal"] = signal_name
    _write_json(path, stage_state)
    run_status_path = stage_state.get("run_status_path")
    if run_status_path:
        run_status = _read_json(Path(str(run_status_path)))
        run_status.update(
            {
                "status": status,
                timestamp_key: stage_state[timestamp_key],
                f"{status}_pid": pid,
                "termination_reason": reason,
            }
        )
        if signal_name is not None:
            run_status["termination_signal"] = signal_name
        _write_json(Path(str(run_status_path)), run_status)


def _mark_campaign_process_state(
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    *,
    status: str,
    pid: int,
    reason: str,
    signal_name: str | None,
) -> None:
    timestamp = _now_iso()
    timestamp_key = f"{status}_at"
    state.update(
        {
            "status": status,
            "phase": status,
            timestamp_key: timestamp,
            f"{status}_pid": pid,
            "termination_reason": reason,
            "updated_at": timestamp,
        }
    )
    if signal_name is not None:
        state["termination_signal"] = signal_name
    if isinstance(state.get("active_launch_state"), dict):
        state["active_launch_state"]["status"] = status
        state["active_launch_state"][timestamp_key] = timestamp
    _persist_campaign_state(campaign_config, state)
    _mark_stage_process_state(state, status=status, pid=pid, reason=reason, signal_name=signal_name)


def _mark_campaign_terminated(
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    *,
    pid: int,
    reason: str,
    signal_name: str,
) -> None:
    _mark_campaign_process_state(
        campaign_config,
        state,
        status="terminated",
        pid=pid,
        reason=reason,
        signal_name=signal_name,
    )


def _mark_campaign_termination_race_stopped(
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    *,
    pid: int,
    reason: str,
) -> None:
    _mark_campaign_process_state(
        campaign_config,
        state,
        status="termination_race_stopped",
        pid=pid,
        reason=reason,
        signal_name=None,
    )


def terminate_campaign(
    campaign_config: dict[str, Any],
    *,
    reason: str = "terminated by campaign CLI",
    force_after: float | None = None,
    require_running: bool = False,
    stream: TextIO = sys.stdout,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    terminate_lock = _acquire_terminate_lock(campaign_config)
    try:
        loop_config = _load_loop_config(campaign_config)
        state_path = _campaign_state_path(campaign_config)
        state = _read_json(state_path)
        if not state:
            print(f"no running campaign found campaign={campaign_config['campaign']['id']} state=missing", file=stream, flush=True)
            return 1 if require_running else 0

        pid, _status, pid_source = _active_campaign_pid(campaign_config, loop_config, state)
        if pid is None:
            print(
                f"no running campaign found campaign={campaign_config['campaign']['id']} "
                f"trial={state.get('current_trial_id')} stage={state.get('current_stage')}",
                file=stream,
                flush=True,
            )
            return 1 if require_running else 0

        _mark_campaign_process_state(
            campaign_config,
            state,
            status="terminating",
            pid=pid,
            reason=reason,
            signal_name="SIGTERM",
        )
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            _mark_campaign_termination_race_stopped(campaign_config, state, pid=pid, reason=reason)
            print(
                f"campaign process already stopped campaign={campaign_config['campaign']['id']} pid={pid}; state marked termination_race_stopped",
                file=stream,
                flush=True,
            )
            return 0

        signal_name = "SIGTERM"
        if force_after is not None:
            sleep_fn(float(force_after))
            if experiment_loop._is_process_running(pid):
                os.killpg(pid, signal.SIGKILL)
                signal_name = "SIGKILL"

        _mark_campaign_terminated(campaign_config, state, pid=pid, reason=reason, signal_name=signal_name)
        print(
            f"campaign terminated campaign={campaign_config['campaign']['id']} "
            f"trial={state.get('current_trial_id')} stage={state.get('current_stage')} "
            f"pid={pid} pid_source={pid_source} signal={signal_name}",
            file=stream,
            flush=True,
        )
        return 0
    finally:
        _release_campaign_lock(terminate_lock)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect a ZebraFish experiment campaign.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the polling campaign loop.")
    run_parser.add_argument("--campaign", default="configs/experiment_campaigns/cnn_campaign.yaml", help="Campaign YAML config.")
    run_parser.add_argument("--poll-seconds", type=int, default=None, help="Override campaign poll interval.")
    run_parser.add_argument("--dry-run", action="store_true", help="Inspect without launching jobs, calling OpenAI, or writing files.")
    run_parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    run_parser.add_argument("--start-trial", default=None, help="Trial id to use when creating a fresh campaign trial.")
    run_parser.add_argument("--new-trial", action="store_true", help="Start a new trial even when campaign state already exists.")
    run_parser.add_argument(
        "--terminate-child-on-exit",
        action="store_true",
        help="On Ctrl-C, also terminate a child process launched by this campaign.",
    )

    status_parser = subparsers.add_parser("status", help="Print current campaign status without calling OpenAI.")
    status_parser.add_argument("--campaign", default="configs/experiment_campaigns/cnn_campaign.yaml", help="Campaign YAML config.")

    terminate_parser = subparsers.add_parser("terminate", help="Terminate the active training child for a campaign.")
    terminate_parser.add_argument("--campaign", default="configs/experiment_campaigns/cnn_campaign.yaml", help="Campaign YAML config.")
    terminate_parser.add_argument("--reason", default="terminated by campaign CLI", help="Reason recorded in campaign state.")
    terminate_parser.add_argument(
        "--force-after",
        type=float,
        default=None,
        help="Seconds to wait after SIGTERM before sending SIGKILL. Omit to avoid escalation.",
    )
    terminate_parser.add_argument(
        "--require-running",
        action="store_true",
        help="Return nonzero if no running process is found.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_campaign_config(args.campaign)
    if getattr(args, "poll_seconds", None) is not None:
        config["campaign"]["poll_seconds"] = int(args.poll_seconds)
    if args.command == "terminate" and args.force_after is not None and args.force_after <= 0:
        parser.error("--force-after must be greater than 0")
    if args.command == "status":
        return status_command(config)
    if args.command == "terminate":
        return terminate_campaign(
            config,
            reason=args.reason,
            force_after=args.force_after,
            require_running=args.require_running,
        )
    if args.command == "run":
        return run_campaign(
            config,
            once=args.once,
            dry_run=args.dry_run,
            start_trial_id=args.start_trial,
            new_trial=args.new_trial,
            terminate_child_on_exit=args.terminate_child_on_exit,
        )
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
