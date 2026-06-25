from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, TextIO

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ALLOWED_DECISIONS = {
    "patch_next_params",
    "launch_next",
    "no_action",
    "update_logbook",
}


@dataclass(frozen=True)
class AgentDecision:
    decision: str
    experiment: str
    reason: str
    logbook_markdown: str | None = None
    parameters_patch: dict[str, Any] | None = None


def load_loop_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{target} must contain a mapping")
    payload.setdefault("agent", {})
    payload["agent"].setdefault("poll_seconds", 18000)
    payload["agent"].setdefault("model", "gpt-5.3-codex")
    payload["agent"].setdefault("reasoning_effort", "medium")
    payload["agent"].setdefault("max_output_tokens", 2000)
    payload["agent"].setdefault("api_key_env", "OPENAI_API_KEY")
    payload.setdefault("controller", {})
    payload["controller"].setdefault("stale_polls_before_analysis", 2)
    if "experiments" not in payload or not isinstance(payload["experiments"], dict):
        raise ValueError(f"{target} must contain an experiments mapping")
    for experiment_cfg in payload["experiments"].values():
        if not isinstance(experiment_cfg, dict):
            continue
        kind = str(experiment_cfg.get("kind", ""))
        default_required = ["history", "summary_metrics", "checkpoint"]
        if kind == "finetune":
            default_required.extend(["confusion_matrices", "umap_pdf"])
        experiment_cfg.setdefault("required_completion_artifacts", default_required)
        experiment_cfg.setdefault("allowed_patch_paths", ["optimization_config", "loss_weight_config"])
        experiment_cfg.setdefault(
            "stale_polls_before_analysis",
            payload["controller"]["stale_polls_before_analysis"],
        )
    return payload


def validate_api_key(config: dict[str, Any]) -> str:
    env_name = str(config.get("agent", {}).get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is not set; refusing to run the OpenAI-backed agent loop.")
    return api_key


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _merge_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(path)
    payload.update(updates)
    _write_json(path, payload)
    return payload


def _is_process_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _latest_run_dir(artifact_root: str | Path) -> Path | None:
    runs_dir = Path(artifact_root) / "runs"
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _summarize_csv_tail(path: Path, *, rows: int = 5) -> dict[str, Any]:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {"path": str(path), "error": "pandas is unavailable"}
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "tail": frame.tail(rows).to_dict(orient="records"),
    }


def _max_mtime(paths: list[Path]) -> float | None:
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return max(mtimes) if mtimes else None


def _paths_with_suffix(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(root.rglob(f"*{suffix}"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime)


def collect_status(
    config: dict[str, Any],
    *,
    state_path: Path | None = None,
    state_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = state_path or Path(config.get("state", {}).get("path", "artifacts/agent_experiment_loop/state.json"))
    state = dict(state_override) if state_override is not None else _read_json(state_path)
    active_experiment = state.get("active_experiment")
    experiment_cfg = config["experiments"].get(active_experiment, {}) if active_experiment else {}
    if not experiment_cfg and config["experiments"]:
        first_name = next(iter(config["experiments"]))
        experiment_cfg = config["experiments"][first_name]
        active_experiment = first_name

    pid = state.get("pid")
    running = _is_process_running(int(pid)) if isinstance(pid, int) else False
    latest_run = _latest_run_dir(experiment_cfg.get("artifact_root", "")) if experiment_cfg else None
    log_path = Path(state["log_path"]) if state.get("log_path") else None
    run_status_path = Path(state["run_status_path"]) if state.get("run_status_path") else None
    run_status = _read_json(run_status_path) if run_status_path else {}
    if state.get("run_dir"):
        run_dir = Path(state["run_dir"])
        run_dir_source = "state"
    elif run_status.get("run_dir"):
        run_dir = Path(run_status["run_dir"])
        run_dir_source = "runner_status"
    else:
        run_dir = latest_run
        run_dir_source = "latest_fallback" if latest_run is not None else "none"

    artifact_summary: dict[str, Any] = {}
    if run_dir and run_dir.exists():
        history_candidates = sorted(run_dir.rglob("*history*.csv"), key=lambda path: path.stat().st_mtime)
        pdf_candidates = sorted(run_dir.rglob("*.pdf"), key=lambda path: path.stat().st_mtime)
        checkpoint_candidates = _paths_with_suffix(run_dir, ("_model_state.pt", "_encoder_state.pt"))
        confusion_candidates = sorted((run_dir / "confusion_matrices").glob("*.csv")) if (run_dir / "confusion_matrices").exists() else []
        umap_pdf_candidates = sorted(run_dir.rglob("*umap*.pdf"), key=lambda path: path.stat().st_mtime)
        metrics_candidates = sorted(
            list(run_dir.rglob("*summary_metrics*.csv")) + list(run_dir.rglob("*metrics*.json")),
            key=lambda path: path.stat().st_mtime,
        )
        artifact_summary = {
            "run_dir": str(run_dir),
            "run_dir_source": run_dir_source,
            "latest_history_csvs": [str(path) for path in history_candidates[-5:]],
            "latest_history_tail": _summarize_csv_tail(history_candidates[-1]) if history_candidates else None,
            "latest_pdfs": [str(path) for path in pdf_candidates[-8:]],
            "latest_metrics": [str(path) for path in metrics_candidates[-5:]],
            "checkpoints": [str(path) for path in checkpoint_candidates[-5:]],
            "confusion_matrices": [str(path) for path in confusion_candidates[-8:]],
            "umap_pdfs": [str(path) for path in umap_pdf_candidates[-5:]],
            "artifact_max_mtime": _max_mtime(
                history_candidates + pdf_candidates + metrics_candidates + checkpoint_candidates + confusion_candidates
            ),
            "has_history": bool(history_candidates),
            "has_metrics": bool(metrics_candidates),
            "has_checkpoint": bool(checkpoint_candidates),
            "has_confusion_matrices": bool(confusion_candidates),
            "has_umap_pdf": bool(umap_pdf_candidates),
        }

    return {
        "checked_at": _now_iso(),
        "state_path": str(state_path),
        "active_experiment": active_experiment,
        "pid": pid,
        "process_running": running,
        "state": state,
        "run_status": run_status,
        "artifacts": artifact_summary,
        "log_tail": _tail_text(log_path) if log_path else "",
    }


def classify_controller_status(config: dict[str, Any], status: dict[str, Any]) -> tuple[str, str]:
    state = status.get("state", {})
    run_status = status.get("run_status", {})
    artifacts = status.get("artifacts", {})
    log_tail = str(status.get("log_tail") or "")
    process_running = bool(status.get("process_running"))
    active_experiment = str(status.get("active_experiment") or "")
    experiment_cfg = config.get("experiments", {}).get(active_experiment, {})
    stale_limit = int(
        experiment_cfg.get(
            "stale_polls_before_analysis",
            config.get("controller", {}).get("stale_polls_before_analysis", 2),
        )
    )

    if process_running:
        current_mtime = artifacts.get("artifact_max_mtime")
        previous_mtime = state.get("last_artifact_max_mtime")
        stale_polls = int(state.get("stale_polls", 0) or 0)
        if current_mtime is not None and current_mtime != previous_mtime:
            return "running_progress", "process is running and artifacts advanced"
        if stale_polls >= stale_limit:
            return "running_stale", f"process is running but artifacts have not advanced for {stale_polls} poll(s)"
        return "running_wait", "process is running; waiting for the next artifact update"

    if run_status.get("status") == "completed" and _completion_requirements_met(config, status):
        return "completed", "runner reported completion and history artifacts exist"
    if run_status.get("status") == "failed":
        return "failed", f"runner reported failure: {run_status.get('error', 'unknown error')}"
    if "Traceback (most recent call last)" in log_tail or "ERROR" in log_tail:
        return "failed", "process is stopped and the log contains an error marker"
    if _completion_requirements_met(config, status):
        return "completed", "process is stopped and required completion artifacts exist"
    if status.get("pid") is not None:
        return "failed", "process is stopped before required final artifacts were found"
    return "idle", "no active process is recorded"


def _completion_requirements_met(config: dict[str, Any], status: dict[str, Any]) -> bool:
    active_experiment = str(status.get("active_experiment") or "")
    experiment_cfg = config.get("experiments", {}).get(active_experiment, {})
    required = list(experiment_cfg.get("required_completion_artifacts", ["history", "summary_metrics", "checkpoint"]))
    artifacts = status.get("artifacts", {})
    predicates = {
        "history": bool(artifacts.get("has_history")),
        "summary_metrics": bool(artifacts.get("has_metrics")),
        "checkpoint": bool(artifacts.get("has_checkpoint")),
        "confusion_matrices": bool(artifacts.get("has_confusion_matrices")),
        "umap_pdf": bool(artifacts.get("has_umap_pdf")),
    }
    return all(predicates.get(name, False) for name in required)


def _build_agent_prompt(config: dict[str, Any], status: dict[str, Any]) -> str:
    prompts = config.get("prompts", {})
    experiment_ids = ", ".join(str(name) for name in config.get("experiments", {}))
    return "\n\n".join(
        [
            "You are managing a local ZebraFish experiment loop.",
            f"Configured experiment ids: {experiment_ids}",
            f"Improvement goal:\n{prompts.get('improvement_goal', '')}",
            f"Analysis policy:\n{prompts.get('analysis_policy', '')}",
            f"Logbook rule:\n{prompts.get('update_logbook', '')}",
            f"Parameter patching rule:\n{prompts.get('patch_parameters', '')}",
            f"Decision schema:\n{prompts.get('status_decision', '')}",
            "Current status snapshot JSON:",
            json.dumps(status, indent=2, sort_keys=True),
        ]
    )


def parse_agent_decision(text: str) -> AgentDecision:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be a JSON object")
    decision = str(payload.get("decision", ""))
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported agent decision: {decision!r}")
    experiment = str(payload.get("experiment", ""))
    reason = str(payload.get("reason", "")).strip()
    parameters_patch = payload.get("parameters_patch")
    if parameters_patch is not None and not isinstance(parameters_patch, dict):
        raise ValueError("parameters_patch must be a mapping when provided")
    return AgentDecision(
        decision=decision,
        experiment=experiment,
        reason=reason,
        logbook_markdown=payload.get("logbook_markdown"),
        parameters_patch=parameters_patch,
    )


def request_agent_decision(config: dict[str, Any], status: dict[str, Any], *, client: Any | None = None) -> AgentDecision:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    response = client.responses.create(
        model=config["agent"]["model"],
        reasoning={"effort": config["agent"]["reasoning_effort"]},
        max_output_tokens=int(config["agent"].get("max_output_tokens", 2000)),
        input=_build_agent_prompt(config, status),
    )
    return parse_agent_decision(response.output_text)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml is not None else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(payload, sort_keys=False) if yaml is not None else json.dumps(payload, indent=2)
    path.write_text(rendered, encoding="utf-8")


def _patch_leaf_paths(patch: dict[str, Any], *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            paths.extend(_patch_leaf_paths(value, prefix=path))
        else:
            paths.append(path)
    return paths


def _validate_patch_allowed(experiment_cfg: dict[str, Any], patch: dict[str, Any]) -> None:
    allowed_paths = [str(path) for path in experiment_cfg.get("allowed_patch_paths", [])]
    rejected = [
        path
        for path in _patch_leaf_paths(patch)
        if not any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)
    ]
    if rejected:
        raise ValueError(
            "parameters_patch contains non-allowlisted path(s): "
            + ", ".join(sorted(rejected))
        )


def _upsert_marked_block(path: Path, marker: str, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else "# Experiments Logbook\n"
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{markdown.strip()}\n{end}\n"
    if start in current and end in current:
        before = current.split(start, 1)[0]
        after = current.split(end, 1)[1]
        updated = before.rstrip() + "\n\n" + block + after.lstrip()
    else:
        updated = current.rstrip() + "\n\n" + block
    path.write_text(updated, encoding="utf-8")


def apply_agent_decision(
    config: dict[str, Any],
    decision: AgentDecision,
    status: dict[str, Any],
    *,
    dry_run: bool = False,
    state_path: Path | None = None,
) -> dict[str, Any]:
    if dry_run or decision.decision == "no_action":
        return {"applied": False, "reason": "dry_run" if dry_run else "no_action"}
    if decision.experiment not in config["experiments"]:
        raise ValueError(f"Decision referenced unknown experiment {decision.experiment!r}")

    experiment_cfg = config["experiments"][decision.experiment]
    if decision.decision == "patch_next_params":
        _validate_patch_allowed(experiment_cfg, decision.parameters_patch or {})
        params_path = Path(experiment_cfg["params_yaml"])
        current = _read_yaml(params_path)
        _write_yaml(params_path, _deep_merge(current, decision.parameters_patch or {}))
        return {"applied": True, "updated": str(params_path)}

    if decision.decision == "update_logbook":
        logbook_path = Path(config.get("logbook", {}).get("path", "EXPERIMENTS_LOGBOOK.md"))
        run_dir = status.get("artifacts", {}).get("run_dir") or decision.experiment
        marker = f"agent-loop:{decision.experiment}:{Path(str(run_dir)).name}"
        markdown = decision.logbook_markdown or f"## Agent Loop Status: {Path(str(run_dir)).name}\n\n{decision.reason}"
        _upsert_marked_block(logbook_path, marker, markdown)
        return {"applied": True, "updated": str(logbook_path)}

    if decision.decision == "launch_next":
        controller_status = str(status.get("controller_status") or "")
        if controller_status != "completed":
            return {"applied": False, "reason": "launch_next requires deterministic completion"}
        active_experiment = str(status.get("active_experiment") or "")
        if decision.experiment != active_experiment:
            return {"applied": False, "reason": "launch_next must reference the completed active experiment"}
        next_experiment = experiment_cfg.get("next")
        if not next_experiment:
            return {"applied": False, "reason": "no next experiment configured"}
        state = launch_experiment(config, str(next_experiment), dry_run=False)
        return {"applied": True, "state": state}

    return {"applied": False, "reason": f"no local action for {decision.decision}"}


def launch_experiment(config: dict[str, Any], experiment: str, *, dry_run: bool = False) -> dict[str, Any]:
    if experiment not in config["experiments"]:
        raise ValueError(f"Unknown experiment {experiment!r}")
    experiment_cfg = config["experiments"][experiment]
    state_path = Path(config.get("state", {}).get("path", "artifacts/agent_experiment_loop/state.json"))
    log_dir = Path(config.get("state", {}).get("log_dir", "artifacts/agent_experiment_loop/logs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{experiment}_{timestamp}.log"
    run_status_path = log_dir / f"{experiment}_{timestamp}.run_status.json"
    state = {
        "active_experiment": experiment,
        "status": "dry_run" if dry_run else "running",
        "runner": experiment_cfg["runner"],
        "params_yaml": experiment_cfg["params_yaml"],
        "log_path": str(log_path),
        "run_status_path": str(run_status_path),
        "started_at": _now_iso(),
    }
    if dry_run:
        return state

    log_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(experiment_cfg["runner"]), "--config", str(experiment_cfg["params_yaml"])]
    child_env = {
        **os.environ,
        "ZF_AGENT_RUN_STATUS_PATH": str(run_status_path),
        "ZF_AGENT_EXPERIMENT_NAME": experiment,
    }
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )
    state["pid"] = int(process.pid)
    _write_json(state_path, state)
    return state


def update_controller_state(
    config: dict[str, Any],
    status: dict[str, Any],
    controller_status: str,
    reason: str,
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    state_path = state_path or Path(config.get("state", {}).get("path", "artifacts/agent_experiment_loop/state.json"))
    artifacts = status.get("artifacts", {})
    current_mtime = artifacts.get("artifact_max_mtime")
    previous_state = status.get("state", {})
    stale_polls = int(previous_state.get("stale_polls", 0) or 0)
    if status.get("process_running"):
        if current_mtime is not None and current_mtime != previous_state.get("last_artifact_max_mtime"):
            stale_polls = 0
        else:
            stale_polls += 1
    updates = {
        "controller_status": controller_status,
        "controller_reason": reason,
        "last_poll_at": _now_iso(),
        "stale_polls": stale_polls,
    }
    if current_mtime is not None:
        updates["last_artifact_max_mtime"] = current_mtime
    run_dir = artifacts.get("run_dir")
    if run_dir:
        updates["run_dir"] = run_dir
    if controller_status in {"completed", "failed"}:
        updates["status"] = controller_status
        updates[f"{controller_status}_at"] = _now_iso()
    return _merge_json(state_path, updates)


def _print_status_line(
    stream: TextIO,
    *,
    status: dict[str, Any],
    decision: AgentDecision | None,
    next_poll: datetime | None,
) -> None:
    reason = decision.reason if decision is not None else "status only"
    decision_name = decision.decision if decision is not None else "none"
    artifacts = status.get("artifacts", {})
    run_dir = artifacts.get("run_dir") or "unknown"
    run_dir_source = artifacts.get("run_dir_source") or "none"
    controller_status = status.get("controller_status") or "unknown"
    print(
        f"[{_now_iso()}] poll experiment={status.get('active_experiment')} "
        f"running={status.get('process_running')} controller={controller_status} "
        f"decision={decision_name} reason={reason} run_source={run_dir_source} run={run_dir}",
        file=stream,
        flush=True,
    )
    if next_poll is not None:
        print(f"next poll: {next_poll.isoformat(timespec='seconds')}", file=stream, flush=True)


def run_loop(
    config: dict[str, Any],
    *,
    start_at: str | None = None,
    once: bool = False,
    dry_run: bool = False,
    terminate_child_on_exit: bool = False,
    new_run: bool = False,
    reset_state: bool = False,
    resume: bool = False,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stream: TextIO = sys.stdout,
) -> int:
    state_path = Path(config.get("state", {}).get("path", "artifacts/agent_experiment_loop/state.json"))
    if not dry_run:
        validate_api_key(config)
    if reset_state and not dry_run:
        state_path.unlink(missing_ok=True)

    state = _read_json(state_path)
    state_status = str(state.get("status") or "")
    should_start_new = bool(
        start_at
        and (
            new_run
            or not state.get("active_experiment")
            or (state_status in {"completed", "failed"} and not resume)
        )
    )
    if should_start_new:
        state = launch_experiment(config, start_at, dry_run=dry_run)
    elif not state.get("active_experiment") and not dry_run:
        first_experiment = start_at or next(iter(config["experiments"]))
        state = launch_experiment(config, first_experiment, dry_run=False)
    elif dry_run and start_at:
        state = launch_experiment(config, start_at, dry_run=True)

    poll_seconds = int(config.get("agent", {}).get("poll_seconds", 18000))
    next_poll = datetime.now() if once else datetime.now() + timedelta(seconds=poll_seconds)
    print(
        f"agent loop started experiment={state.get('active_experiment')} poll_seconds={poll_seconds} "
        f"state={state_path}",
        file=stream,
        flush=True,
    )
    print(f"next poll: {next_poll.isoformat(timespec='seconds')}", file=stream, flush=True)

    child_pid = state.get("pid") if isinstance(state.get("pid"), int) else None
    try:
        while True:
            if not once:
                sleep_fn(float(poll_seconds))
            status = collect_status(
                config,
                state_path=state_path,
                state_override=state if dry_run and state.get("active_experiment") else None,
            )
            controller_status, controller_reason = classify_controller_status(config, status)
            status["controller_status"] = controller_status
            status["controller_reason"] = controller_reason
            if not dry_run:
                update_controller_state(config, status, controller_status, controller_reason, state_path=state_path)
            if dry_run:
                decision = AgentDecision(
                    decision="no_action",
                    experiment=str(status.get("active_experiment") or start_at or next(iter(config["experiments"]))),
                    reason="dry-run status inspection only",
                )
            elif controller_status in {"running_progress", "running_wait"}:
                decision = AgentDecision(
                    decision="no_action",
                    experiment=str(status.get("active_experiment") or next(iter(config["experiments"]))),
                    reason=controller_reason,
                )
            else:
                decision = request_agent_decision(config, status, client=client)
                result = apply_agent_decision(config, decision, status, dry_run=False, state_path=state_path)
                if isinstance(result.get("state"), dict):
                    state = result["state"]
                    child_pid = state.get("pid") if isinstance(state.get("pid"), int) else child_pid
            next_poll = None if once else datetime.now() + timedelta(seconds=poll_seconds)
            _print_status_line(stream, status=status, decision=decision, next_poll=next_poll)
            if once:
                return 0
    except KeyboardInterrupt:
        if terminate_child_on_exit and child_pid and _is_process_running(child_pid):
            os.killpg(child_pid, signal.SIGTERM)
        status = collect_status(config, state_path=state_path)
        print(
            f"agent loop stopped experiment={status.get('active_experiment')} "
            f"running={status.get('process_running')} state={state_path}",
            file=stream,
            flush=True,
        )
        return 130


def status_command(config: dict[str, Any], *, stream: TextIO = sys.stdout) -> int:
    status = collect_status(config)
    controller_status, controller_reason = classify_controller_status(config, status)
    status["controller_status"] = controller_status
    status["controller_reason"] = controller_reason
    _print_status_line(stream, status=status, decision=None, next_poll=None)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect the ZebraFish agent experiment loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the polling agent loop.")
    run_parser.add_argument("--config", default="configs/agent_experiment_loop.yaml", help="Agent loop YAML config.")
    run_parser.add_argument("--poll-seconds", type=int, default=None, help="Override config poll interval.")
    run_parser.add_argument("--dry-run", action="store_true", help="Inspect without launching jobs, calling OpenAI, or writing files.")
    run_parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    run_parser.add_argument("--start-at", default=None, help="Experiment id to launch if no state exists.")
    run_parser.add_argument("--new-run", action="store_true", help="Launch a fresh run for --start-at even if state exists.")
    run_parser.add_argument("--reset-state", action="store_true", help="Delete harness state before starting.")
    run_parser.add_argument("--resume", action="store_true", help="Keep completed/failed state instead of replacing it when --start-at is given.")
    run_parser.add_argument(
        "--terminate-child-on-exit",
        action="store_true",
        help="On Ctrl-C, also terminate a child process launched by this harness.",
    )

    status_parser = subparsers.add_parser("status", help="Print current loop status without calling OpenAI.")
    status_parser.add_argument("--config", default="configs/agent_experiment_loop.yaml", help="Agent loop YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_loop_config(args.config)
    if getattr(args, "poll_seconds", None) is not None:
        config["agent"]["poll_seconds"] = int(args.poll_seconds)
    if getattr(args, "start_at", None) is not None and args.start_at not in config["experiments"]:
        parser.error(f"Unknown experiment {args.start_at!r}; configured ids: {', '.join(config['experiments'])}")
    if args.command == "status":
        return status_command(config)
    if args.command == "run":
        return run_loop(
            config,
            start_at=args.start_at,
            once=args.once,
            dry_run=args.dry_run,
            terminate_child_on_exit=args.terminate_child_on_exit,
            new_run=args.new_run,
            reset_state=args.reset_state,
            resume=args.resume,
        )
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
