from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any, Callable, TextIO

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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    campaign.setdefault("poll_seconds", 18000)
    campaign.setdefault("trial_budget", 20)
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
    objective.setdefault("fallback_metrics", ["balanced_accuracy", "accuracy", "roc_auc_ovr_macro"])
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
    return f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _trial_dir(config: dict[str, Any], trial_id: str) -> Path:
    return _campaign_root(config) / "trials" / trial_id


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
        stage_statuses[stage] = {
            "controller_status": controller_status,
            "controller_reason": controller_reason,
            "latest_artifacts": status.get("artifacts", {}),
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
            write_yaml_mapping(target, patched)
        else:
            shutil.copy2(source, target)
        copied[stage] = str(target)
    return copied


def _trial_count(campaign_config: dict[str, Any]) -> int:
    trials_csv = Path(campaign_config["artifacts"]["trials_csv"])
    if not trials_csv.exists():
        return 0
    with trials_csv.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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


def _validate_trial_patch(loop_config: dict[str, Any], trial_patch: dict[str, Any]) -> None:
    for experiment, patch in trial_patch.items():
        if experiment not in loop_config["experiments"]:
            raise ValueError(f"trial_patch references unknown experiment {experiment!r}")
        if not isinstance(patch, dict):
            raise ValueError(f"trial_patch for {experiment} must be a mapping")
        allowed = [str(path) for path in loop_config["experiments"][experiment].get("allowed_patch_paths", [])]
        rejected = [
            path
            for path in _patch_leaf_paths(patch)
            if not any(path == allowed_path or path.startswith(f"{allowed_path}.") for allowed_path in allowed)
        ]
        if rejected:
            raise ValueError(
                f"trial_patch for {experiment} contains non-allowlisted path(s): "
                + ", ".join(sorted(rejected))
            )


def start_trial(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    *,
    start_trial: str | None = None,
    trial_patch: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_chain(campaign_config, loop_config)
    _validate_trial_patch(loop_config, trial_patch or {})
    if not dry_run:
        _assert_trial_budget_available(campaign_config)
    campaign_id = str(campaign_config["campaign"]["id"])
    trial_id = _trial_id(campaign_id, start_trial)
    trial_dir = _trial_dir(campaign_config, trial_id)
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
        launched = experiment_loop.launch_experiment(
            _loop_config_for_stage(loop_config, trial_dir, stages[0], trial_configs),
            stages[0],
            dry_run=False,
        )

    state = {
        "campaign_id": campaign_id,
        "status": "running",
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
        "active_launch_state": launched,
    }
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


def wire_pretrain_checkpoint_into_finetune_config(
    finetune_config_path: str | Path,
    pretrain_run_dir: str | Path,
) -> Path:
    pretrain_config = _find_pretraining_config(Path(pretrain_run_dir))
    if pretrain_config is None:
        raise FileNotFoundError(f"No *_config.yaml found in pretrain run folder {pretrain_run_dir}")
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
        state["objective_metric_path"] = summary.get("metrics_path")
        state["updated_at"] = _now_iso()
        if not dry_run:
            _persist_campaign_state(campaign_config, state)
            _write_trial_outputs(campaign_config, state, summary)
        return state, "trial_completed"

    next_stage = stages[next_index]
    if stage_runs.get(stage):
        wire_pretrain_checkpoint_into_finetune_config(state["trial_configs"][next_stage], stage_runs[stage])
    trial_dir = Path(state["current_trial_dir"])
    next_loop_config = _loop_config_for_stage(loop_config, trial_dir, next_stage, state["trial_configs"])
    launched = experiment_loop.launch_experiment(next_loop_config, next_stage, dry_run=dry_run)
    state.update(
        {
            "status": "running",
            "current_stage_index": next_index,
            "current_stage": next_stage,
            "stage_state_path": str(_stage_state_path(trial_dir, next_stage)),
            "active_launch_state": launched,
            "updated_at": _now_iso(),
        }
    )
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
    candidates = [primary_metric] + [str(metric) for metric in objective.get("fallback_metrics", [])]
    selected_metric = None
    selected_value = None
    for metric in candidates:
        key = f"{target}.{metric}"
        if key in metrics:
            selected_metric = key
            selected_value = metrics[key]
            break
    minimums = {str(key): float(value) for key, value in objective.get("minimums", {}).items()}
    guardrail_failures = {
        key: {"minimum": minimum, "actual": metrics.get(key)}
        for key, minimum in minimums.items()
        if metrics.get(key) is None or float(metrics[key]) < minimum
    }
    return {
        "score": selected_value,
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
    leaderboard = sorted(
        [record for record in trial_records if record.get("score") not in ("", None)],
        key=lambda record: float(record["score"]),
        reverse=bool(campaign_config["objective"].get("maximize", True)),
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
        "selected_metric": summary.get("selected_metric") or "",
        "guardrail_passed": summary.get("guardrail_passed"),
        "trial_dir": summary.get("trial_dir"),
        "metrics_path": summary.get("metrics_path") or "",
        "updated_at": _now_iso(),
    }
    by_trial = {str(record.get("trial_id")): record for record in previous}
    by_trial[str(current["trial_id"])] = current
    return list(by_trial.values())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "campaign_id",
        "trial_id",
        "status",
        "score",
        "selected_metric",
        "guardrail_passed",
        "trial_dir",
        "metrics_path",
        "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


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
        "### Next Things To Try",
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
    if trial_patch is not None and not isinstance(trial_patch, dict):
        raise ValueError("trial_patch must be a mapping when provided")
    return CampaignDecision(
        decision=decision,
        reason=str(payload.get("reason", "")).strip(),
        trial_patch=trial_patch,
        logbook_markdown=payload.get("logbook_markdown"),
    )


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
            "Inspect the current status, recent logs, and logbook tail. Write the next things to try before the first stage launches.",
            "The controller will launch the first campaign stage after this decision unless you return stop_campaign.",
            f"Proposed trial id: {trial_id}",
            f"Objective:\n{json.dumps(config.get('objective', {}), indent=2, sort_keys=True)}",
            f"Campaign policy:\n{prompts.get('analysis_policy', '')}",
            f"Logbook rule:\n{prompts.get('update_logbook', '')}",
            f"Parameter patching rule:\n{prompts.get('patch_parameters', '')}",
            f"Decision schema:\n{prompts.get('decision_schema', '')}",
            "Return propose_trial when you have a concrete first trial patch. Return update_logbook or no_action only if the baseline config should be launched unchanged.",
            "Initialization snapshot JSON:",
            json.dumps(snapshot, indent=2, sort_keys=True),
        ]
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
        input=_build_campaign_prompt(config, state, summary),
    )
    return parse_campaign_decision(response.output_text)


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
        input=_build_init_prompt(config, snapshot, trial_id),
    )
    return parse_campaign_decision(response.output_text)


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
    decision = request_init_decision(campaign_config, snapshot, trial_id, client=client)
    if decision.decision == "stop_campaign":
        state = {
            "campaign_id": campaign_id,
            "status": "campaign_completed",
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
    _validate_trial_patch(loop_config, trial_patch or {})
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
    )


def apply_campaign_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    decision: CampaignDecision,
    *,
    dry_run: bool = False,
    stream: TextIO = sys.stdout,
) -> dict[str, Any]:
    if dry_run or decision.decision == "no_action":
        return {"applied": False, "reason": "dry_run" if dry_run else "no_action"}
    summary = summarize_trial(campaign_config, state)
    if decision.decision == "update_logbook":
        markdown = decision.logbook_markdown or decision.reason
        _upsert_campaign_logbook(campaign_config, summary, model_markdown=markdown)
        _print_analysis_block(
            stream,
            title=f"campaign result analysis: {summary.get('trial_id')}",
            markdown=markdown,
        )
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
        _validate_trial_patch(loop_config, decision.trial_patch or {})
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

    poll_seconds = int(campaign_config["campaign"].get("poll_seconds", campaign_config["agent"].get("poll_seconds", 18000)))
    state_path = _campaign_state_path(campaign_config)
    state = _read_json(state_path)
    if not state or state.get("status") in {"trial_completed", "campaign_completed", "failed"}:
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

    next_poll = datetime.now() if once else datetime.now() + timedelta(seconds=poll_seconds)
    print(
        f"campaign loop started campaign={campaign_config['campaign']['id']} poll_seconds={poll_seconds} state={state_path}",
        file=stream,
        flush=True,
    )
    print(f"next poll: {next_poll.isoformat(timespec='seconds')}", file=stream, flush=True)

    try:
        while True:
            if not once:
                sleep_fn(float(poll_seconds))

            if state.get("status") == "campaign_completed":
                action = "no_action"
                reason = "campaign already completed"
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
                        decision = request_campaign_decision(campaign_config, state, summary, client=client)
                        apply_campaign_decision(campaign_config, loop_config, state, decision, dry_run=False, stream=stream)
                        action = decision.decision
                        reason = decision.reason or reason
                elif controller_status in {"failed", "running_stale"}:
                    state["status"] = controller_status
                    state["failure_reason"] = controller_reason
                    state["updated_at"] = _now_iso()
                    if not dry_run:
                        _persist_campaign_state(campaign_config, state)
                        summary = summarize_trial(campaign_config, state)
                        decision = request_campaign_decision(campaign_config, state, summary, client=client)
                        apply_campaign_decision(campaign_config, loop_config, state, decision, dry_run=False, stream=stream)
                        action = decision.decision
                        reason = decision.reason or controller_reason
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect a ZebraFish experiment campaign.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the polling campaign loop.")
    run_parser.add_argument("--campaign", default="configs/experiment_campaigns/cnn_campaign.yaml", help="Campaign YAML config.")
    run_parser.add_argument("--poll-seconds", type=int, default=None, help="Override campaign poll interval.")
    run_parser.add_argument("--dry-run", action="store_true", help="Inspect without launching jobs, calling OpenAI, or writing files.")
    run_parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    run_parser.add_argument("--start-trial", default=None, help="Trial id to use when creating a fresh campaign trial.")
    run_parser.add_argument(
        "--terminate-child-on-exit",
        action="store_true",
        help="On Ctrl-C, also terminate a child process launched by this campaign.",
    )

    status_parser = subparsers.add_parser("status", help="Print current campaign status without calling OpenAI.")
    status_parser.add_argument("--campaign", default="configs/experiment_campaigns/cnn_campaign.yaml", help="Campaign YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_campaign_config(args.campaign)
    if getattr(args, "poll_seconds", None) is not None:
        config["campaign"]["poll_seconds"] = int(args.poll_seconds)
    if args.command == "status":
        return status_command(config)
    if args.command == "run":
        return run_campaign(
            config,
            once=args.once,
            dry_run=args.dry_run,
            start_trial_id=args.start_trial,
            terminate_child_on_exit=args.terminate_child_on_exit,
        )
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
