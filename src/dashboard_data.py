"""Read-only normalized data adapter for the mission-control dashboard.

This module deliberately keeps all dashboard derivation outside the React
components. It reads the scientific YAML and campaign artifacts, but never
launches processes or writes state.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from src.observation_engine import read_history, read_summary_metrics
from src.scientific_dashboard import _graphviz_svg, build_reasoning_graph
from src.scientific_state import load_state

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - the project environment provides PyYAML.
    yaml = None


CAMPAIGN_CONFIGS = {
    "cnn": "configs/experiment_campaigns/cnn_campaign.yaml",
    "transformer": "configs/experiment_campaigns/transformer_campaign.yaml",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat(timespec="seconds")


def _read_json(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None or not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Could not parse {path}: {exc}"
    return (value if isinstance(value, dict) else {}), None


def _read_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"Missing YAML file: {path}"
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml is not None else json.loads(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, f"Could not parse {path}: {exc}"
    return (value if isinstance(value, dict) else {}), None


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def resolve_campaign(root: Path, campaign: str) -> tuple[str, Path]:
    config_value = CAMPAIGN_CONFIGS.get(campaign, campaign)
    config_path = _resolve(root, config_value)
    if config_path is None or not config_path.exists():
        for alias, relative in CAMPAIGN_CONFIGS.items():
            candidate = _resolve(root, relative)
            if candidate and candidate.exists():
                config, _ = _read_yaml(candidate)
                if config.get("campaign", {}).get("id") == campaign:
                    return alias, candidate
        raise FileNotFoundError(f"Unknown campaign or missing config: {campaign}")
    config, error = _read_yaml(config_path)
    if error:
        raise ValueError(error)
    return campaign if campaign in CAMPAIGN_CONFIGS else str(config.get("campaign", {}).get("id", campaign)), config_path


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _short_text(value: Any, limit: int = 240) -> str:
    if isinstance(value, dict):
        for key in ("statement", "question", "purpose", "title", "description"):
            if value.get(key):
                return _short_text(value[key], limit)
        return ""
    if isinstance(value, list):
        return "; ".join(_short_text(item, limit) for item in value if _short_text(item, limit))[:limit]
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _entity_text(entity: dict[str, Any]) -> str:
    return str(entity.get("statement") or entity.get("question") or entity.get("title") or entity.get("purpose") or "")


def _file_mtime(path: Path | None) -> float | None:
    try:
        return path.stat().st_mtime if path and path.exists() else None
    except OSError:
        return None


def _is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _timestamp_seconds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _latest_history(run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.exists():
        return None
    candidates = [path for path in run_dir.rglob("*.history.csv") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _latest_summary(run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.exists():
        return None
    candidates = [
        path
        for path in run_dir.rglob("*.csv")
        if path.is_file() and any(token in path.name.lower() for token in ("summary", "metric", "leaderboard"))
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _metric_descriptor(rows: list[dict[str, float]], primary_metric: str) -> dict[str, Any]:
    primary_keys = [primary_metric, primary_metric.replace(".", "_"), f"val_{primary_metric}"]
    primary_key = next((key for key in primary_keys if any(key in row for row in rows)), None)
    diagnostic_key = next((key for key in ("val_loss", "val_self_probe_loss", "train_loss") if any(key in row for row in rows)), None)
    if primary_key:
        return {
            "requested_metric": primary_metric,
            "display_metric": primary_key,
            "role": "primary",
            "available": True,
            "direction": "higher_is_better",
            "source": "history_csv",
            "fallback_reason": None,
        }
    return {
        "requested_metric": primary_metric,
        "display_metric": diagnostic_key,
        "role": "diagnostic" if diagnostic_key else "unavailable",
        "available": bool(diagnostic_key),
        "direction": "lower_is_better" if diagnostic_key and "loss" in diagnostic_key.lower() else "unknown",
        "source": "history_csv" if diagnostic_key else None,
        "fallback_reason": f"{primary_metric} is not recorded for this stage; showing {diagnostic_key}." if diagnostic_key else f"{primary_metric} is not recorded for this stage.",
    }


def _metric_series(rows: list[dict[str, float]], primary_metric: str, descriptor: dict[str, Any] | None = None) -> list[dict[str, float | int | None]]:
    if not rows:
        return []
    descriptor = descriptor or _metric_descriptor(rows, primary_metric)
    primary_key = descriptor.get("display_metric")
    validation_key = next((key for key in ("val_loss", "val_self_probe_loss") if any(key in row for row in rows)), None)
    train_key = next((key for key in ("train_loss", "train_self_probe_loss") if any(key in row for row in rows)), None)
    series: list[dict[str, float | int | None]] = []
    for index, row in enumerate(rows, start=1):
        series.append(
            {
                "step": index,
                "epoch": index,
                "primary": row.get(primary_key) if primary_key else None,
                "train": row.get(train_key) if train_key else None,
                "validation": row.get(validation_key) if validation_key else None,
            }
        )
    return series


def _latest_metric(series: list[dict[str, Any]], key: str = "primary") -> float | None:
    values = [_float(row.get(key)) for row in series]
    values = [value for value in values if value is not None]
    return values[-1] if values else None


def _best_metric(series: list[dict[str, Any]], key: str = "primary", maximize: bool = True) -> float | None:
    values = [_float(row.get(key)) for row in series]
    values = [value for value in values if value is not None]
    return (max(values) if maximize else min(values)) if values else None


def _references(state: dict[str, Any], entity_id: str) -> list[dict[str, Any]]:
    entities = state.get("entities", {})
    result: list[dict[str, Any]] = []
    for collection, records in entities.items():
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            if record_id == entity_id:
                continue
            encoded = json.dumps(record, sort_keys=True, default=str)
            if entity_id in encoded:
                result.append({"collection": collection, "id": record_id})
    return result[:30]


def _classify_observation(state: dict[str, Any], observation_id: str, hypothesis_id: str | None) -> tuple[str, str, str]:
    for relation in state.get("relations", []):
        if relation.get("type") not in {"supports", "contradicts"}:
            continue
        endpoints = {str(relation.get("source")), str(relation.get("target"))}
        if observation_id in endpoints and (not hypothesis_id or hypothesis_id in endpoints):
            direction = "supporting" if relation["type"] == "supports" else "contradicting"
            return direction, "explicit_relation", f"The state records a {relation['type']} relation for this observation and hypothesis."
    observation = state.get("entities", {}).get("observations", {}).get(observation_id, {})
    direction = observation.get("direction") or observation.get("evidence_direction")
    if direction in {"supports", "supporting"}:
        return "supporting", "explicit_observation_field", "The observation explicitly marks itself as supporting."
    if direction in {"contradicts", "contradicting"}:
        return "contradicting", "explicit_observation_field", "The observation explicitly marks itself as contradicting."
    if direction == "inconclusive":
        return "inconclusive", "explicit_observation_field", "The observation explicitly marks itself as inconclusive."
    return "unclassified", "unavailable", "No explicit support, contradiction, or inconclusive classification is recorded."


def _evidence(state: dict[str, Any], hypothesis_id: str | None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    filters = filters or {}
    result: list[dict[str, Any]] = []
    observations = state.get("entities", {}).get("observations", {})
    for observation_id, observation in observations.items():
        direction, classification_source, explanation = _classify_observation(state, observation_id, hypothesis_id)
        source_ids = observation.get("source_experiments", [])
        reliability = _float(observation.get("reliability"))
        evidence_strength = _float(observation.get("evidence_strength"))
        confidence = max(value for value in (reliability, evidence_strength) if value is not None) if any(value is not None for value in (reliability, evidence_strength)) else None
        if filters.get("confidence_min") is not None and (confidence is None or confidence < filters["confidence_min"]):
            continue
        created_at = _iso(observation.get("created_at"))
        if filters.get("time_from") and created_at and created_at < str(filters["time_from"]):
            continue
        if filters.get("time_to") and created_at and created_at > str(filters["time_to"]):
            continue
        result.append(
            {
                "id": observation_id,
                "type": observation.get("type", "observation"),
                "summary": _short_text(observation.get("statement", "")),
                "statement": observation.get("statement", ""),
                "direction": direction,
                "classification_source": classification_source,
                "explanation": explanation,
                "source_experiments": source_ids,
                "reliability": reliability,
                "evidence_strength": evidence_strength,
                "confidence": confidence,
                "created_at": created_at,
                "measurements": observation.get("measurements", {}),
                "detection": observation.get("detection", {}),
                "references": _references(state, observation_id),
            }
        )
    return sorted(result, key=lambda item: (item.get("evidence_strength") or 0.0, item.get("created_at") or ""), reverse=True)


def _belief(hypothesis: dict[str, Any] | None) -> dict[str, Any]:
    raw = hypothesis.get("belief", {}) if hypothesis else {}
    if not isinstance(raw, dict):
        raw = {}
    probability = _float(raw.get("probability", raw.get("score")))
    previous = _float(raw.get("previous_probability", raw.get("previous_score")))
    delta = probability - previous if probability is not None and previous is not None else None
    return {
        "probability": probability,
        "previous_probability": previous,
        "delta": delta,
        "confidence": raw.get("confidence"),
        "calibrated": raw.get("calibrated") if "calibrated" in raw else None,
        "interpretation": "calibrated statistical probability" if raw.get("calibrated") is True else "heuristic decision-confidence score" if probability is not None else None,
        "history_available": False,
        "history_state": "no_belief_model" if probability is None else "initial_only",
    }


def _hypothesis_quality(hypothesis: dict[str, Any] | None) -> dict[str, Any]:
    if not hypothesis:
        return {"quality": "missing", "missing_fields": ["statement", "mechanism", "scope", "assumptions"], "falsification_criteria": []}
    provenance = hypothesis.get("provenance", {}) if isinstance(hypothesis.get("provenance"), dict) else {}
    missing = [field for field in ("statement", "mechanism", "scope", "assumptions") if not hypothesis.get(field)]
    is_seed = "seed" in str(provenance.get("reason", "")).lower() or "bounded optimisation change" in str(hypothesis.get("title", "")).lower()
    return {
        "quality": "generic_seed" if is_seed else "specific" if not missing else "generic_seed",
        "missing_fields": missing,
        "mechanism": hypothesis.get("mechanism"),
        "scope": hypothesis.get("scope"),
        "assumptions": hypothesis.get("assumptions"),
        "falsification_criteria": hypothesis.get("falsification_criteria", []),
    }


def _active_entities(state: dict[str, Any], campaign: dict[str, Any], campaign_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None, str | None]:
    entities = state.get("entities", {})
    current_trial_id = campaign.get("current_trial_id")
    trial = entities.get("trials", {}).get(current_trial_id)
    purpose = trial.get("purpose", {}) if isinstance(trial, dict) else {}
    if not isinstance(purpose, dict):
        purpose = {}
    question_id = purpose.get("question_id")
    hypothesis_ids = purpose.get("hypothesis_ids", [])
    if not question_id:
        question_id = f"q_{campaign_id}_objective"
    hypothesis_id = hypothesis_ids[0] if hypothesis_ids else f"hyp_{campaign_id}_objective"
    question = entities.get("questions", {}).get(question_id)
    hypothesis = entities.get("hypotheses", {}).get(hypothesis_id)
    if hypothesis is None and entities.get("hypotheses"):
        hypothesis_id, hypothesis = next(iter(entities["hypotheses"].items()))
    if question is None and entities.get("questions"):
        question_id, question = next(iter(entities["questions"].items()))
    return question, hypothesis, question_id, hypothesis_id


def _candidate_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    controller = state.get("controller_state", {})
    rejected = controller.get("last_rejected_candidate", {})
    rejected_id = rejected.get("candidate", {}).get("id") if isinstance(rejected, dict) else None
    result = []
    for candidate_id, candidate in state.get("entities", {}).get("candidate_experiments", {}).items():
        status = candidate.get("status", "proposed")
        reasons: list[str] = []
        if candidate_id == rejected_id:
            status = "rejected"
            reasons = list(rejected.get("reasons", []))
        result.append(
            {
                "id": candidate_id,
                "title": candidate.get("title") or _short_text(candidate.get("purpose", ""), 120),
                "purpose": candidate.get("purpose", ""),
                "status": status,
                "question_id": candidate.get("question_id") or candidate.get("addresses", {}).get("question_id"),
                "hypothesis_ids": candidate.get("hypothesis_ids") or candidate.get("addresses", {}).get("hypothesis_ids", []),
                "rationale": candidate.get("rationale", ""),
                "estimated_gpu_hours": _float(candidate.get("estimated_gpu_hours", candidate.get("cost", {}).get("estimated_gpu_hours"))),
                "estimated_wall_hours": _float(candidate.get("estimated_wall_hours", candidate.get("cost", {}).get("estimated_wall_hours"))),
                "expected_information_gain": _float(candidate.get("decision_value", {}).get("expected_information_gain")),
                "expected_metric_improvement": _float(candidate.get("decision_value", {}).get("expected_metric_improvement")),
                "scientific_value": _float(candidate.get("decision_value", {}).get("scientific_value")),
                "value_per_gpu_hour": _float(candidate.get("score", {}).get("value_per_gpu_hour")),
                "fixed_variables": candidate.get("fixed_variables", {}),
                "configuration_patch": candidate.get("configuration_patch", candidate.get("trial_patch", {})),
                "expected_outcomes": candidate.get("expected_outcomes", []),
                "falsification_criteria": candidate.get("falsification_criteria", []),
                "risks": candidate.get("risks", []),
                "validation_reasons": reasons,
                "created_at": _iso(candidate.get("created_at")),
            }
        )
    return sorted(result, key=lambda item: (item.get("status") != "running", item.get("status") == "rejected", -(item.get("value_per_gpu_hour") or 0.0)))


def _alerts(state: dict[str, Any], current: dict[str, Any], artifact_age_seconds: float | None) -> list[dict[str, Any]]:
    controller = state.get("controller_state", {})
    result: list[dict[str, Any]] = []
    safe_reason = controller.get("safe_stop_reason")
    if safe_reason:
        result.append(
            {
                "id": "controller-safe-stop-metadata",
                "severity": "warning" if current.get("process_running") else "critical",
                "type": "stale_controller_metadata" if current.get("process_running") else "safe_stop",
                "condition": safe_reason,
                "measurements": {"controller_status": controller.get("status"), "process_running": current.get("process_running")},
                "recommended_action": "Continue monitoring; reconcile controller metadata after the active stage boundary." if current.get("process_running") else "Do not launch another trial until the safe-stop reason is resolved.",
                "automatic": not bool(current.get("process_running")),
            }
        )
    if artifact_age_seconds is not None and current.get("process_running") and artifact_age_seconds > 7200:
        result.append(
            {
                "id": "artifact-staleness",
                "severity": "warning",
                "type": "excessive_runtime",
                "condition": "The process is alive but no run artifact has changed recently.",
                "measurements": {"artifact_age_seconds": artifact_age_seconds},
                "recommended_action": "Inspect the runner and checkpoint before considering intervention.",
                "automatic": False,
            }
        )
    return result


def _focused_graph(state: dict[str, Any], hypothesis_id: str | None, evidence: list[dict[str, Any]], candidates: list[dict[str, Any]], level: int, relation_depth: int, filters: dict[str, Any]) -> dict[str, Any]:
    graph = build_reasoning_graph(state, level=max(0, min(5, level)))
    if graph is None:
        return {"nodes": [], "edges": [], "svg": None}
    starts: set[str] = set()
    if hypothesis_id and f"hypotheses:{hypothesis_id}" in graph:
        starts.add(f"hypotheses:{hypothesis_id}")
    for item in evidence:
        node = f"observations:{item['id']}"
        if node in graph:
            starts.add(node)
        for experiment_id in item.get("source_experiments", []):
            node = f"experiments:{experiment_id}"
            if node in graph:
                starts.add(node)
    for candidate in candidates[:1]:
        node = f"candidate_experiments:{candidate['id']}"
        if node in graph:
            starts.add(node)
    visible = set(starts)
    frontier = set(starts)
    for _ in range(max(0, relation_depth)):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        next_frontier -= visible
        visible.update(next_frontier)
        frontier = next_frontier
    entity_filter = filters.get("entity_type")
    if entity_filter:
        visible = {node for node in visible if graph.nodes[node].get("kind") == entity_filter or node.split(":", 1)[0] == entity_filter}
    nodes = [
        {"id": node, **{key: value for key, value in graph.nodes[node].items() if key in {"label", "tooltip", "kind", "status", "color"}}}
        for node in graph
        if node in visible
    ]
    edges = [
        {"source": source, "target": target, "relation": data.get("relation", "")}
        for source, target, _key, data in graph.edges(keys=True, data=True)
        if source in visible and target in visible and (not filters.get("relation_type") or data.get("relation") == filters["relation_type"])
    ]
    focused = graph.subgraph(visible).copy()
    if filters.get("relation_type"):
        focused.remove_edges_from(
            [
                (source, target, key)
                for source, target, key, data in focused.edges(keys=True, data=True)
                if data.get("relation") != filters["relation_type"]
            ]
        )
    svg = _graphviz_svg(focused, level)
    return {"nodes": nodes, "edges": edges, "svg": svg.decode("utf-8") if svg else None}


def build_investigation(
    root: str | Path,
    campaign: str,
    *,
    view: str = "current",
    level: int = 3,
    relation_depth: int = 1,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    campaign_name, config_path = resolve_campaign(root_path, campaign)
    config, config_error = _read_yaml(config_path)
    campaign_config = config.get("campaign", {})
    campaign_id = str(campaign_config.get("id", campaign_name))
    state_path = _resolve(root_path, config.get("scientific_state", {}).get("path") or campaign_config.get("scientific_state_path") or "state/scientific_state.yaml")
    state = load_state(state_path) if state_path and state_path.exists() else {"entities": {}, "relations": [], "controller_state": {}, "project": {}}
    campaign_state_path = _resolve(root_path, config.get("artifacts", {}).get("state_path") or f"artifacts/campaigns/{campaign_id}/campaign_state.json")
    campaign_state, campaign_state_error = _read_json(campaign_state_path)
    active_question, active_hypothesis, question_id, hypothesis_id = _active_entities(state, campaign_state, campaign_id)
    launch = campaign_state.get("active_launch_state", {})
    run_status_path = _resolve(root_path, launch.get("run_status_path"))
    run_status, run_status_error = _read_json(run_status_path)
    stage_state_path = _resolve(root_path, campaign_state.get("stage_state_path"))
    stage_state, stage_state_error = _read_json(stage_state_path)
    run_dir = _resolve(root_path, run_status.get("run_dir") or stage_state.get("run_dir"))
    history_path = _latest_history(run_dir)
    summary_path = _latest_summary(run_dir)
    history = read_history(history_path) if history_path else []
    summary_metrics = read_summary_metrics(summary_path) if summary_path else {}
    primary_metric = str(config.get("objective", {}).get("primary_metric", "macro_f1"))
    metric_display = _metric_descriptor(history, primary_metric)
    series = _metric_series(history, primary_metric, metric_display)
    configured_epochs = None
    stage = campaign_state.get("current_stage")
    if stage:
        stage_config_path = _resolve(root_path, campaign_state.get("trial_configs", {}).get(stage))
        stage_config, _ = _read_yaml(stage_config_path) if stage_config_path else ({}, None)
        configured_epochs = stage_config.get("optimization_config", {}).get("epochs") or stage_config.get("epochs")
    latest_epoch = len(history)
    process_running = _is_running(launch.get("pid"))
    started_seconds = _timestamp_seconds(campaign_state.get("started_at") or launch.get("started_at"))
    elapsed_seconds = max(0.0, _now().timestamp() - started_seconds) if started_seconds is not None else None
    artifact_mtimes = [_file_mtime(path) for path in (history_path, summary_path, run_status_path, stage_state_path)]
    artifact_mtimes = [value for value in artifact_mtimes if value is not None]
    artifact_age = _now().timestamp() - max(artifact_mtimes) if artifact_mtimes else None
    status = "running" if process_running else str(campaign_state.get("status") or state.get("controller_state", {}).get("status") or "unknown")
    if status == "autonomous_safe_stop":
        investigation_status = "stalled" if not process_running else "running"
    elif process_running:
        investigation_status = "running"
    elif campaign_state.get("status") in {"trial_completed", "campaign_completed", "analysis_completed"}:
        investigation_status = "completed"
    else:
        investigation_status = "awaiting results" if stage else "stalled"
    current = {
        "id": run_status.get("experiment_id") or stage,
        "title": str(stage or "No active stage"),
        "status": status,
        "campaign_status": campaign_state.get("status"),
        "trial_id": campaign_state.get("current_trial_id"),
        "stage": stage,
        "purpose": _entity_text(campaign_state.get("purpose", {}) if isinstance(campaign_state.get("purpose"), dict) else active_hypothesis or {}),
        "started_at": _iso(campaign_state.get("started_at") or launch.get("started_at")),
        "elapsed_seconds": elapsed_seconds,
        "estimated_remaining_seconds": (max(0, int(configured_epochs) - latest_epoch) * elapsed_seconds / latest_epoch) if configured_epochs and latest_epoch and elapsed_seconds is not None else None,
        "progress_fraction": min(1.0, latest_epoch / float(configured_epochs)) if configured_epochs else None,
        "current_epoch": latest_epoch or None,
        "total_epochs": configured_epochs,
        "pid": launch.get("pid"),
        "process_running": process_running,
        "run_dir": str(run_dir) if run_dir else None,
        "checkpoint": run_status.get("checkpoint_path") or run_status.get("resume_checkpoint_path"),
        "history_path": str(history_path) if history_path else None,
        "primary_metric": primary_metric,
        "metric_display": metric_display,
        "current_metric": _latest_metric(series),
        "best_metric": _best_metric(series, maximize=bool(config.get("objective", {}).get("maximize", True))),
        "summary_metrics": summary_metrics,
        "metric_series": series[-80:],
        "artifact_updated_at": datetime.fromtimestamp(max(artifact_mtimes), tz=timezone.utc).isoformat(timespec="seconds") if artifact_mtimes else None,
    }
    filters = filters or {}
    all_evidence = _evidence(state, hypothesis_id, filters)
    evidence = all_evidence[-12:] if view == "current" else all_evidence[-50:] if view == "history" else all_evidence
    candidates = _candidate_rows(state)
    beliefs = _belief(active_hypothesis)
    hypothesis_quality = _hypothesis_quality(active_hypothesis)
    unclassified_count = sum(1 for item in all_evidence if item["direction"] == "unclassified")
    classified_count = len(all_evidence) - unclassified_count
    primary_candidate = candidates[0] if candidates else None
    current["purpose"] = (
        _entity_text(campaign_state.get("purpose", {}))
        if isinstance(campaign_state.get("purpose"), dict)
        else ""
    ) or (primary_candidate.get("purpose", "") if primary_candidate else "") or _entity_text(stage_state.get("purpose", {})) or "Purpose not recorded for this stage."
    current["compute"] = {
        "consumed_gpu_hours": None,
        "expected_gpu_hours": primary_candidate.get("estimated_gpu_hours") if primary_candidate else None,
        "remaining_gpu_hours": campaign_config.get("remaining_gpu_hours"),
        "available": bool(primary_candidate and primary_candidate.get("estimated_gpu_hours") is not None) or campaign_config.get("remaining_gpu_hours") is not None,
    }
    current["baseline_comparison"] = {"available": False, "reason": "No directly comparable baseline metric is recorded for the active stage."}
    current["artifact_freshness"] = {"status": "fresh" if artifact_age is not None and artifact_age <= 7200 else "stale" if artifact_age is not None else "unknown", "age_seconds": artifact_age, "updated_at": current.get("artifact_updated_at")}
    raw_predictions = primary_candidate.get("expected_outcomes", []) if primary_candidate else []
    raw_falsification = primary_candidate.get("falsification_criteria", []) if primary_candidate else []
    registration_status = "registered" if raw_predictions and raw_falsification else "partial" if raw_predictions or raw_falsification else "missing"
    expected_outcomes = {
        "registration_status": registration_status,
        "source_candidate_id": primary_candidate.get("id") if primary_candidate else None,
        "predictions": [
            {
                "id": f"prediction-{index}",
                "statement": statement,
                "hypothesis_ids": primary_candidate.get("hypothesis_ids", [hypothesis_id]) if primary_candidate else [hypothesis_id],
                "observed_status": "not_yet_observed",
                "source": "pre_registered",
            }
            for index, statement in enumerate(raw_predictions, start=1)
        ],
        "falsification_criteria": raw_falsification,
        "missing_reason": "This experiment has no registered predictions and cannot currently distinguish competing hypotheses." if registration_status == "missing" else None,
    }
    errors = [error for error in (config_error, campaign_state_error, run_status_error, stage_state_error) if error]
    alerts = _alerts(state, current, artifact_age)
    if registration_status != "registered":
        alerts.append(
            {
                "id": "missing-expected-outcomes",
                "severity": "warning",
                "type": "missing_predictions",
                "condition": expected_outcomes["missing_reason"] or "Expected outcomes are only partially registered.",
                "measurements": {"registration_status": registration_status, "prediction_count": len(raw_predictions), "falsification_count": len(raw_falsification)},
                "recommended_action": "Register predictions and falsification criteria before treating the experiment as discriminative.",
                "automatic": False,
            }
        )
    if metric_display.get("role") != "primary":
        alerts.append(
            {
                "id": "primary-metric-unavailable",
                "severity": "warning",
                "type": "missing_metric",
                "condition": f"Primary metric {primary_metric} is unavailable for this stage; the displayed series is diagnostic.",
                "measurements": {"requested_metric": primary_metric, "display_metric": metric_display.get("display_metric")},
                "recommended_action": "Do not interpret the diagnostic curve as the primary optimization result.",
                "automatic": False,
            }
        )
    if unclassified_count:
        alerts.append(
            {
                "id": "unclassified-evidence",
                "severity": "warning",
                "type": "insufficient_evidence_classification",
                "condition": f"{unclassified_count} observations have no explicit support, contradiction, or inconclusive classification.",
                "measurements": {"unclassified": unclassified_count, "classified": classified_count, "total": len(evidence)},
                "recommended_action": "Add explicit evidence relations or observation direction fields before using these observations to update belief.",
                "automatic": False,
            }
        )
    for observation in all_evidence:
        if observation["type"] in {"non_finite_metric", "unstable_metric", "regression"}:
            alerts.append(
                {
                    "id": observation["id"],
                    "severity": "critical" if observation["type"] == "non_finite_metric" else "warning",
                    "type": observation["type"],
                    "condition": observation["summary"],
                    "measurements": observation.get("measurements", {}),
                    "recommended_action": "Review the source observation and deterministic policy response.",
                    "automatic": True,
                }
            )
    focused_graph = _focused_graph(state, hypothesis_id, evidence, candidates, level, relation_depth, filters or {})
    histories = []
    for audit in state.get("audit_log", []):
        operation = audit.get("operation", {})
        if operation.get("operation") == "controller_update" and any(key in operation.get("value", {}) for key in ("belief", "belief_probability")):
            histories.append({"id": audit.get("id"), "timestamp": audit.get("created_at"), "rationale": operation, "actor": audit.get("actor")})
    if histories:
        beliefs["history_available"] = True
        beliefs["history_state"] = "auditable_updates"
    elif beliefs.get("probability") is not None:
        histories.append(
            {
                "id": "belief-baseline",
                "timestamp": active_hypothesis.get("created_at") if active_hypothesis else None,
                "previous_score": None,
                "new_score": beliefs.get("probability"),
                "direction": "initial",
                "rationale": "Initial belief recorded with the hypothesis; no subsequent belief update is available.",
                "actor": (active_hypothesis.get("provenance", {}) or {}).get("created_by", "unknown") if active_hypothesis else "unknown",
                "provenance": active_hypothesis.get("provenance", {}) if active_hypothesis else {},
            }
        )
    project = state.get("project", {})
    return {
        "schema_version": 1,
        "campaign": {"name": campaign_name, "id": campaign_id, "config_path": str(config_path)},
        "project": {
            "id": project.get("id", campaign_id),
            "name": project.get("name", project.get("id", campaign_id)),
            "objective": project.get("objective", config.get("objective", {})),
            "primary_metric": primary_metric,
            "guardrails": project.get("guardrails", config.get("objective", {}).get("minimums", {})),
            "remaining_gpu_hours": campaign_config.get("remaining_gpu_hours"),
            "trial_budget": campaign_config.get("trial_budget"),
        },
        "investigation": {
            "status": investigation_status,
            "health": "critical" if any(alert["severity"] == "critical" for alert in alerts) else "warning" if alerts else "healthy",
            "last_updated": current.get("artifact_updated_at") or _iso(state.get("controller_state", {}).get("last_poll_at")),
            "view": view,
        },
        "active_question": {"id": question_id, **(active_question or {}), "text": _entity_text(active_question or {})} if active_question or question_id else None,
        "active_hypothesis": {"id": hypothesis_id, **(active_hypothesis or {}), "belief": beliefs, "belief_score": beliefs, "hypothesis_quality": hypothesis_quality} if active_hypothesis or hypothesis_id else None,
        "current_experiment": current,
        "expected_outcomes": expected_outcomes,
        "evidence": {
            "supporting": [item for item in evidence if item["direction"] == "supporting"],
            "contradicting": [item for item in evidence if item["direction"] == "contradicting"],
            "inconclusive": [item for item in evidence if item["direction"] == "inconclusive"],
            "unclassified": [item for item in evidence if item["direction"] == "unclassified"],
            "total": len(all_evidence),
            "counts": {
                "supporting": sum(1 for item in all_evidence if item["direction"] == "supporting"),
                "contradicting": sum(1 for item in all_evidence if item["direction"] == "contradicting"),
                "inconclusive": sum(1 for item in all_evidence if item["direction"] == "inconclusive"),
                "unclassified": unclassified_count,
            },
        },
        "candidates": candidates,
        "belief_history": histories,
        "alerts": alerts,
        "health": {
            "status": "critical" if any(alert["severity"] == "critical" for alert in alerts) else "warning" if alerts else "healthy",
            "process_live": process_running,
            "artifact_freshness": {"status": "fresh" if artifact_age is not None and artifact_age <= 7200 else "stale" if artifact_age is not None else "unknown", "age_seconds": artifact_age, "updated_at": current.get("artifact_updated_at")},
            "controller_metadata": "stale" if state.get("controller_state", {}).get("safe_stop_reason") and process_running else "consistent",
            "intervention_required": any(alert["severity"] == "critical" for alert in alerts),
        },
        "controller": state.get("controller_state", {}),
        "graph": focused_graph,
        "diagnostics": {
            "errors": errors,
            "missing_fields": hypothesis_quality.get("missing_fields", []),
            "reference_warnings": [],
            "data_coverage": {
                "missing_beliefs": beliefs.get("probability") is None,
                "missing_predictions": registration_status == "missing",
                "partial_predictions": registration_status == "partial",
                "unclassified_observations": unclassified_count,
                "classified_observations": classified_count,
                "missing_baseline": True,
                "missing_metric": metric_display.get("role") != "primary",
            },
        },
    }


def find_entity(root: str | Path, campaign: str, entity_id: str, collection: str | None = None) -> dict[str, Any] | None:
    _, config_path = resolve_campaign(Path(root), campaign)
    config, _ = _read_yaml(config_path)
    state_path = _resolve(Path(root), config.get("scientific_state", {}).get("path") or config.get("campaign", {}).get("scientific_state_path") or "state/scientific_state.yaml")
    state = load_state(state_path) if state_path and state_path.exists() else {"entities": {}}
    collections = [collection] if collection else list(state.get("entities", {}))
    for name in collections:
        record = state.get("entities", {}).get(name, {}).get(entity_id)
        if record is not None:
            return {"id": entity_id, "collection": name, "record": record}
    return None


def find_observation(root: str | Path, campaign: str, observation_id: str) -> dict[str, Any] | None:
    return find_entity(root, campaign, observation_id, "observations")
