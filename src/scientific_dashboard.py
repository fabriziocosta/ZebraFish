"""Compact dashboard views for the autonomous scientific experiment loop."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import re
import textwrap
from typing import Any, Iterable

from src.scientific_state import load_state

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - notebook environment normally provides it.
    nx = None

try:
    import pygraphviz  # noqa: F401
    from networkx.drawing.nx_agraph import to_agraph
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional notebook dependency.
    to_agraph = None


ENTITY_LABELS = {
    "trials": "trial",
    "experiments": "stage",
    "observations": "observation",
    "hypotheses": "hypothesis",
    "beliefs": "belief",
    "questions": "question",
    "candidate_experiments": "candidate",
    "components": "component",
    "datasets": "dataset",
}

NODE_COLORS = {
    "trial": "#1f77b4",
    "stage": "#4c78a8",
    "observation": "#f2a541",
    "hypothesis": "#e45756",
    "belief": "#b279a2",
    "question": "#72b7b2",
    "candidate": "#54a24b",
    "component": "#9d9d9d",
    "dataset": "#79706e",
}

RELATION_COLORS = {
    "supports": "#2f8f6b",
    "contradicts": "#c95b5b",
    "tests": "#4f8fe8",
    "produced": "#c88a2b",
    "contains": "#71808a",
    "addresses": "#7b6fb1",
    "motivates": "#b36b9c",
    "reuses_checkpoint": "#3b9caa",
    "alternative_to": "#8b6f47",
    "derived_from": "#6f88a8",
}

_DECIMAL_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LLM_LABEL_CACHE: dict[str, tuple[str, str]] = {}


def _short_id(value: Any, limit: int = 24) -> str:
    text = str(value)
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def _summary_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("statement", "question", "purpose", "description", "title", "type"):
            if value.get(key):
                return _summary_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_summary_text(item) for item in value if _summary_text(item))
    return str(value).strip()


def _format_numbers_in_text(value: Any) -> str:
    """Compact decimal numbers embedded in narrative dashboard text."""

    text = str(value)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            number = float(raw)
        except ValueError:
            return raw
        if not math.isfinite(number):
            return raw
        return f"{number:.3g}"

    return _DECIMAL_NUMBER.sub(replace, text)


def _format_display_value(value: Any) -> Any:
    """Format numeric values for tables without changing persisted state."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return _format_score(value)
    if isinstance(value, dict):
        return {key: _format_display_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_format_display_value(child) for child in value]
    return value


def _truncate(value: Any, limit: int = 46) -> str:
    text = " ".join(_format_numbers_in_text(_summary_text(value)).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _wrapped_lines(value: Any, *, width: int = 28, max_lines: int = 2) -> list[str]:
    text = " ".join(_format_numbers_in_text(_summary_text(value)).split())
    if not text:
        return []
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(1, width - 1)].rstrip() + "…"
    return lines


def _format_score(value: Any) -> str:
    """Format a score with at most three significant digits for node labels."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return _truncate(value, 12)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.3g}"


def _label_entity_payload(collection: str, entity_id: str, entity: dict[str, Any]) -> dict[str, Any]:
    """Select the semantic fields needed to explain a graph node.

    Graph labels are presentation metadata.  Keep the LLM prompt compact and
    avoid sending raw epoch histories, artifact paths, or unrelated state.
    """

    fields_by_collection = {
        "trials": ("purpose", "status", "outcome", "stage_experiment_ids"),
        "experiments": ("stage", "purpose", "status", "final_objective_score"),
        "observations": ("type", "statement", "measurements", "detection"),
        "hypotheses": ("title", "statement", "mechanism", "scope", "status"),
        "beliefs": ("belief", "probability", "statement", "status"),
        "questions": ("question", "statement", "purpose", "status"),
        "candidate_experiments": ("title", "purpose", "expected_outcomes", "status"),
        "components": ("title", "description", "status"),
        "datasets": ("title", "description", "status"),
    }
    selected = {
        key: entity[key]
        for key in fields_by_collection.get(collection, ())
        if entity.get(key) not in (None, "", [], {})
    }
    return {"key": f"{collection}:{entity_id}", "kind": ENTITY_LABELS.get(collection, collection), "fields": selected}


def _label_schema() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "scientific_graph_labels",
            "description": "Concise human-readable labels for scientific reasoning graph nodes.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["labels"],
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["key", "label"],
                            "properties": {
                                "key": {"type": "string"},
                                "label": {"type": "string"},
                            },
                        },
                    }
                },
            },
        }
    }


def _normalise_llm_label(value: Any) -> str | None:
    """Accept a short label and wrap it without dropping its meaning."""

    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > 90:
        return None
    lines = textwrap.wrap(text, width=30, break_long_words=False, break_on_hyphens=False)
    if len(lines) > 3:
        return None
    return "\n".join(lines)


def _llm_graph_labels(records: list[dict[str, Any]], *, client: Any | None = None) -> dict[str, str]:
    """Ask the LLM for concise semantic labels, with a safe deterministic fallback.

    The dashboard remains usable without credentials or during API failures.
    Labels are cached in memory by node content, so refreshing the dashboard
    does not repeatedly spend tokens for unchanged state.
    """

    if not records:
        return {}
    payload_by_key = {record["key"]: record for record in records}
    uncached = []
    for key, record in payload_by_key.items():
        fingerprint = json.dumps(record, sort_keys=True, default=str)
        cached = _LLM_LABEL_CACHE.get(key)
        if cached and cached[0] == fingerprint:
            continue
        uncached.append(record)
    if uncached:
        if client is None:
            enabled = os.environ.get("ZEBRAFISH_DASHBOARD_LLM_LABELS", "off").lower() in {"1", "true", "on", "yes"}
            if not enabled or not os.environ.get("OPENAI_API_KEY"):
                uncached = []
            else:
                try:
                    from openai import OpenAI

                    client = OpenAI(timeout=10.0)
                except Exception:
                    uncached = []
        if client is not None and uncached:
            prompt = "\n\n".join(
                [
                    "Create concise graph labels for a scientific experiment dashboard.",
                    "Each label must tell a human what the item means, not merely repeat its identifier.",
                    "Use 3 to 10 plain-language words, at most 90 characters, and do not use ellipses.",
                    "Include the entity kind when useful (for example, 'Question:' or 'Observation:').",
                    "For observations, name the scientific finding; for questions and hypotheses, name the testable claim; for trials and candidates, name the intervention or purpose.",
                    "Return one label for every supplied key and do not invent facts.",
                    json.dumps(uncached, indent=2, sort_keys=True, default=str),
                ]
            )
            try:
                response = client.responses.create(
                    model=os.environ.get("ZEBRAFISH_DASHBOARD_LABEL_MODEL", "gpt-5.3-codex"),
                    max_output_tokens=max(300, len(uncached) * 40),
                    text=_label_schema(),
                    input=prompt,
                )
                output_text = getattr(response, "output_text", None)
                result = json.loads(output_text) if isinstance(output_text, str) else {}
                for item in result.get("labels", []) if isinstance(result, dict) else []:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("key", ""))
                    label = _normalise_llm_label(item.get("label"))
                    if key in payload_by_key and label:
                        fingerprint = json.dumps(payload_by_key[key], sort_keys=True, default=str)
                        _LLM_LABEL_CACHE[key] = (fingerprint, label)
            except Exception:
                # Presentation labels must never break the read-only dashboard.
                pass
    return {key: cached[1] for key in payload_by_key if (cached := _LLM_LABEL_CACHE.get(key)) and cached[0] == json.dumps(payload_by_key[key], sort_keys=True, default=str)}


def _node_display_label(
    collection: str,
    entity_id: str,
    entity: dict[str, Any],
    *,
    trial_number: int | None = None,
) -> str:
    """Create a compact, human-readable Graphviz node label."""

    kind = ENTITY_LABELS.get(collection, collection).upper()
    if collection == "trials" and trial_number is not None:
        lines = [f"TRIAL {trial_number}"]
    else:
        lines = [kind, _short_id(entity_id)]
    status = entity.get("status")
    if status:
        lines.append(f"status: {status}")
    if collection == "trials":
        outcome = entity.get("outcome", {}) if isinstance(entity.get("outcome"), dict) else {}
        score = outcome.get("score", outcome.get("ranking_score"))
        if score not in (None, ""):
            lines.append(f"score: {_format_score(score)}")
        stages = entity.get("stage_experiment_ids", [])
        if stages:
            lines.append(f"stages: {len(stages)}")
    elif collection == "experiments":
        if entity.get("stage"):
            lines.append(f"stage: {entity['stage']}")
        if entity.get("final_objective_score") not in (None, ""):
            lines.append(f"score: {_format_score(entity['final_objective_score'])}")
    elif collection == "observations":
        if entity.get("type"):
            lines.append(f"kind: {_truncate(entity['type'], 32)}")
        if entity.get("statement"):
            lines.extend(_wrapped_lines(entity["statement"]))
    elif collection in {"hypotheses", "beliefs", "questions"}:
        text = entity.get("statement") or entity.get("question") or entity.get("belief") or entity.get("title")
        if text:
            lines.extend(_wrapped_lines(text))
    elif collection == "candidate_experiments":
        if entity.get("purpose"):
            lines.extend(_wrapped_lines(entity["purpose"]))
        if entity.get("estimated_gpu_hours") not in (None, ""):
            lines.append(f"cost: {entity['estimated_gpu_hours']} GPU h")
    else:
        text = entity.get("title") or entity.get("description")
        if text:
            lines.extend(_wrapped_lines(text))
    return "\n".join(lines)


def _node_tooltip(collection: str, entity_id: str, entity: dict[str, Any]) -> str:
    """Return the untruncated scientific content shown on hover."""

    kind = ENTITY_LABELS.get(collection, collection)
    lines = [f"{kind}: {entity_id}"]
    if entity.get("status"):
        lines.append(f"status: {entity['status']}")
    if collection == "trials":
        purpose = entity.get("purpose")
        if purpose:
            lines.append(f"purpose: {_format_numbers_in_text(_summary_text(purpose))}")
        outcome = entity.get("outcome", {})
        if isinstance(outcome, dict) and outcome.get("score") not in (None, ""):
            lines.append(f"score: {_format_score(outcome['score'])}")
    elif collection == "experiments":
        for key in ("stage", "trial_id", "final_objective_score"):
            if entity.get(key) not in (None, ""):
                value = _format_score(entity[key]) if key == "final_objective_score" else _format_numbers_in_text(entity[key])
                lines.append(f"{key}: {value}")
    elif collection == "observations":
        if entity.get("type"):
            lines.append(f"type: {entity['type']}")
        if entity.get("statement"):
            lines.append(f"statement: {_format_numbers_in_text(entity['statement'])}")
        if entity.get("measurements"):
            lines.append(f"measurements: {_format_numbers_in_text(entity['measurements'])}")
    elif collection in {"hypotheses", "beliefs", "questions"}:
        text = entity.get("statement") or entity.get("question") or entity.get("belief") or entity.get("title")
        if text:
            lines.append(_format_numbers_in_text(_summary_text(text)))
    elif collection == "candidate_experiments":
        for key in ("purpose", "expected_outcomes", "falsification_criteria", "risks", "estimated_gpu_hours"):
            if entity.get(key) not in (None, ""):
                value = _format_score(entity[key]) if key == "estimated_gpu_hours" else _format_numbers_in_text(_summary_text(entity[key]))
                lines.append(f"{key}: {value}")
    else:
        for key in ("title", "description"):
            if entity.get(key):
                lines.append(f"{key}: {_format_numbers_in_text(_summary_text(entity[key]))}")
    return "\n".join(lines)


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or os.getcwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "state" / "scientific_state.yaml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def _resolve(root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    return target if target.is_absolute() else root / target


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.update(_flatten_metrics(child, f"{prefix}.{key}" if prefix else str(key)))
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return flattened
        if math.isfinite(number):
            flattened[prefix] = number
    return flattened


def _entity_rows(state: dict[str, Any], collection: str, level: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id, entity in state.get("entities", {}).get(collection, {}).items():
        row = {
            "id": entity_id,
            "type": ENTITY_LABELS.get(collection, collection),
            "status": entity.get("status", ""),
            "title": _format_numbers_in_text(entity.get("title") or entity.get("statement") or entity.get("question") or entity.get("purpose") or ""),
        }
        if level >= 3:
            row["created_at"] = entity.get("created_at", "")
            row["provenance"] = entity.get("provenance", {}).get("created_by", "") if isinstance(entity.get("provenance"), dict) else ""
        if level >= 4:
            row["details"] = _format_display_value(entity)
        rows.append(row)
    return rows


def _trial_rows(state: dict[str, Any], level: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_id, trial in state.get("entities", {}).get("trials", {}).items():
        outcome = trial.get("outcome", {}) if isinstance(trial.get("outcome"), dict) else {}
        row = {
            "trial_id": trial_id,
            "status": trial.get("status", ""),
            "score": _format_score(outcome.get("score", outcome.get("ranking_score", ""))),
            "selected_metric": outcome.get("selected_metric", ""),
            "guardrail": outcome.get("guardrail_passed", ""),
            "stages": ", ".join(trial.get("stage_experiment_ids", [])),
        }
        if level >= 3:
            row["objective_eligible"] = outcome.get("objective_eligible", "")
            row["metrics"] = _format_display_value(_flatten_metrics(outcome.get("metrics", {})))
        if level >= 5:
            row["outcome"] = outcome
        rows.append(row)
    return rows


def _current_run_snapshot(root: Path, campaign_state: dict[str, Any]) -> dict[str, Any]:
    launch = campaign_state.get("active_launch_state", {})
    run_status = _read_json(_resolve(root, launch.get("run_status_path")))
    stage_state = _read_json(_resolve(root, campaign_state.get("stage_state_path")))
    return {
        "campaign_id": campaign_state.get("campaign_id", ""),
        "status": campaign_state.get("status", "idle"),
        "phase": campaign_state.get("phase", ""),
        "trial_id": campaign_state.get("current_trial_id", ""),
        "stage": campaign_state.get("current_stage", ""),
        "pid": launch.get("pid"),
        "runner": launch.get("runner", ""),
        "started_at": campaign_state.get("started_at", launch.get("started_at", "")),
        "run_dir": run_status.get("run_dir", ""),
        "run_status": run_status.get("status", stage_state.get("status", "")),
        "checkpoint": run_status.get("checkpoint_path") or run_status.get("resume_checkpoint_path", ""),
        "latest_status": run_status or stage_state,
    }


def _observation_linked_experiments(
    state: dict[str, Any],
    observation_ids: set[str] | None = None,
) -> set[str]:
    """Return stage IDs that have explicit deterministic observation evidence."""

    entities = state.get("entities", {})
    experiments = entities.get("experiments", {})
    observations = entities.get("observations", {})
    linked: set[str] = set()
    for observation_id, observation in observations.items():
        if observation_ids is not None and observation_id not in observation_ids:
            continue
        for experiment_id in observation.get("source_experiments", []):
            if experiment_id in experiments:
                linked.add(str(experiment_id))
    observation_ids_all = set(observations)
    experiment_ids = set(experiments)
    for relation in state.get("relations", []):
        if relation.get("type") != "produced":
            continue
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        if observation_ids is not None and target not in observation_ids:
            continue
        if source in experiment_ids and target in observation_ids_all:
            linked.add(source)
        elif target in experiment_ids and source in observation_ids_all:
            linked.add(target)
    return linked


def build_reasoning_graph(state: dict[str, Any], level: int = 3, *, label_client: Any | None = None):
    """Build a NetworkX graph at the requested detail level."""

    if nx is None:
        return None
    level = max(0, min(5, int(level)))
    graph = nx.MultiDiGraph()
    collections_by_level = {
        0: {"trials", "hypotheses", "questions", "candidate_experiments"},
        1: {"trials", "experiments", "hypotheses", "questions", "candidate_experiments"},
        2: {"trials", "experiments", "observations", "hypotheses", "questions", "candidate_experiments"},
        3: {"trials", "experiments", "observations", "hypotheses", "beliefs", "questions", "candidate_experiments"},
        4: set(ENTITY_LABELS),
        5: set(ENTITY_LABELS),
    }
    selected = collections_by_level[level]
    entities = state.get("entities", {})
    observation_records = list(state.get("entities", {}).get("observations", {}).items())
    if level in {2, 3, 4}:
        # Keep the graph readable while ensuring displayed stages still have
        # an observation edge in the displayed evidence window.
        observation_records = observation_records[-{2: 12, 3: 24, 4: 48}[level]:]
        observation_ids = {observation_id for observation_id, _ in observation_records}
    else:
        observation_ids = None
    observed_experiment_ids = _observation_linked_experiments(state, observation_ids)
    observed_trial_ids = {
        str(entities.get("experiments", {}).get(experiment_id, {}).get("trial_id"))
        for experiment_id in observed_experiment_ids
        if entities.get("experiments", {}).get(experiment_id, {}).get("trial_id")
    }
    trial_numbers = {
        trial_id: index
        for index, trial_id in enumerate(sorted(observed_trial_ids), start=1)
    }
    graph_records: list[dict[str, Any]] = []
    records_to_add: list[tuple[str, str, dict[str, Any]]] = []
    for collection in selected:
        records = list(state.get("entities", {}).get(collection, {}).items())
        if collection == "observations":
            records = observation_records
        elif collection == "experiments":
            records = [(entity_id, entity) for entity_id, entity in records if entity_id in observed_experiment_ids]
        elif collection == "trials":
            records = [(entity_id, entity) for entity_id, entity in records if entity_id in observed_trial_ids]
        for entity_id, entity in records:
            node_id = f"{collection}:{entity_id}"
            records_to_add.append((collection, entity_id, entity))
            graph_records.append(_label_entity_payload(collection, entity_id, entity))
    llm_labels = _llm_graph_labels(graph_records, client=label_client)
    for collection, entity_id, entity in records_to_add:
        node_id = f"{collection}:{entity_id}"
        label = ENTITY_LABELS.get(collection, collection)
        semantic_label = entity.get("display_label") or entity.get("label_summary") or llm_labels.get(node_id)
        if collection == "trials" and trial_numbers.get(entity_id) is not None and semantic_label:
            semantic_label = f"TRIAL {trial_numbers[entity_id]}\n{semantic_label}"
        graph.add_node(
            node_id,
            label=semantic_label or _node_display_label(collection, entity_id, entity, trial_number=trial_numbers.get(entity_id)),
            label_source="entity" if entity.get("display_label") or entity.get("label_summary") else "llm" if node_id in llm_labels else "deterministic",
            tooltip=_node_tooltip(collection, entity_id, entity),
            kind=label,
            status=entity.get("status", ""),
            color=NODE_COLORS.get(label, "#999999"),
        )
    for relation in state.get("relations", []):
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        source_node = next((node for node in graph if node.endswith(f":{source}")), None)
        target_node = next((node for node in graph if node.endswith(f":{target}")), None)
        if source_node and target_node:
            graph.add_edge(source_node, target_node, relation=relation.get("type", ""))

    for trial_id, trial in entities.get("trials", {}).items():
        trial_node = f"trials:{trial_id}"
        if trial_node not in graph:
            continue
        purpose = trial.get("purpose", {})
        for question_id in [purpose.get("question_id")]:
            if question_id and f"questions:{question_id}" in graph:
                graph.add_edge(trial_node, f"questions:{question_id}", relation="addresses")
        for hypothesis_id in purpose.get("hypothesis_ids", []):
            if f"hypotheses:{hypothesis_id}" in graph:
                graph.add_edge(trial_node, f"hypotheses:{hypothesis_id}", relation="tests")
        for experiment_id in trial.get("stage_experiment_ids", []):
            if f"experiments:{experiment_id}" in graph:
                graph.add_edge(trial_node, f"experiments:{experiment_id}", relation="contains")
    for experiment_id, experiment in entities.get("experiments", {}).items():
        experiment_node = f"experiments:{experiment_id}"
        trial_id = experiment.get("trial_id")
        if experiment_node in graph and f"trials:{trial_id}" in graph:
            graph.add_edge(f"trials:{trial_id}", experiment_node, relation="contains")
    for observation_id, observation in entities.get("observations", {}).items():
        observation_node = f"observations:{observation_id}"
        for experiment_id in observation.get("source_experiments", []):
            if observation_node in graph and f"experiments:{experiment_id}" in graph:
                graph.add_edge(f"experiments:{experiment_id}", observation_node, relation="produced")
    for candidate_id, candidate in entities.get("candidate_experiments", {}).items():
        candidate_node = f"candidate_experiments:{candidate_id}"
        if candidate_node not in graph:
            continue
        question_id = candidate.get("question_id") or candidate.get("addresses", {}).get("question_id")
        if question_id and f"questions:{question_id}" in graph:
            graph.add_edge(candidate_node, f"questions:{question_id}", relation="addresses")
        for hypothesis_id in candidate.get("hypothesis_ids", candidate.get("addresses", {}).get("hypothesis_ids", [])):
            if f"hypotheses:{hypothesis_id}" in graph:
                graph.add_edge(candidate_node, f"hypotheses:{hypothesis_id}", relation="tests")
    return graph


def build_dashboard_snapshot(
    *,
    state_path: str | Path = "state/scientific_state.yaml",
    campaign_state_path: str | Path | None = None,
    level: int = 3,
) -> dict[str, Any]:
    level = max(0, min(5, int(level)))
    root = _repo_root(Path(state_path).parent)
    state_target = _resolve(root, state_path) or Path(state_path)
    state = load_state(state_target)
    if campaign_state_path is None:
        campaign_ids = [
            path
            for path in (root / "artifacts" / "campaigns").glob("*/campaign_state.json")
            if path.exists()
        ]
        campaign_state_path = max(campaign_ids, key=lambda path: path.stat().st_mtime) if campaign_ids else None
    campaign_state = _read_json(_resolve(root, campaign_state_path))
    current = _current_run_snapshot(root, campaign_state)
    entities = state.get("entities", {})
    observations = _entity_rows(state, "observations", level)
    hypotheses = _entity_rows(state, "hypotheses", level)
    questions = _entity_rows(state, "questions", level)
    candidates = _entity_rows(state, "candidate_experiments", level)
    if level <= 1:
        observations = observations[-3:]
    elif level <= 3:
        observations = observations[-12:]
    return {
        "level": level,
        "state_path": str(state_target),
        "controller": state.get("controller_state", {}),
        "current_run": current,
        "trial_rows": _trial_rows(state, level)[-20:],
        "hypothesis_rows": hypotheses[-20:],
        "question_rows": questions[-20:],
        "observation_rows": observations,
        "candidate_rows": candidates[-20:],
        "relations": state.get("relations", []) if level >= 3 else state.get("relations", [])[-10:],
        "graph": build_reasoning_graph(state, level),
        "counts": {collection: len(entities.get(collection, {})) for collection in ENTITY_LABELS},
    }


def _as_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _graphviz_svg(graph: Any, level: int) -> bytes | None:
    """Render a reasoning graph with Graphviz's hierarchical DOT layout."""

    if to_agraph is None or graph is None:
        return None
    dot = to_agraph(graph)
    dot.graph_attr.update(
        rankdir="LR",
        ranksep="0.8",
        nodesep="0.35",
        splines="polyline",
        overlap="false",
        bgcolor="transparent",
        pad="0.2",
    )
    for node_id, data in graph.nodes(data=True):
        node = dot.get_node(node_id)
        node.attr.update(
            label=data.get("label", _short_id(node_id)),
            tooltip=data.get("tooltip", data.get("label", _short_id(node_id))),
            shape="box",
            style="rounded,filled",
            fillcolor=data.get("color", "#999999"),
            color="#4a4a4a",
            fontname="Helvetica",
            fontsize="10",
            margin="0.12,0.08",
        )
    for source, target, key, data in graph.edges(keys=True, data=True):
        edge = dot.get_edge(source, target, key)
        relation = str(data.get("relation", ""))
        relation_color = RELATION_COLORS.get(relation, "#777777")
        edge.attr.update(
            label=relation,
            color=relation_color,
            fontcolor=relation_color,
            fontname="Helvetica",
            fontsize="7",
            arrowsize="0.7",
        )
    dot.layout(prog="dot")
    return dot.draw(format="svg", prog="dot")


def render_dashboard(
    *,
    state_path: str | Path = "state/scientific_state.yaml",
    campaign_state_path: str | Path | None = None,
    level: int = 3,
    figsize: tuple[float, float] = (15, 9),
) -> dict[str, Any]:
    """Render a notebook-friendly dashboard and return the snapshot."""

    snapshot = build_dashboard_snapshot(
        state_path=state_path,
        campaign_state_path=campaign_state_path,
        level=level,
    )
    from IPython.display import Markdown, SVG, display

    current = snapshot["current_run"]
    controller = snapshot["controller"]
    counts = snapshot["counts"]
    display(Markdown(
        f"## ZebraFish autonomous dashboard — detail level {snapshot['level']}/5\n\n"
        f"**Campaign:** `{current.get('campaign_id') or 'not detected'}`  "
        f"**Status:** `{current.get('status')}`  **Stage:** `{current.get('stage') or '—'}`  "
        f"**Trial:** `{current.get('trial_id') or '—'}`  **PID:** `{current.get('pid') or '—'}`\n\n"
        f"**Scientific state:** {counts['trials']} trials · {counts['experiments']} stages · "
        f"{counts['observations']} observations · {counts['hypotheses']} hypotheses · "
        f"{counts['questions']} questions · {counts['candidate_experiments']} candidates"
    ))
    if controller.get("safe_stop_reason") and controller.get("status") == "autonomous_safe_stop":
        display(Markdown(f"> **Autonomous warning:** {controller['safe_stop_reason']}"))

    display(Markdown("### Current run"))
    display(_as_dataframe([{
        "status": current.get("status"),
        "run_status": current.get("run_status"),
        "stage": current.get("stage"),
        "started_at": current.get("started_at"),
        "run_dir": current.get("run_dir"),
        "checkpoint": current.get("checkpoint"),
    }]))
    display(Markdown("### Recent trials"))
    display(_as_dataframe(snapshot["trial_rows"]))

    if snapshot["hypothesis_rows"] or snapshot["question_rows"]:
        display(Markdown("### Hypotheses and open questions"))
        display(_as_dataframe(snapshot["hypothesis_rows"] + snapshot["question_rows"]))
    if snapshot["candidate_rows"] and level >= 2:
        display(Markdown("### Candidate experiments"))
        display(_as_dataframe(snapshot["candidate_rows"]))
    if snapshot["observation_rows"] and level >= 2:
        display(Markdown("### Recent deterministic observations"))
        display(_as_dataframe(snapshot["observation_rows"]))

    graph = snapshot["graph"]
    display(Markdown("### Reasoning graph"))
    if graph is None:
        display(_as_dataframe(snapshot["relations"]))
    elif graph.number_of_nodes() == 0:
        display(Markdown("_No graph entities are available at this level._"))
    else:
        svg = _graphviz_svg(graph, level)
        if svg is not None:
            display(SVG(data=svg))
        else:
            # Keep the dashboard usable in minimal environments without
            # Graphviz; normal installations use the DOT renderer above.
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            positions = nx.spring_layout(graph, seed=7, k=1.3 / max(1, math.sqrt(graph.number_of_nodes())))
            node_colors = [graph.nodes[node].get("color", "#999999") for node in graph.nodes]
            labels = {node: graph.nodes[node].get("label", node) for node in graph.nodes}
            nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=950, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(graph, positions, arrows=True, alpha=0.35, arrowsize=14, ax=ax)
            if level >= 2:
                nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=ax)
            if level >= 4:
                edge_labels = {(u, v, key): data.get("relation", "") for u, v, key, data in graph.edges(keys=True, data=True)}
                nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=6, ax=ax)
            ax.set_axis_off()
            display(fig)
            plt.close(fig)
    return snapshot


def dashboard_widget(
    *,
    state_path: str | Path = "state/scientific_state.yaml",
    campaign_state_path: str | Path | None = None,
):
    """Return an interactive level slider for Jupyter environments."""

    import ipywidgets as widgets
    from IPython.display import display

    control = widgets.IntSlider(value=3, min=0, max=5, step=1, description="detail", continuous_update=False)
    output = widgets.interactive_output(
        lambda level: render_dashboard(state_path=state_path, campaign_state_path=campaign_state_path, level=level),
        {"level": control},
    )
    display(widgets.VBox([control, output]))
    return control, output
