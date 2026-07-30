"""Autonomous scientific campaign controller.

This module layers the scientific state and deterministic candidate policy on
top of the existing stage runners.  It deliberately reuses the runner and
artifact lifecycle from :mod:`agent_campaign_loop` so existing experiments do
not need to be rewritten to participate.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, TextIO
import uuid

from src import agent_campaign_loop as legacy
from src import agent_experiment_loop as experiment_loop
from src.campaign_watchdog import inspect_campaign
from src.autonomous_policy import CandidateValidation, validate_candidate
from src.experiment_protocol import (
    aggregate_guardrail,
    aggregate_replicates,
    actual_resource_usage,
    checkpoint_manifest,
    config_hash,
    current_code_revision,
    environment_metadata,
    file_hash,
    lockbox_confirmation,
    stable_hash,
)
from src.observation_engine import DetectorConfig, generate_observations, read_summary_metrics
from src.experiment_runner_shared import read_yaml_mapping
from src.scientific_state import (
    apply_operations,
    append_belief_update,
    append_lifecycle_event,
    compact_context,
    ENTITY_COLLECTIONS,
    empty_state,
    load_state,
    merge_nonconflicting_states,
    record_entity,
    reserve_launch,
    update_launch_reservation,
    transactional_update,
    save_state,
    update_controller_state,
)


DECISIONS = {"continue", "propose_trial", "terminate_trial", "stop_campaign", "no_action"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def record_lockbox_confirmation(
    scientific_state: dict[str, Any],
    *,
    candidate_id: str,
    baseline_values: list[float],
    candidate_values: list[float],
    guardrail_values: list[float] | None = None,
    guardrail_minimum: float = 0.30,
    minimum_effect: float = 0.02,
    minimum_replicates: int = 3,
) -> dict[str, Any]:
    """Append one immutable protected-split confirmation record."""

    if candidate_id not in scientific_state.get("entities", {}).get("candidate_experiments", {}):
        raise ValueError(f"unknown candidate for lockbox confirmation: {candidate_id}")
    confirmation = lockbox_confirmation(
        candidate_id=candidate_id,
        baseline_values=baseline_values,
        candidate_values=candidate_values,
        minimum_effect=minimum_effect,
        guardrail_values=guardrail_values,
        guardrail_minimum=guardrail_minimum if guardrail_values is not None else None,
        minimum_replicates=minimum_replicates,
    )
    if confirmation["id"] in scientific_state.get("entities", {}).get("lockbox_confirmations", {}):
        return scientific_state
    updated = record_entity(
        scientific_state,
        "lockbox_confirmations",
        confirmation["id"],
        {**confirmation, "created_at": _now(), "provenance": {"created_by": "lockbox_evaluator", "candidate_id": candidate_id}},
        actor="lockbox_evaluator",
    )
    return updated


def _artifact_manifest(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    paths: set[str] = set()
    for value in artifacts.values():
        if isinstance(value, str):
            paths.add(value)
        elif isinstance(value, list):
            paths.update(str(item) for item in value if isinstance(item, str))
    manifest: list[dict[str, Any]] = []
    for value in sorted(paths):
        path = Path(value)
        if path.exists() and path.is_file():
            try:
                manifest.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_hash(path)})
            except OSError:
                manifest.append({"path": str(path), "status": "unavailable"})
    return manifest


def _state_path(config: dict[str, Any]) -> Path:
    configured = config.get("scientific_state", {}).get("path")
    if configured:
        return Path(str(configured))
    return Path(config.get("campaign", {}).get("scientific_state_path", "state/scientific_state.yaml"))


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("id") or candidate.get("candidate_id") or f"candidate_{uuid.uuid4().hex[:10]}")


def _ensure_seed_knowledge(state: dict[str, Any], campaign_config: dict[str, Any]) -> dict[str, Any]:
    """Create minimal objective-linked knowledge so the first trial is valid."""

    campaign_id = str(campaign_config["campaign"]["id"])
    question_id = f"q_{campaign_id}_objective"
    hypothesis_id = f"hyp_{campaign_id}_objective"
    if question_id not in state["entities"]["questions"]:
        state = record_entity(
            state,
            "questions",
            question_id,
            {
                "question": "Which permitted training change most improves downstream compound discrimination while preserving action performance?",
                "status": "open",
                "importance": 1.0,
                "decision_relevance": 1.0,
                "provenance": {"created_by": "autonomous_controller", "reason": "campaign objective seed"},
            },
            actor="autonomous_controller",
        )
    if hypothesis_id not in state["entities"]["hypotheses"]:
        state = record_entity(
            state,
            "hypotheses",
            hypothesis_id,
            {
                "title": "A bounded optimisation change can improve the campaign objective.",
                "statement": "At least one allowlisted optimisation or loss-weight change will improve compound macro-F1 without violating the action guardrail.",
                "mechanism": "A bounded training intervention changes representation learning enough to improve compound discrimination while leaving the action decision boundary within the registered guardrail.",
                "scope": {"campaign": campaign_id, "stages": list(campaign_config["campaign"].get("stages", []))},
                "assumptions": ["the fixed split is representative", "the registered primary metric is available at the evaluation stage", "the baseline comparison is paired"],
                "intervention": "one allowlisted configuration leaf or one explicitly paired loss-weight change",
                "expected_primary_metric": {"metric": "compound.macro_f1", "direction": "increase", "minimum_effect": float(campaign_config.get("campaign", {}).get("minimum_effect", 0.02))},
                "expected_guardrail": {"metric": "action.accuracy", "direction": "no_decrease", "minimum": 0.30},
                "falsification_criterion": {"metric": "compound.macro_f1", "direction": "no_improvement", "minimum_effect": float(campaign_config.get("campaign", {}).get("minimum_effect", 0.02))},
                "baseline": "registered paired baseline required before confirmatory interpretation",
                "status": "active",
                "belief": {"score": 0.5, "interpretation": "heuristic_decision_score", "confidence": "initial"},
                "question_id": question_id,
                "provenance": {"created_by": "autonomous_controller", "reason": "campaign objective seed"},
            },
            actor="autonomous_controller",
        )
    return state


def _decision_schema() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "autonomous_scientific_decision",
            "description": "A validated autonomous scientific state update and campaign action.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["decision", "reason", "operations", "candidate", "evidence_references"],
                "properties": {
                    "decision": {"type": "string", "enum": sorted(DECISIONS)},
                    "reason": {"type": "string"},
                    "evidence_references": {"type": "array", "items": {"type": "string"}},
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["operation", "path", "value", "expected_old"],
                            "properties": {
                                "operation": {"type": "string", "enum": ["create", "update", "append", "relation", "transition"]},
                                "path": {"type": "string"},
                                "value": {"type": "string", "description": "JSON-encoded operation value."},
                                "expected_old": {"type": "string", "description": "JSON-encoded expected old value, or null."},
                            },
                        },
                    },
                    "candidate": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": [
                            "id",
                            "title",
                            "purpose",
                            "question_id",
                            "hypothesis_ids",
                            "base_experiment",
                            "base_stage",
                            "configuration_patch",
                            "fixed_variables",
                            "varied_variables",
                            "expected_outcomes",
                            "falsification_criteria",
                            "resolved_base_configuration_hash",
                            "source_checkpoint",
                            "replicate_seeds",
                            "estimated_gpu_hours",
                            "estimated_wall_hours",
                            "risks",
                            "allowed_stages",
                            "baseline",
                        ],
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "purpose": {"type": "string"},
                            "question_id": {"type": "string"},
                            "hypothesis_ids": {"type": "array", "items": {"type": "string"}},
                            "base_experiment": {"type": "string"},
                            "base_stage": {"type": "string"},
                            "configuration_patch": {"type": "string", "description": "JSON-encoded stage-keyed YAML patch."},
                            "fixed_variables": {"type": "string", "description": "JSON-encoded mapping."},
                            "varied_variables": {"type": "string", "description": "JSON-encoded mapping of changed leaves to values."},
                            "expected_outcomes": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["metric", "comparison", "direction", "minimum_effect"], "properties": {"metric": {"type": "string"}, "comparison": {"type": "string"}, "direction": {"type": "string", "enum": ["increase", "decrease", "no_change"]}, "minimum_effect": {"type": "number"}}}},
                            "falsification_criteria": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["metric", "comparison", "direction", "minimum_effect"], "properties": {"metric": {"type": "string"}, "comparison": {"type": "string"}, "direction": {"type": "string", "enum": ["increase", "decrease", "no_change"]}, "minimum_effect": {"type": "number"}}}},
                            "resolved_base_configuration_hash": {"type": "string"},
                            "source_checkpoint": {"type": "string"},
                            "replicate_seeds": {"type": "array", "items": {"type": "integer"}},
                            "estimated_gpu_hours": {"type": "number"},
                            "estimated_wall_hours": {"type": "number"},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "allowed_stages": {"type": "array", "items": {"type": "string"}},
                            "baseline": {"type": "string"},
                        },
                    },
                },
            },
        }
    }


def parse_decision(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("autonomous decision must be a JSON object")
    decision = str(payload.get("decision", ""))
    if decision not in DECISIONS:
        raise ValueError(f"unsupported autonomous decision: {decision!r}")
    if not isinstance(payload.get("evidence_references", []), list) or any(
        not isinstance(reference, str) for reference in payload.get("evidence_references", [])
    ):
        raise ValueError("evidence_references must be a list of strings")
    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    normalized_operations: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("each operation must be an object")
        normalized = dict(operation)
        if normalized.get("operation") in {"update", "transition"} and "expected_old" not in normalized:
            raise ValueError("updates and transitions require expected_old for optimistic concurrency")
        for field in ("value", "expected_old"):
            raw = normalized.get(field, "null")
            if not isinstance(raw, str):
                raise ValueError(f"operation {field} must be a JSON-encoded string")
            if field == "expected_old" and not raw.strip():
                raw = "null"
            try:
                normalized[field] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"operation {field} is not valid JSON: {exc}") from exc
        original_path = normalized["path"]
        normalized["path"] = _normalize_operation_path(original_path, normalized.get("value"))
        if normalized.get("operation") == "append" and original_path.startswith("/"):
            if normalized["path"].startswith("entities.") and normalized["path"].count(".") >= 2:
                normalized["operation"] = "create"
            elif original_path.startswith("/relations/"):
                normalized["operation"] = "relation"
        if normalized.get("operation") == "append":
            collection_path = normalized.get("path", "")
            if collection_path == "relations" or collection_path == "entities.relations":
                normalized["operation"] = "relation"
                normalized["path"] = "relations"
            else:
                for collection in ENTITY_COLLECTIONS:
                    if collection_path in {collection, f"entities.{collection}"} and isinstance(normalized.get("value"), dict):
                        entity_id = normalized["value"].get("id") or f"entity_{uuid.uuid4().hex[:10]}"
                        normalized["operation"] = "create"
                        normalized["path"] = f"entities.{collection}.{entity_id}"
                        break
        normalized_operations.append(normalized)
    operations = normalized_operations
    candidate = payload.get("candidate")
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError("candidate must be an object or null")
    if candidate is not None:
        for field in ("configuration_patch", "fixed_variables", "varied_variables"):
            raw = candidate.get(field)
            if not isinstance(raw, str):
                raise ValueError(f"candidate {field} must be a JSON-encoded string")
            try:
                candidate[field] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"candidate {field} is not valid JSON: {exc}") from exc
    return {
        "decision": decision,
        "reason": str(payload.get("reason", "")).strip(),
        "evidence_references": list(payload.get("evidence_references", [])),
        "operations": operations,
        "candidate": candidate,
    }


def _validate_evidence_references(state: dict[str, Any], references: list[str]) -> None:
    known = {
        str(entity_id)
        for collection in state.get("entities", {}).values()
        if isinstance(collection, dict)
        for entity_id in collection
    }
    for reference in references:
        if reference in {"controller_state", "project"}:
            continue
        normalized = reference.split(":", 1)[-1] if ":" in reference else reference
        if normalized not in known:
            raise ValueError(f"decision references unknown evidence: {reference}")


def _normalize_operation_path(path: str, value: Any) -> str:
    """Accept JSON-Pointer entity paths emitted by the LLM."""

    if not isinstance(path, str) or not path.startswith("/"):
        return path
    parts = [part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:]]
    if not parts:
        return path
    if parts[0] == "entities":
        parts = parts[1:]
    if parts[0] in ENTITY_COLLECTIONS:
        collection = parts[0]
        if len(parts) >= 2 and parts[1] == "-":
            entity_id = value.get("id") if isinstance(value, dict) else None
            if not entity_id:
                entity_id = f"entity_{uuid.uuid4().hex[:10]}"
            return f"entities.{collection}.{entity_id}"
        return "entities." + ".".join(parts)
    if parts[0] == "relations" and parts[-1:] == ["-"]:
        return "relations"
    return ".".join(parts)


def _build_prompt(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    summary: dict[str, Any],
    *,
    validation_error: str | None = None,
) -> str:
    policy = campaign_config.get("prompts", {})
    sections = [
        "You are the autonomous scientific controller for ZebraFish.",
        "No human will approve or repair your proposal. Return one bounded, evidence-based decision.",
        "Raw measurements are calculated by deterministic software. Use observations rather than inventing numeric findings.",
        "A candidate must be one controlled trial, address an existing question and hypothesis, use only allowlisted parameters, and include expected outcomes and falsification criteria.",
        "Use exact hypothesis and question IDs from the scientific context, never their titles or statements.",
        "Encode operation value and expected_old as JSON strings. Encode candidate configuration_patch and fixed_variables as JSON strings.",
        "Every created hypothesis, belief, question, or candidate must include created_at and provenance; every update must use expected_old when changing existing state.",
        "If no safe evidence-based trial is available, return stop_campaign or no_action.",
        f"Campaign objective:\n{json.dumps(campaign_config.get('objective', {}), indent=2, sort_keys=True)}",
        f"Campaign policy:\n{json.dumps(policy, indent=2, sort_keys=True)}",
        f"Stage configuration allowlists:\n{json.dumps({stage: loop_config['experiments'][stage].get('allowed_patch_paths', []) for stage in campaign_config['campaign']['stages']}, indent=2, sort_keys=True)}",
        f"Scientific context:\n{json.dumps(compact_context(state), indent=2, sort_keys=True)}",
        f"Current trial summary:\n{json.dumps(summary, indent=2, sort_keys=True)}",
        "Return strict JSON matching the supplied response schema.",
    ]
    if validation_error:
        sections.append(f"The previous proposal was rejected by deterministic policy. Correct these errors:\n{validation_error}")
    return "\n\n".join(sections)


def request_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    state: dict[str, Any],
    summary: dict[str, Any],
    *,
    client: Any | None = None,
    validation_error: str | None = None,
) -> dict[str, Any]:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    agent = campaign_config.get("agent", {})
    attempts = max(1, min(3, int(campaign_config.get("campaign", {}).get("max_decision_retries", 3))))
    last_error: Exception | None = None
    current_validation_error = validation_error
    for attempt in range(attempts):
        response = client.responses.create(
            model=agent.get("model", "gpt-5.3-codex"),
            reasoning={"effort": agent.get("reasoning_effort", "medium")},
            max_output_tokens=int(agent.get("max_output_tokens", 6000)),
            text=_decision_schema(),
            input=_build_prompt(campaign_config, loop_config, state, summary, validation_error=current_validation_error),
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ValueError("OpenAI response did not contain output_text")
        try:
            return parse_decision(output_text)
        except ValueError as exc:
            last_error = exc
            current_validation_error = f"Malformed decision on attempt {attempt + 1}: {exc}. Return valid JSON strings for value and expected_old; use the literal JSON string 'null' when no old value exists."
    raise ValueError(f"autonomous decision remained invalid after {attempts} attempts: {last_error}")


def _summary_for_state(campaign_config: dict[str, Any], legacy_state: dict[str, Any]) -> dict[str, Any]:
    return legacy.summarize_trial(campaign_config, legacy_state)


def _record_stage(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    legacy_state: dict[str, Any],
    stage_status: dict[str, Any],
    controller_status: str,
) -> dict[str, Any]:
    trial_id = str(legacy_state.get("current_trial_id"))
    stage = str(legacy_state.get("current_stage"))
    experiment_id = f"{trial_id}:{stage}"
    if experiment_id in scientific_state["entities"]["experiments"]:
        return scientific_state
    artifacts = stage_status.get("artifacts", {})
    run_dir = artifacts.get("run_dir")
    checkpoint_reuse = legacy_state.get("checkpoint_reuse")
    checkpoint_reuse = checkpoint_reuse if isinstance(checkpoint_reuse, dict) else {}
    resolved_path = legacy_state.get("trial_configs", {}).get(stage)
    resolved_configuration = {}
    if resolved_path and Path(str(resolved_path)).exists():
        try:
            resolved_configuration = read_yaml_mapping(Path(str(resolved_path)))
        except (OSError, ValueError):
            resolved_configuration = {}
    split_manifest = {}
    if run_dir:
        split_manifest_path = Path(str(run_dir)) / "split_manifest.yaml"
        if split_manifest_path.exists():
            try:
                split_manifest = read_yaml_mapping(split_manifest_path)
            except (OSError, ValueError):
                split_manifest = {}
    summary_metrics_path = artifacts.get("summary_metrics_path") if isinstance(artifacts, dict) else None
    if not summary_metrics_path and run_dir:
        summary_candidates = sorted(Path(str(run_dir)).rglob("*summary_metrics*.csv"))
        summary_metrics_path = str(summary_candidates[-1]) if summary_candidates else None
    summary_metrics = read_summary_metrics(summary_metrics_path) if summary_metrics_path else {}
    checkpoint_manifest_value = legacy_state.get("checkpoint_manifest") or checkpoint_reuse.get("manifest")
    if not isinstance(checkpoint_manifest_value, dict):
        checkpoint_candidates = []
        if isinstance(stage_status.get("run_status"), dict) and stage_status["run_status"].get("checkpoint_path"):
            checkpoint_candidates.append(stage_status["run_status"]["checkpoint_path"])
        if isinstance(artifacts, dict) and isinstance(artifacts.get("checkpoints"), list):
            checkpoint_candidates.extend(artifacts["checkpoints"])
        checkpoint_path = next((Path(str(path)) for path in checkpoint_candidates if Path(str(path)).exists()), None)
        if checkpoint_path is not None:
            checkpoint_manifest_value = checkpoint_manifest(
                checkpoint_path,
                source_stage=stage,
                source_experiment=experiment_id,
                model_family="transformer" if "transformer" in str(campaign_config.get("campaign", {}).get("id", "")) else "cnn",
                dataset_hash=str(split_manifest.get("dataset_hash", "unknown")),
                resolved_config=resolved_configuration,
                code_revision=current_code_revision(),
                format_version="torch_state_dict_v1",
                compatibility_status="verified",
            )
    record = {
        "id": experiment_id,
        "trial_id": trial_id,
        "stage": stage,
        "status": "completed" if controller_status == "completed" else controller_status,
        "configuration": {
            "resolved_config_paths": artifacts.get("latest_config_yamls", []),
            "trial_config": resolved_path,
            "resolved_configuration": resolved_configuration,
            "resolved_configuration_hash": config_hash(resolved_configuration),
            "dataset_split_hash": split_manifest.get("hash") or config_hash({"dataset_artifacts": artifacts.get("dataset_artifacts", []), "stage": stage}),
            "split_manifest": split_manifest or None,
        },
        "execution": {
            "run_dir": run_dir,
            "started_at": legacy_state.get("started_at"),
            "completed_at": _now(),
            "artifacts": artifacts,
            "artifact_manifest": _artifact_manifest(artifacts),
            "code_revision": current_code_revision(),
            "random_seed": legacy_state.get("replicate_seed") or resolved_configuration.get("optimization_config", {}).get("random_state"),
            "detector_version": campaign_config.get("observation", {}).get("detector_version", "observation-v2"),
            "controller_version": "autonomous-controller-v2",
            "environment": environment_metadata(),
            "hardware": {"gpu_count": float(stage_status.get("gpu_count", 1.0) or 1.0), "gpu_name": stage_status.get("gpu_name")},
            "resources": actual_resource_usage(
                started_at=stage_status.get("started_at") or legacy_state.get("started_at"),
                completed_at=stage_status.get("completed_at") or _now(),
                gpu_count=float(stage_status.get("gpu_count", 1.0) or 1.0),
                failure_reason=stage_status.get("controller_reason") if controller_status not in {"completed", "running"} else None,
            ),
            "checkpoint_manifest": checkpoint_manifest_value,
            "provenance": {"created_by": "autonomous_controller", "legacy_state": legacy_state.get("stage_state_path")},
        },
        "outcome": {
            "controller_status": controller_status,
            "run_status": stage_status.get("run_status", {}),
            "summary_metrics": summary_metrics,
        },
        "provenance": {"created_by": "autonomous_controller", "source": "campaign_stage_status"},
    }
    if split_manifest:
        split_id = str(split_manifest.get("id") or f"split_{config_hash(split_manifest)[:16]}")
        if split_id not in scientific_state.get("entities", {}).get("split_manifests", {}):
            scientific_state = record_entity(
                scientific_state,
                "split_manifests",
                split_id,
                {**split_manifest, "id": split_id, "status": "protected", "provenance": {"created_by": "deterministic_split_engine", "source": str(Path(str(run_dir)) / "split_manifest.yaml")}},
                actor="deterministic_split_engine",
            )
    scientific_state = record_entity(scientific_state, "experiments", experiment_id, record, actor="autonomous_controller")
    if split_manifest:
        split_id = str(split_manifest.get("id") or f"split_{config_hash(split_manifest)[:16]}")
        scientific_state = apply_operations(
            scientific_state,
            [{"operation": "relation", "value": {"type": "evaluated_on", "source": experiment_id, "target": split_id}}],
            actor="deterministic_split_engine",
        )
    stages = list(campaign_config["campaign"].get("stages", []))
    if stage in stages and stages.index(stage) > 0:
        checkpoint_reuse = legacy_state.get("checkpoint_reuse", {})
        previous_id = str(checkpoint_reuse.get("source_experiment") or f"{trial_id}:{stages[stages.index(stage) - 1]}")
        relation = {"type": "reuses_checkpoint", "source": experiment_id, "target": previous_id}
        if not any(
            item.get("type") == relation["type"]
            and item.get("source") == relation["source"]
            and item.get("target") == relation["target"]
            for item in scientific_state.get("relations", [])
        ):
            scientific_state = apply_operations(
                scientific_state,
                [{"operation": "relation", "value": relation}],
                actor="autonomous_controller",
            )
    outcome_summary = stage_status.get("summary", {}) if isinstance(stage_status.get("summary"), dict) else {}
    summary_score = outcome_summary.get("score")
    comparable_scores: list[float] = []
    comparable_runtimes: list[float] = []
    replicate_scores: list[float] = []
    for prior_id, prior in scientific_state.get("entities", {}).get("experiments", {}).items():
        if not isinstance(prior, dict) or prior_id == experiment_id or prior.get("stage") != stage:
            continue
        prior_outcome = prior.get("outcome", {}) if isinstance(prior.get("outcome"), dict) else {}
        value = prior_outcome.get("score")
        if isinstance(value, (int, float)):
            comparable_scores.append(float(value))
        prior_trial = scientific_state.get("entities", {}).get("trials", {}).get(prior.get("trial_id"), {})
        if isinstance(prior_trial, dict) and prior_trial.get("replicate_group_id") == legacy_state.get("replicate_group_id"):
            if isinstance(value, (int, float)):
                replicate_scores.append(float(value))
        started = prior.get("execution", {}).get("started_at") if isinstance(prior.get("execution"), dict) else None
        completed = prior.get("execution", {}).get("completed_at") if isinstance(prior.get("execution"), dict) else None
        try:
            if started and completed:
                comparable_runtimes.append(max(0.0, (datetime.fromisoformat(str(completed)) - datetime.fromisoformat(str(started))).total_seconds() / 3600.0))
        except ValueError:
            pass
    runtime_hours = None
    try:
        started = legacy_state.get("started_at")
        completed = _now()
        if started:
            runtime_hours = max(0.0, (datetime.fromisoformat(str(completed)) - datetime.fromisoformat(str(started))).total_seconds() / 3600.0)
    except ValueError:
        pass
    observations = generate_observations(
        experiment_id,
        run_dir=run_dir,
        score=float(summary_score) if isinstance(summary_score, (int, float)) else None,
        comparable_scores=comparable_scores,
        runtime_hours=runtime_hours,
        comparable_runtime_hours=comparable_runtimes,
        replicate_scores=replicate_scores,
        config=DetectorConfig(**campaign_config.get("observation", {})),
    )
    for observation in observations:
        observation_id = observation["id"]
        if observation_id not in scientific_state["entities"]["observations"]:
            scientific_state = record_entity(scientific_state, "observations", observation_id, observation, actor="observation_engine")
        produced_relation = {"type": "produced", "source": experiment_id, "target": observation_id}
        if not any(
            item.get("type") == produced_relation["type"]
            and item.get("source") == produced_relation["source"]
            and item.get("target") == produced_relation["target"]
            for item in scientific_state.get("relations", [])
        ):
            scientific_state = apply_operations(
                scientific_state,
                [{"operation": "relation", "value": produced_relation}],
                actor="observation_engine",
            )
    return scientific_state


def _observation_fingerprint(observation: dict[str, Any]) -> str:
    payload = {
        "type": observation.get("type"),
        "source_experiments": observation.get("source_experiments", []),
        "measurements": observation.get("measurements", {}),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _record_live_observations(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    legacy_state: dict[str, Any],
    stage_status: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run cheap deterministic detectors against the newest live artifacts."""

    trial_id = str(legacy_state.get("current_trial_id"))
    stage = str(legacy_state.get("current_stage"))
    experiment_id = f"{trial_id}:{stage}"
    artifacts = stage_status.get("artifacts", {}) if isinstance(stage_status.get("artifacts", {}), dict) else {}
    run_dir = artifacts.get("run_dir") or stage_status.get("run_dir")
    if not run_dir:
        return scientific_state, []
    observations = generate_observations(
        experiment_id,
        run_dir=run_dir,
        config=DetectorConfig(**campaign_config.get("observation", {})),
    )
    existing = {
        _observation_fingerprint(record)
        for record in scientific_state.get("entities", {}).get("observations", {}).values()
        if isinstance(record, dict)
    }
    new_observations: list[dict[str, Any]] = []
    for observation in observations:
        fingerprint = _observation_fingerprint(observation)
        if fingerprint in existing:
            continue
        existing.add(fingerprint)
        scientific_state = record_entity(
            scientific_state,
            "observations",
            observation["id"],
            observation,
            actor="live_observation_engine",
        )
        new_observations.append(observation)
    # Return the current deterministic findings even when their immutable
    # records already exist; persistence tracking and intervention triggering
    # are separate concerns.
    return scientific_state, observations


def _live_intervention_ready(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
    """Track persistent deterministic triggers and enforce the LLM cooldown."""

    observation_config = campaign_config.get("observation", {})
    controller = scientific_state.get("controller_state", {})
    trigger_types = set(observation_config.get("live_trigger_types", []))
    triggered = [item for item in observations if item.get("type") in trigger_types]
    previous_polls = int(controller.get("live_trigger_polls", 0))
    polls = previous_polls + 1 if triggered else 0
    now = datetime.now().timestamp()
    last_call = controller.get("last_live_llm_intervention_at")
    try:
        last_call_seconds = datetime.fromisoformat(str(last_call).replace("Z", "+00:00")).timestamp() if last_call else None
    except ValueError:
        last_call_seconds = None
    cooldown = float(observation_config.get("live_llm_cooldown_seconds", 3600))
    ready = bool(
        observation_config.get("live_monitor_enabled", True)
        and triggered
        and polls >= int(observation_config.get("live_min_persistent_polls", 2))
        and (last_call_seconds is None or now - last_call_seconds >= cooldown)
    )
    updated = update_controller_state(
        scientific_state,
        {
            "live_trigger_polls": polls,
            "live_trigger_types": sorted({str(item.get("type")) for item in triggered}),
            "last_live_observation_at": _now() if observations else controller.get("last_live_observation_at"),
        },
        actor="live_observation_engine",
    )
    return updated, ready, triggered


def _request_live_intervention(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    scientific_state: dict[str, Any],
    legacy_state: dict[str, Any],
    stage_status: dict[str, Any],
    controller_status: str,
    controller_reason: str,
    *,
    client: Any | None,
    stream: TextIO,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Ask the LLM only after persistent deterministic live evidence."""

    scientific_state, live_observations = _record_live_observations(
        scientific_state, campaign_config, legacy_state, stage_status
    )
    scientific_state, ready, triggered = _live_intervention_ready(
        scientific_state, campaign_config, live_observations
    )
    if not ready:
        return scientific_state, legacy_state, controller_reason
    summary = {
        "status": "live_intervention_triggered",
        "controller_status": controller_status,
        "controller_reason": controller_reason,
        "trial_id": legacy_state.get("current_trial_id"),
        "stage": legacy_state.get("current_stage"),
        "observations": triggered[-20:],
        "instruction": "Decide whether to continue, terminate this low-yield trial, or propose one bounded replacement trial. Do not infer raw metrics beyond these deterministic observations.",
    }
    decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
    scientific_state = update_controller_state(
        scientific_state,
        {"last_live_llm_intervention_at": _now(), "live_trigger_polls": 0, "last_live_decision": decision["decision"]},
        actor="autonomous_controller",
    )
    scientific_state, legacy_state, reason = _apply_decision(
        campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
    )
    if decision["decision"] == "terminate_trial" and not legacy.campaign_live_status(campaign_config).get("running"):
        followup = request_decision(
            campaign_config,
            loop_config,
            scientific_state,
            {**summary, "status": "post_termination_recovery", "termination_reason": reason},
            client=client,
        )
        scientific_state, legacy_state, followup_reason = _apply_decision(
            campaign_config, loop_config, scientific_state, legacy_state, followup, stream=stream
        )
        reason = f"{reason}; recovery: {followup_reason}"
    return scientific_state, legacy_state, reason


def _record_trial(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    legacy_state: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    trial_id = str(legacy_state.get("current_trial_id"))
    if trial_id in scientific_state["entities"]["trials"]:
        return scientific_state
    executed_stages = list(legacy_state.get("executed_stages", []))
    if not executed_stages and legacy_state.get("current_stage"):
        executed_stages = [str(legacy_state["current_stage"])]
    checkpoint_reuse = legacy_state.get("checkpoint_reuse")
    checkpoint_reuse = checkpoint_reuse if isinstance(checkpoint_reuse, dict) else {}
    trial = {
        "id": trial_id,
        "status": summary.get("status", "completed"),
        "stage_experiment_ids": [f"{trial_id}:{stage}" for stage in executed_stages],
        "purpose": {"question_id": f"q_{campaign_config['campaign']['id']}_objective", "hypothesis_ids": [f"hyp_{campaign_config['campaign']['id']}_objective"]},
        "configuration": legacy_state.get("trial_configs", {}),
        "checkpoint_reuse": legacy_state.get("checkpoint_reuse"),
        "checkpoint_manifest": legacy_state.get("checkpoint_manifest") or checkpoint_reuse.get("manifest"),
        "replicate_group_id": legacy_state.get("replicate_group_id"),
        "replicate_index": legacy_state.get("replicate_index", 0),
        "replicate_seed": legacy_state.get("replicate_seed"),
        "evaluation_protocol": campaign_config.get("campaign", {}).get("evaluation_protocol", "single_seed_historical"),
        "objective_eligibility": "protocol_pending" if campaign_config.get("campaign", {}).get("require_protocol_compliance") else "legacy_single_seed",
        "lockbox_status": "protected_not_evaluated" if campaign_config.get("campaign", {}).get("lockbox_required") else "not_configured",
        "checkpoint_status": "verified" if isinstance(legacy_state.get("checkpoint_manifest"), dict) and legacy_state.get("checkpoint_manifest", {}).get("compatibility_status") == "verified" else "legacy_unverified",
        "outcome": summary,
        "provenance": {"created_by": "autonomous_controller", "source": "campaign_trial_summary"},
    }
    updated = record_entity(scientific_state, "trials", trial_id, trial, actor="autonomous_controller")
    group_id = trial.get("replicate_group_id")
    if group_id and group_id in updated.get("entities", {}).get("replicate_groups", {}):
        relation = {"type": "replicates", "source": trial_id, "target": str(group_id)}
        if not any(
            item.get("type") == relation["type"] and item.get("source") == relation["source"] and item.get("target") == relation["target"]
            for item in updated.get("relations", [])
        ):
            updated = apply_operations(updated, [{"operation": "relation", "value": relation}], actor="autonomous_controller")
    return updated


def _candidate_start_stage(candidate: dict[str, Any], campaign_config: dict[str, Any]) -> str | None:
    """Return the stage to launch when a candidate explicitly targets one stage.

    A one-stage candidate is the scientific declaration that earlier stages are
    fixed prerequisites, not new variables.  We only skip stages for that
    explicit form; ordinary full-chain candidates retain the legacy lifecycle.
    """

    allowed_stages = candidate.get("allowed_stages")
    stages = list(campaign_config.get("campaign", {}).get("stages", []))
    if not isinstance(allowed_stages, list) or len(allowed_stages) != 1:
        return None
    target = str(allowed_stages[0])
    if target not in stages or stages.index(target) == 0:
        return None
    patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
    if not isinstance(patch, dict):
        return None
    # Do not skip a prerequisite when the candidate also changes it.
    if any(str(key) != target for key in patch):
        return None
    return target


def _reusable_checkpoint(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    target_stage: str,
    candidate: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Find the newest completed predecessor checkpoint for a stage."""

    stages = list(campaign_config.get("campaign", {}).get("stages", []))
    if target_stage not in stages or stages.index(target_stage) == 0:
        return None
    predecessor = stages[stages.index(target_stage) - 1]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    required_source = str(candidate.get("base_experiment")) if isinstance(candidate, dict) and candidate.get("base_experiment") else None
    strict_protocol = str(campaign_config.get("campaign", {}).get("evaluation_protocol", "")).startswith("three_seed")
    for experiment_id, record in scientific_state.get("entities", {}).get("experiments", {}).items():
        if required_source and str(experiment_id) != required_source:
            continue
        if not isinstance(record, dict) or str(record.get("stage")) != predecessor:
            continue
        if str(record.get("status")) != "completed":
            continue
        execution = record.get("execution", {}) if isinstance(record.get("execution", {}), dict) else {}
        artifacts = execution.get("artifacts", {}) if isinstance(execution.get("artifacts", {}), dict) else {}
        outcome = record.get("outcome", {}) if isinstance(record.get("outcome", {}), dict) else {}
        run_status = outcome.get("run_status", {}) if isinstance(outcome.get("run_status", {}), dict) else {}
        checkpoint_values = []
        if run_status.get("checkpoint_path"):
            checkpoint_values.append(run_status["checkpoint_path"])
        checkpoint_values.extend(artifacts.get("checkpoints", []) if isinstance(artifacts.get("checkpoints", []), list) else [])
        config_values = []
        config_values.extend(
            record.get("configuration", {}).get("resolved_config_paths", [])
            if isinstance(record.get("configuration", {}), dict)
            else []
        )
        run_dir = execution.get("run_dir") or artifacts.get("run_dir")
        if not run_dir:
            continue
        checkpoint = next((str(path) for path in checkpoint_values if Path(str(path)).exists()), None)
        config_path = next((str(path) for path in config_values if Path(str(path)).exists()), None)
        if checkpoint and config_path and Path(str(run_dir)).exists():
            timestamp = str(execution.get("completed_at") or record.get("created_at") or "")
            source_manifest = execution.get("checkpoint_manifest")
            if not isinstance(source_manifest, dict):
                source_manifest = record.get("checkpoint_manifest")
            if not isinstance(source_manifest, dict):
                # Historical stages predate manifests.  They can be used only
                # as explicitly marked legacy evidence; a future campaign may
                # choose to reject this status in its policy.
                source_manifest = checkpoint_manifest(
                    checkpoint,
                    source_stage=predecessor,
                    source_experiment=str(experiment_id),
                    model_family="transformer" if "transformer" in str(campaign_config.get("campaign", {}).get("id", "")) else "cnn",
                    resolved_config={"path": config_path},
                    code_revision=execution.get("code_revision"),
                    compatibility_status="legacy_unverified",
                )
            if strict_protocol and source_manifest.get("compatibility_status") != "verified":
                continue
            declared_checkpoint = candidate.get("source_checkpoint") if isinstance(candidate, dict) else None
            if declared_checkpoint and str(declared_checkpoint) not in {
                str(source_manifest.get("checkpoint_id")),
                str(source_manifest.get("checkpoint_hash")),
                str(checkpoint),
            }:
                continue
            candidates.append((timestamp, str(experiment_id), {
                "source_experiment": str(experiment_id),
                "source_stage": predecessor,
                "run_dir": str(run_dir),
                "config_path": config_path,
                "checkpoint_path": checkpoint,
                "manifest": source_manifest,
            }))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _reject_candidate(
    state: dict[str, Any],
    candidate: dict[str, Any],
    reasons: list[str],
    campaign_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    candidate_id = _candidate_id(candidate)
    rejected = dict(candidate)
    rejected["id"] = candidate_id
    rejected["status"] = "rejected"
    rejected["validation_reasons"] = list(reasons)
    rejected.setdefault("provenance", {"created_by": "autonomous_controller", "reason": "deterministic candidate policy rejection"})
    if candidate_id not in state.get("entities", {}).get("candidate_experiments", {}):
        state = record_entity(state, "candidate_experiments", candidate_id, rejected, actor="autonomous_controller")
    failures = int(state.get("controller_state", {}).get("decision_rejections", 0)) + 1
    safe_stop = failures >= int(campaign_config.get("campaign", {}).get("max_decision_retries", 3))
    updated = update_controller_state(
        state,
        {
            "decision_rejections": failures,
            "last_rejected_candidate": {"candidate": candidate, "reasons": reasons, "created_at": _now()},
            "status": "autonomous_safe_stop" if safe_stop else "running",
            "safe_stop_reason": "repeated invalid candidates" if safe_stop else None,
        },
    )
    updated = append_lifecycle_event(
        updated,
        "candidate_events",
        f"candidate_event_{stable_hash({'candidate': candidate_id, 'event': 'rejected', 'reasons': reasons})[:20]}",
        subject_id=candidate_id,
        event_type="rejected",
        payload={"reasons": list(reasons)},
        actor="autonomous_controller",
    )
    return updated, "rejected candidate: " + "; ".join(reasons)


def _apply_decision(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    scientific_state: dict[str, Any],
    legacy_state: dict[str, Any],
    decision: dict[str, Any],
    *,
    stream: TextIO,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    from src.scientific_state import apply_operations

    _validate_evidence_references(scientific_state, list(decision.get("evidence_references", [])))

    candidate_payload = decision.get("candidate")
    candidate_id = candidate_payload.get("id") if isinstance(candidate_payload, dict) else None
    operations = decision.get("operations", [])
    if candidate_id:
        operations = [
            operation
            for operation in operations
            if not (
                operation.get("operation") == "create"
                and str(operation.get("path", "")) == f"entities.candidate_experiments.{candidate_id}"
            )
        ]
    try:
        proposed_state = apply_operations(scientific_state, operations, actor="llm")
    except Exception as exc:
        failures = int(scientific_state.get("controller_state", {}).get("decision_rejections", 0)) + 1
        safe_stop = failures >= int(campaign_config.get("campaign", {}).get("max_decision_retries", 3))
        proposed_state = update_controller_state(
            scientific_state,
            {
                "decision_rejections": failures,
                "last_rejection_reason": f"{type(exc).__name__}: {exc}",
                "status": "autonomous_safe_stop" if safe_stop else "running",
                "safe_stop_reason": f"repeated invalid state operations: {exc}" if safe_stop else None,
            },
        )
        return proposed_state, legacy_state, f"rejected state operations: {type(exc).__name__}: {exc}"
    candidate = decision.get("candidate")
    if decision["decision"] == "propose_trial" and candidate is None:
        return scientific_state, legacy_state, "rejected proposal: propose_trial requires candidate"
    if candidate is not None:
        candidate = dict(candidate)
        candidate.setdefault("id", _candidate_id(candidate))
        candidate.setdefault("status", "proposed")
        candidate.setdefault("replicate_seeds", _candidate_seeds(candidate, campaign_config))
        candidate.setdefault("replicate_group_id", candidate["id"])
        validation = validate_candidate(
            candidate,
            loop_config=loop_config,
            campaign_config=campaign_config,
            state=proposed_state,
            active_process=decision["decision"] == "propose_trial" and _campaign_process_active(campaign_config),
        )
        if not validation.valid:
            proposed_state, reason = _reject_candidate(
                proposed_state, candidate, list(validation.reasons), campaign_config
            )
            return proposed_state, legacy_state, reason
        start_stage = _candidate_start_stage(candidate, campaign_config)
        checkpoint = _reusable_checkpoint(proposed_state, campaign_config, start_stage, candidate) if start_stage else None
        if start_stage and checkpoint is None:
            proposed_state, reason = _reject_candidate(
                proposed_state,
                candidate,
                [
                    f"no completed compatible {campaign_config['campaign']['stages'][campaign_config['campaign']['stages'].index(start_stage) - 1]} checkpoint is available for direct {start_stage} launch"
                ],
                campaign_config,
            )
            return proposed_state, legacy_state, reason
        # Candidate records are semantic state and therefore get provenance.
        candidate.setdefault("provenance", {"created_by": "llm", "decision": decision["decision"]})
        if candidate["id"] not in proposed_state["entities"]["candidate_experiments"]:
            if decision["decision"] == "propose_trial":
                candidate["status"] = "selected_for_execution"
            proposed_state = record_entity(proposed_state, "candidate_experiments", candidate["id"], candidate, actor="llm")
            proposed_state = append_lifecycle_event(
                proposed_state,
                "candidate_events",
                f"candidate_event_{stable_hash({'candidate': candidate['id'], 'event': candidate.get('status')})[:20]}",
                subject_id=str(candidate["id"]),
                event_type=str(candidate.get("status", "proposed")),
                payload={"decision": decision["decision"], "reason": decision.get("reason", "")},
                actor="autonomous_controller",
            )
        elif decision["decision"] == "propose_trial":
            existing_candidate = proposed_state["entities"]["candidate_experiments"][candidate["id"]]
            if existing_candidate.get("status") == "proposed":
                proposed_state = apply_operations(
                    proposed_state,
                    [{
                        "operation": "transition",
                        "path": f"entities.candidate_experiments.{candidate['id']}.status",
                        "value": "selected_for_execution",
                        "expected_old": "proposed",
                    }],
                    actor="autonomous_controller",
                )
        if decision["decision"] == "propose_trial":
            patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
            replicate_seed = decision.get("_replicate_seed")
            replicate_index = int(decision.get("_replicate_index", 0))
            replicate_group_id = str(candidate.get("replicate_group_id") or candidate["id"])
            existing_reservations = proposed_state.get("controller_state", {}).get("launch_reservations", {})
            if isinstance(existing_reservations, dict):
                for existing in existing_reservations.values():
                    if (
                        isinstance(existing, dict)
                        and existing.get("candidate_id") == candidate["id"]
                        and existing.get("replicate_group_id") == replicate_group_id
                        and int(existing.get("replicate_index", -1)) == replicate_index
                        and existing.get("status") in {"reserved", "launched"}
                    ):
                        if _campaign_process_active(campaign_config):
                            return proposed_state, legacy_state, f"launch already reserved for candidate {candidate['id']} replicate {replicate_index + 1}"
                        proposed_state = update_launch_reservation(
                            proposed_state,
                            str(existing.get("id")),
                            status="failed",
                            actor="autonomous_recovery",
                        )
            if replicate_group_id not in proposed_state.get("entities", {}).get("replicate_groups", {}):
                proposed_state = record_entity(
                    proposed_state,
                    "replicate_groups",
                    replicate_group_id,
                    {
                        "id": replicate_group_id,
                        "candidate_id": candidate["id"],
                        "seeds": list(candidate["replicate_seeds"]),
                        "minimum_replicates": int(campaign_config.get("campaign", {}).get("minimum_replicates", 3)),
                        "status": "selected",
                        "evaluation_protocol": campaign_config.get("campaign", {}).get("evaluation_protocol", "three_seed_replicate_lockbox_v1"),
                        "fixed_variables": candidate.get("fixed_variables", {}),
                        "provenance": {"created_by": "autonomous_controller", "candidate_id": candidate["id"]},
                    },
                    actor="autonomous_controller",
                )
                proposed_state = append_lifecycle_event(
                    proposed_state,
                    "replicate_group_events",
                    f"replicate_group_event_{stable_hash({'group': replicate_group_id, 'event': 'selected'})[:20]}",
                    subject_id=replicate_group_id,
                    event_type="selected",
                    payload={"candidate_id": candidate["id"], "seeds": list(candidate["replicate_seeds"])},
                    actor="autonomous_controller",
                )
            launch_kwargs: dict[str, Any] = {
                "trial_patch": patch,
                "dry_run": False,
                "replicate_group_id": replicate_group_id,
                "replicate_index": replicate_index,
                "replicate_seed": int(replicate_seed) if replicate_seed is not None else int(candidate["replicate_seeds"][0]),
            }
            if start_stage and checkpoint:
                launch_kwargs.update(
                    {
                        "start_stage": start_stage,
                        "checkpoint_run_dir": checkpoint["run_dir"],
                        "checkpoint_config_path": checkpoint["config_path"],
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_source_experiment": checkpoint["source_experiment"],
                        "checkpoint_manifest": checkpoint.get("manifest"),
                    }
                )
            cost = candidate.get("cost") if isinstance(candidate.get("cost"), dict) else {}
            estimated_gpu_hours = float(cost.get("estimated_gpu_hours", candidate.get("estimated_gpu_hours", 0.0)))
            # Reserve the exact trial identity before creating the process. The
            # legacy launcher accepts this ID, making retries idempotent.
            launched_trial_id = legacy._trial_id(str(campaign_config["campaign"]["id"]))
            reservation_id = "reservation_" + stable_hash({
                "candidate_id": candidate["id"],
                "replicate_group_id": replicate_group_id,
                "replicate_index": replicate_index,
                "trial_id": launched_trial_id,
            })[:24]
            proposed_state = reserve_launch(
                proposed_state,
                reservation_id=reservation_id,
                candidate_id=str(candidate["id"]),
                trial_id=launched_trial_id,
                estimated_gpu_hours=estimated_gpu_hours,
                replicate_group_id=replicate_group_id,
                replicate_index=replicate_index,
                actor="autonomous_controller",
            )
            proposed_state = append_lifecycle_event(
                proposed_state,
                "launch_events",
                f"launch_event_{stable_hash({'reservation': reservation_id, 'event': 'reserved'})[:20]}",
                subject_id=reservation_id,
                event_type="reserved",
                payload={"candidate_id": candidate["id"], "trial_id": launched_trial_id},
                actor="autonomous_controller",
            )
            launch_kwargs["start_trial"] = launched_trial_id
            state_path = _state_path(campaign_config)
            if state_path.exists() and campaign_config.get("campaign", {}).get("evaluation_protocol"):
                save_state(state_path, proposed_state)
            try:
                next_legacy = _launch_trial_guarded(campaign_config, loop_config, launch_kwargs)
            except Exception:
                proposed_state = update_launch_reservation(
                    proposed_state, reservation_id, status="failed", actor="autonomous_controller"
                )
                if state_path.exists() and campaign_config.get("campaign", {}).get("evaluation_protocol"):
                    save_state(state_path, proposed_state)
                raise
            proposed_state = update_launch_reservation(
                proposed_state, reservation_id, status="launched", actor="autonomous_controller"
            )
            current_group = proposed_state.get("controller_state", {}).get("active_replicate_group", {})
            completed_trial_ids = []
            if isinstance(current_group, dict) and current_group.get("id") == replicate_group_id:
                completed_trial_ids = list(current_group.get("completed_trial_ids", []))
            proposed_state = update_controller_state(
                proposed_state,
                {
                    "status": "running",
                    "active_trial_id": next_legacy.get("current_trial_id"),
                    "last_decision": decision["decision"],
                    "last_decision_at": _now(),
                    "decision_rejections": 0,
                    "last_rejection_reason": None,
                    "safe_stop_reason": None,
                    "active_replicate_group": {
                        "id": replicate_group_id,
                        "candidate_id": candidate["id"],
                        "seeds": list(candidate["replicate_seeds"]),
                        "completed_trial_ids": completed_trial_ids,
                        "status": "running",
                    },
                    "active_launch_reservation_id": reservation_id,
                },
            )
            return proposed_state, next_legacy, f"launched candidate {candidate['id']}"
    if decision["decision"] == "terminate_trial":
        force_after = float(campaign_config.get("observation", {}).get("live_termination_force_after", 5))
        return_code = legacy.terminate_campaign(
            campaign_config,
            reason=decision.get("reason") or "deterministic live intervention requested termination",
            force_after=force_after,
            stream=stream,
        )
        refreshed_legacy = legacy._read_json(legacy._campaign_state_path(campaign_config))
        proposed_state = update_controller_state(
            proposed_state,
            {
                "status": "running" if return_code == 0 else "autonomous_safe_stop",
                "last_decision": decision["decision"],
                "last_decision_at": _now(),
                "termination_returncode": return_code,
            },
        )
        return proposed_state, refreshed_legacy or legacy_state, (
            f"terminated live trial with returncode={return_code}: {decision.get('reason', 'no reason')}"
        )
    if decision["decision"] == "stop_campaign":
        proposed_state = update_controller_state(
            proposed_state,
            {"status": "autonomous_safe_stop", "safe_stop_reason": decision.get("reason"), "last_decision_at": _now()},
        )
        return proposed_state, legacy_state, "autonomous safe stop: " + decision.get("reason", "no reason")
    if decision["decision"] in {"continue", "no_action"} and legacy_state.get("status") in {"trial_completed", "failed"}:
        proposed_state = update_controller_state(
            proposed_state,
            {
                "status": "autonomous_safe_stop",
                "safe_stop_reason": decision.get("reason") or "no safe autonomous candidate was proposed",
                "last_decision": decision["decision"],
                "last_decision_at": _now(),
            },
        )
        return proposed_state, legacy_state, "autonomous safe stop: no next candidate"
    proposed_state = update_controller_state(
        proposed_state,
        {"status": "running", "last_decision": decision["decision"], "last_decision_at": _now(), "decision_rejections": 0, "safe_stop_reason": None},
    )
    return proposed_state, legacy_state, decision.get("reason") or decision["decision"]


def _selected_recovery_candidate(scientific_state: dict[str, Any]) -> dict[str, Any] | None:
    """Return an explicitly selected candidate after an interrupted launch.

    This is a deterministic recovery path, not a new scientific decision: the
    candidate was already selected and recorded before the previous process
    ended.  It prevents a transient LLM no_action response from stranding the
    campaign between trials.
    """

    candidates = scientific_state.get("entities", {}).get("candidate_experiments", {})
    selected = [
        value
        for value in candidates.values()
        if isinstance(value, dict) and value.get("status") == "selected_for_execution"
    ]
    if not selected:
        return None
    selected.sort(key=lambda value: str(value.get("created_at", "")), reverse=True)
    return dict(selected[0])


def _candidate_seeds(candidate: dict[str, Any], campaign_config: dict[str, Any]) -> list[int]:
    configured = candidate.get("replicate_seeds") or campaign_config.get("campaign", {}).get("replicate_seeds", [0, 1, 2])
    seeds = [int(value) for value in configured]
    return list(dict.fromkeys(seeds)) or [0, 1, 2]


def _mark_replicate_completion(
    scientific_state: dict[str, Any],
    legacy_state: dict[str, Any],
) -> dict[str, Any]:
    group = scientific_state.get("controller_state", {}).get("active_replicate_group")
    group = dict(group) if isinstance(group, dict) else None
    group_id = legacy_state.get("replicate_group_id")
    if not group or not group_id or group.get("id") != group_id:
        return scientific_state
    completed = list(group.get("completed_trial_ids", []))
    trial_id = legacy_state.get("current_trial_id")
    if trial_id and trial_id not in completed:
        completed.append(trial_id)
    updated = update_controller_state(
        scientific_state,
        {"active_replicate_group": {**group, "completed_trial_ids": completed}},
        actor="autonomous_controller",
    )
    event_id = f"replicate_group_event_{stable_hash({'group': group_id, 'trial': trial_id, 'event': 'completed'})[:20]}"
    if event_id not in updated.get("entities", {}).get("replicate_group_events", {}):
        updated = append_lifecycle_event(
            updated,
            "replicate_group_events",
            event_id,
            subject_id=str(group_id),
            event_type="replicate_completed",
            payload={"trial_id": trial_id, "completed_trial_ids": completed},
            actor="autonomous_controller",
        )
    return updated


def _replicate_candidate_and_seed(
    scientific_state: dict[str, Any],
) -> tuple[dict[str, Any], int, int] | None:
    group = scientific_state.get("controller_state", {}).get("active_replicate_group")
    if not isinstance(group, dict):
        return None
    seeds = [int(value) for value in group.get("seeds", [])]
    completed = list(group.get("completed_trial_ids", []))
    if len(completed) >= len(seeds):
        return None
    candidate_id = group.get("candidate_id")
    candidate = scientific_state.get("entities", {}).get("candidate_experiments", {}).get(candidate_id)
    if not isinstance(candidate, dict):
        return None
    index = len(completed)
    return dict(candidate), seeds[index], index


def _replicate_aggregate(scientific_state: dict[str, Any]) -> dict[str, Any] | None:
    group = scientific_state.get("controller_state", {}).get("active_replicate_group")
    if not isinstance(group, dict):
        return None
    group_id = group.get("id")
    trials = [
        record
        for record in scientific_state.get("entities", {}).get("trials", {}).values()
        if isinstance(record, dict) and record.get("replicate_group_id") == group_id
    ]
    scores: list[float] = []
    guardrail_values: list[float] = []
    for trial in trials:
        outcome = trial.get("outcome", {}) if isinstance(trial.get("outcome"), dict) else {}
        stage_ids = trial.get("stage_experiment_ids", []) if isinstance(trial.get("stage_experiment_ids"), list) else []
        stage_records = [
            scientific_state.get("entities", {}).get("experiments", {}).get(stage_id, {})
            for stage_id in stage_ids
        ]
        final_record = stage_records[-1] if stage_records else {}
        metrics = final_record.get("outcome", {}).get("summary_metrics", {}) if isinstance(final_record, dict) else {}
        score = metrics.get("compound.macro_f1") if isinstance(metrics, dict) else None
        if score is None:
            score = outcome.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        guardrail = metrics.get("action.accuracy") if isinstance(metrics, dict) else None
        if isinstance(guardrail, (int, float)):
            guardrail_values.append(float(guardrail))
    aggregate = aggregate_replicates([value for value in scores if value is not None], minimum_replicates=len(group.get("seeds", [])) or 3)
    guardrail = aggregate_guardrail(guardrail_values, minimum=0.30, minimum_replicates=len(group.get("seeds", [])) or 3)
    valid = aggregate.status == "replicate_complete" and guardrail["all_replicates_pass"]
    return {
        "score": aggregate.as_dict(),
        "guardrail": guardrail,
        "trial_count": len(trials),
        "objective_eligible": valid,
        "evidence_status": "replicate_supported" if valid else "invalid_or_incomplete_evidence",
        "lockbox_status": "protected_not_evaluated",
        "protocol_version": "replicate-lockbox-v1",
    }


def _campaign_process_active(campaign_config: dict[str, Any]) -> bool:
    try:
        return bool(legacy.campaign_live_status(campaign_config).get("running"))
    except (KeyError, OSError, ValueError):
        return False


def _launch_trial_guarded(
    campaign_config: dict[str, Any],
    loop_config: dict[str, Any],
    launch_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Launch exactly once, sharing the campaign lock with the supervisor."""

    if not isinstance(campaign_config.get("artifacts"), dict):
        return legacy.start_trial(campaign_config, loop_config, **launch_kwargs)
    if _campaign_process_active(campaign_config):
        raise RuntimeError("another campaign process is active")
    lock_path = legacy._campaign_lock_path(campaign_config)
    owner = legacy._read_lock_pid(lock_path)
    acquired = False
    if owner not in {None, os.getpid()}:
        raise RuntimeError(f"campaign launch lock is owned by pid={owner}")
    if owner is None:
        lock_path = legacy._acquire_campaign_lock(campaign_config)
        acquired = True
    try:
        return legacy.start_trial(campaign_config, loop_config, **launch_kwargs)
    finally:
        if acquired:
            legacy._release_campaign_lock(lock_path)


def _run_autonomous_campaign_body(
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
    loop_config = legacy._load_loop_config(campaign_config)
    campaign_config.setdefault("agent", dict(loop_config.get("agent", {})))
    campaign_config["agent"] = {**dict(loop_config.get("agent", {})), **dict(campaign_config.get("agent", {}))}
    campaign_config["observation"] = {
        **dict(loop_config.get("observation", {})),
        **dict(campaign_config.get("observation", {})),
    }
    if not dry_run:
        experiment_loop.validate_api_key({"agent": campaign_config["agent"]})
    scientific_path = _state_path(campaign_config)
    scientific_state = _ensure_seed_knowledge(load_state(scientific_path), campaign_config)
    legacy_state = legacy._read_json(legacy._campaign_state_path(campaign_config))
    if new_trial or not legacy_state or legacy_state.get("status") in {"campaign_completed", "analysis_completed", "autonomous_safe_stop"}:
        if dry_run:
            return 0
        summary = {"status": "initializing", "trial_id": start_trial_id or "pending"}
        try:
            decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
        except Exception as exc:
            scientific_state = update_controller_state(
                scientific_state,
                {"status": "autonomous_safe_stop", "safe_stop_reason": f"decision protocol failure: {type(exc).__name__}: {exc}", "last_decision_error_at": _now()},
                actor="autonomous_controller",
            )
            transactional_update(scientific_path, lambda current: merge_nonconflicting_states(current, scientific_state))
            print(f"autonomous safe stop: decision protocol failure {type(exc).__name__}: {exc}", file=stream, flush=True)
            return 1
        if decision["decision"] in {"no_action", "continue"}:
            recovery_candidate = _selected_recovery_candidate(scientific_state)
            if recovery_candidate is not None:
                decision = {
                    "decision": "propose_trial",
                    "reason": "deterministically recovering the previously selected candidate after interruption",
                    "operations": [],
                    "candidate": recovery_candidate,
                    "evidence_references": [],
                }
        scientific_state, legacy_state, reason = _apply_decision(
            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
        )
        transactional_update(scientific_path, lambda current: merge_nonconflicting_states(current, scientific_state))
        print(f"autonomous initialization decision={decision['decision']} reason={reason}", file=stream, flush=True)
        if scientific_state.get("controller_state", {}).get("status") == "autonomous_safe_stop":
            return 1
    poll_seconds = int(campaign_config["campaign"].get("poll_seconds", campaign_config["agent"].get("poll_seconds", 3600)))
    first_poll = True
    lock_path = None
    try:
        while True:
            if not first_poll and not once:
                sleep_fn(float(poll_seconds))
            first_poll = False
            latest = legacy._read_json(legacy._campaign_state_path(campaign_config))
            if latest:
                legacy_state = latest
            if not dry_run:
                watchdog = inspect_campaign(
                    campaign_config,
                    campaign_state_path=legacy._campaign_state_path(campaign_config),
                    stale_after_seconds=float(campaign_config.get("observation", {}).get("watchdog_stale_after_seconds", 180.0)),
                )
                scientific_state = update_controller_state(
                    scientific_state,
                    {"watchdog": watchdog},
                    actor="campaign_watchdog",
                )
            if scientific_state.get("controller_state", {}).get("status") == "autonomous_safe_stop":
                return 1
            if not legacy_state.get("current_stage"):
                return 0
            status, controller_status, controller_reason = legacy._collect_stage_status(campaign_config, loop_config, legacy_state)
            stage_state_path = legacy_state.get("stage_state_path")
            if stage_state_path and not dry_run:
                experiment_loop.update_controller_state(
                    legacy._loop_config_for_stage(loop_config, Path(legacy_state["current_trial_dir"]), str(legacy_state["current_stage"]), legacy_state["trial_configs"]),
                    status,
                    controller_status,
                    controller_reason,
                    state_path=Path(str(stage_state_path)),
                )
            if controller_status in {"running_progress", "running_wait"}:
                action = "wait"
                reason = controller_reason
                if not dry_run:
                    try:
                        scientific_state, legacy_state, live_reason = _request_live_intervention(
                            campaign_config,
                            loop_config,
                            scientific_state,
                            legacy_state,
                            status,
                            controller_status,
                            controller_reason,
                            client=client,
                            stream=stream,
                        )
                        reason = live_reason
                        if live_reason != controller_reason:
                            action = "live_intervention"
                    except Exception as exc:
                        # A failed LLM intervention must not kill the training
                        # process. Keep deterministic monitoring alive and let
                        # the next cooldown window retry it.
                        scientific_state = update_controller_state(
                            scientific_state,
                            {
                                "status": "running",
                                "last_live_intervention_error": f"{type(exc).__name__}: {exc}",
                                "last_live_intervention_error_at": _now(),
                            },
                            actor="autonomous_controller",
                        )
                        action = "live_intervention_retry"
                        reason = f"{controller_reason}; live LLM intervention deferred: {type(exc).__name__}: {exc}"
            elif controller_status == "completed":
                scientific_state = _record_stage(scientific_state, campaign_config, legacy_state, status, controller_status)
                legacy_state, action = legacy._advance_stage_or_complete_trial(campaign_config, loop_config, legacy_state, status, dry_run=dry_run)
                reason = controller_reason
                if legacy_state.get("status") == "trial_completed" and not dry_run:
                    summary = _summary_for_state(campaign_config, legacy_state)
                    scientific_state = _record_trial(scientific_state, campaign_config, legacy_state, summary)
                    scientific_state = _mark_replicate_completion(scientific_state, legacy_state)
                    replicate = _replicate_candidate_and_seed(scientific_state)
                    if replicate is not None:
                        candidate, seed, index = replicate
                        decision = {
                            "decision": "propose_trial",
                            "reason": f"launching replicate {index + 1} with fixed seed {seed}",
                            "operations": [],
                            "candidate": candidate,
                            "evidence_references": [],
                            "_replicate_seed": seed,
                            "_replicate_index": index,
                        }
                        scientific_state, legacy_state, reason = _apply_decision(
                            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
                        )
                        action = "replicate"
                    else:
                        aggregate = _replicate_aggregate(scientific_state)
                        if aggregate is not None:
                            scientific_state = update_controller_state(
                                scientific_state,
                                {"last_replicate_aggregate": aggregate},
                                actor="deterministic_statistics",
                            )
                            group_id = scientific_state.get("controller_state", {}).get("active_replicate_group", {}).get("id")
                            if group_id:
                                update_id = f"belief_update_{stable_hash({'group': group_id, 'aggregate': aggregate})[:16]}"
                                if update_id not in scientific_state.get("entities", {}).get("belief_updates", {}):
                                    hypothesis_id = f"hyp_{campaign_config['campaign']['id']}_objective"
                                    previous = scientific_state.get("entities", {}).get("hypotheses", {}).get(hypothesis_id, {}).get("belief", {})
                                    scientific_state = append_belief_update(
                                        scientific_state,
                                        update_id,
                                        {
                                            "belief_id": hypothesis_id,
                                            "previous_belief": previous,
                                            "new_belief": {"status": "unresolved", "score": None},
                                            "triggering_observations": [],
                                            "rationale": "Replicate aggregate recorded; a causal belief update is deferred until a preregistered comparison or lockbox confirmation exists.",
                                            "method": "deterministic_replicate_gate",
                                            "calibration_status": "not_calibrated",
                                            "candidate_id": scientific_state.get("controller_state", {}).get("active_replicate_group", {}).get("candidate_id"),
                                        },
                                        actor="deterministic_statistics",
                                    )
                            summary = {**summary, "replicate_aggregate": aggregate}
                        try:
                            decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
                        except Exception as exc:
                            scientific_state = update_controller_state(
                                scientific_state,
                                {"status": "autonomous_safe_stop", "safe_stop_reason": f"decision protocol failure: {type(exc).__name__}: {exc}", "last_decision_error_at": _now()},
                                actor="autonomous_controller",
                            )
                            transactional_update(scientific_path, lambda current: merge_nonconflicting_states(current, scientific_state))
                            print(f"autonomous safe stop: decision protocol failure {type(exc).__name__}: {exc}", file=stream, flush=True)
                            return 1
                        scientific_state, legacy_state, reason = _apply_decision(
                            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
                        )
                        action = decision["decision"]
            elif controller_status in {"failed", "running_stale"}:
                scientific_state = _record_stage(scientific_state, campaign_config, legacy_state, status, controller_status)
                if controller_status == "running_stale":
                    action = "wait_stale"
                    reason = controller_reason
                else:
                    legacy_state["status"] = "failed"
                    summary = _summary_for_state(campaign_config, legacy_state)
                    recovery_candidate = _selected_recovery_candidate(scientific_state)
                    active_reservation_id = scientific_state.get("controller_state", {}).get("active_launch_reservation_id")
                    if recovery_candidate is not None and active_reservation_id:
                        reservations = scientific_state.get("controller_state", {}).get("launch_reservations", {})
                        reservation = reservations.get(active_reservation_id) if isinstance(reservations, dict) else None
                        if isinstance(reservation, dict) and reservation.get("status") in {"reserved", "launched"}:
                            scientific_state = update_launch_reservation(
                                scientific_state,
                                str(active_reservation_id),
                                status="failed",
                                actor="autonomous_recovery",
                            )
                        group = scientific_state.get("controller_state", {}).get("active_replicate_group", {})
                        replicate_index = int(recovery_candidate.get("replicate_index", len(group.get("completed_trial_ids", [])) if isinstance(group, dict) else 0))
                        seeds = _candidate_seeds(recovery_candidate, campaign_config)
                        decision = {
                            "decision": "propose_trial",
                            "reason": "deterministically recovering the same selected replicate after process failure",
                            "operations": [],
                            "candidate": recovery_candidate,
                            "evidence_references": [],
                            "_replicate_seed": seeds[min(replicate_index, len(seeds) - 1)],
                            "_replicate_index": replicate_index,
                        }
                        scientific_state, legacy_state, reason = _apply_decision(
                            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
                        )
                        action = "recover_replicate"
                    else:
                        try:
                            decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
                        except Exception as exc:
                            scientific_state = update_controller_state(
                                scientific_state,
                                {"status": "autonomous_safe_stop", "safe_stop_reason": f"decision protocol failure: {type(exc).__name__}: {exc}", "last_decision_error_at": _now()},
                                actor="autonomous_controller",
                            )
                            transactional_update(scientific_path, lambda current: merge_nonconflicting_states(current, scientific_state))
                            print(f"autonomous safe stop: decision protocol failure {type(exc).__name__}: {exc}", file=stream, flush=True)
                            return 1
                        scientific_state, legacy_state, reason = _apply_decision(
                            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
                        )
                        action = decision["decision"]
            else:
                action = "wait"
                reason = controller_reason
            scientific_state = update_controller_state(
                scientific_state,
                {"active_trial_id": legacy_state.get("current_trial_id"), "active_stage": legacy_state.get("current_stage"), "last_poll_at": _now(), "last_controller_status": controller_status},
            )
            if not dry_run:
                transactional_update(scientific_path, lambda current: merge_nonconflicting_states(current, scientific_state))
            print(
                f"[{_now()}] autonomous campaign={campaign_config['campaign']['id']} trial={legacy_state.get('current_trial_id')} stage={legacy_state.get('current_stage')} controller={controller_status} action={action} reason={reason}",
                file=stream,
                flush=True,
            )
            if once:
                return 0
    except KeyboardInterrupt:
        if terminate_child_on_exit:
            active_pid = legacy_state.get("active_launch_state", {}).get("pid")
            if isinstance(active_pid, int) and experiment_loop._is_process_running(active_pid):
                import os

                os.killpg(active_pid, 15)
        print("autonomous campaign interrupted; artifacts and scientific state preserved", file=stream, flush=True)
        return 130
    finally:
        if lock_path is not None:
            legacy._release_campaign_lock(lock_path)


def run_autonomous_campaign(
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
    """Run the campaign under one lock covering decision, launch, and polling."""

    lock_path = None
    if not dry_run:
        lock_path = legacy._acquire_campaign_lock(campaign_config)
    try:
        return _run_autonomous_campaign_body(
            campaign_config,
            once=once,
            dry_run=dry_run,
            start_trial_id=start_trial_id,
            new_trial=new_trial,
            terminate_child_on_exit=terminate_child_on_exit,
            client=client,
            sleep_fn=sleep_fn,
            stream=stream,
        )
    finally:
        if lock_path is not None:
            legacy._release_campaign_lock(lock_path)
