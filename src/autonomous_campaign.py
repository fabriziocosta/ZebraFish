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
from src.autonomous_policy import CandidateValidation, validate_candidate
from src.observation_engine import DetectorConfig, generate_observations
from src.scientific_state import (
    apply_operations,
    compact_context,
    ENTITY_COLLECTIONS,
    empty_state,
    load_state,
    record_entity,
    save_state,
    update_controller_state,
)


DECISIONS = {"continue", "propose_trial", "terminate_trial", "stop_campaign", "no_action"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
                "status": "active",
                "belief": {"probability": 0.5, "confidence": "initial"},
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
                            "configuration_patch",
                            "fixed_variables",
                            "expected_outcomes",
                            "falsification_criteria",
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
                            "configuration_patch": {"type": "string", "description": "JSON-encoded stage-keyed YAML patch."},
                            "fixed_variables": {"type": "string", "description": "JSON-encoded mapping."},
                            "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                            "falsification_criteria": {"type": "array", "items": {"type": "string"}},
                            "estimated_gpu_hours": {"type": "number"},
                            "estimated_wall_hours": {"type": "number"},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "allowed_stages": {"type": "array", "items": {"type": "string"}},
                            "baseline": {"type": "boolean"},
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
    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    normalized_operations: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("each operation must be an object")
        normalized = dict(operation)
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
        for field in ("configuration_patch", "fixed_variables"):
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
    record = {
        "id": experiment_id,
        "trial_id": trial_id,
        "stage": stage,
        "status": "completed" if controller_status == "completed" else controller_status,
        "configuration": {
            "resolved_config_paths": artifacts.get("latest_config_yamls", []),
            "trial_config": legacy_state.get("trial_configs", {}).get(stage),
        },
        "execution": {
            "run_dir": run_dir,
            "started_at": legacy_state.get("started_at"),
            "completed_at": _now(),
            "artifacts": artifacts,
            "provenance": {"created_by": "autonomous_controller", "legacy_state": legacy_state.get("stage_state_path")},
        },
        "outcome": {"controller_status": controller_status, "run_status": stage_status.get("run_status", {})},
        "provenance": {"created_by": "autonomous_controller", "source": "campaign_stage_status"},
    }
    scientific_state = record_entity(scientific_state, "experiments", experiment_id, record, actor="autonomous_controller")
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
    observations = generate_observations(
        experiment_id,
        run_dir=run_dir,
        config=DetectorConfig(**campaign_config.get("observation", {})),
    )
    for observation in observations:
        observation_id = observation["id"]
        if observation_id not in scientific_state["entities"]["observations"]:
            scientific_state = record_entity(scientific_state, "observations", observation_id, observation, actor="observation_engine")
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
    trial = {
        "id": trial_id,
        "status": summary.get("status", "completed"),
        "stage_experiment_ids": [f"{trial_id}:{stage}" for stage in executed_stages],
        "purpose": {"question_id": f"q_{campaign_config['campaign']['id']}_objective", "hypothesis_ids": [f"hyp_{campaign_config['campaign']['id']}_objective"]},
        "configuration": legacy_state.get("trial_configs", {}),
        "checkpoint_reuse": legacy_state.get("checkpoint_reuse"),
        "outcome": summary,
        "provenance": {"created_by": "autonomous_controller", "source": "campaign_trial_summary"},
    }
    return record_entity(scientific_state, "trials", trial_id, trial, actor="autonomous_controller")


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
) -> dict[str, str] | None:
    """Find the newest completed predecessor checkpoint for a stage."""

    stages = list(campaign_config.get("campaign", {}).get("stages", []))
    if target_stage not in stages or stages.index(target_stage) == 0:
        return None
    predecessor = stages[stages.index(target_stage) - 1]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for experiment_id, record in scientific_state.get("entities", {}).get("experiments", {}).items():
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
            candidates.append((timestamp, str(experiment_id), {
                "source_experiment": str(experiment_id),
                "source_stage": predecessor,
                "run_dir": str(run_dir),
                "config_path": config_path,
                "checkpoint_path": checkpoint,
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
        validation = validate_candidate(
            candidate,
            loop_config=loop_config,
            campaign_config=campaign_config,
            state=proposed_state,
            active_process=False,
        )
        if not validation.valid:
            proposed_state, reason = _reject_candidate(
                proposed_state, candidate, list(validation.reasons), campaign_config
            )
            return proposed_state, legacy_state, reason
        start_stage = _candidate_start_stage(candidate, campaign_config)
        checkpoint = _reusable_checkpoint(proposed_state, campaign_config, start_stage) if start_stage else None
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
            proposed_state = record_entity(proposed_state, "candidate_experiments", candidate["id"], candidate, actor="llm")
        if decision["decision"] == "propose_trial":
            patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
            launch_kwargs: dict[str, Any] = {"trial_patch": patch, "dry_run": False}
            if start_stage and checkpoint:
                launch_kwargs.update(
                    {
                        "start_stage": start_stage,
                        "checkpoint_run_dir": checkpoint["run_dir"],
                        "checkpoint_config_path": checkpoint["config_path"],
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_source_experiment": checkpoint["source_experiment"],
                    }
                )
            next_legacy = legacy.start_trial(campaign_config, loop_config, **launch_kwargs)
            proposed_state = update_controller_state(
                proposed_state,
                {"status": "running", "active_trial_id": next_legacy.get("current_trial_id"), "last_decision": decision["decision"], "last_decision_at": _now(), "decision_rejections": 0, "last_rejection_reason": None, "safe_stop_reason": None},
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
        decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
        scientific_state, legacy_state, reason = _apply_decision(
            campaign_config, loop_config, scientific_state, legacy_state, decision, stream=stream
        )
        save_state(scientific_path, scientific_state)
        print(f"autonomous initialization decision={decision['decision']} reason={reason}", file=stream, flush=True)
        if scientific_state.get("controller_state", {}).get("status") == "autonomous_safe_stop":
            return 1
    poll_seconds = int(campaign_config["campaign"].get("poll_seconds", campaign_config["agent"].get("poll_seconds", 3600)))
    first_poll = True
    lock_path = legacy._acquire_campaign_lock(campaign_config) if not dry_run else None
    try:
        while True:
            if not first_poll and not once:
                sleep_fn(float(poll_seconds))
            first_poll = False
            latest = legacy._read_json(legacy._campaign_state_path(campaign_config))
            if latest:
                legacy_state = latest
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
                    decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
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
                    decision = request_decision(campaign_config, loop_config, scientific_state, summary, client=client)
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
                save_state(scientific_path, scientific_state)
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
