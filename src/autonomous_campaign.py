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
    response = client.responses.create(
        model=agent.get("model", "gpt-5.3-codex"),
        reasoning={"effort": agent.get("reasoning_effort", "medium")},
        max_output_tokens=int(agent.get("max_output_tokens", 6000)),
        text=_decision_schema(),
        input=_build_prompt(campaign_config, loop_config, state, summary, validation_error=validation_error),
    )
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str):
        raise ValueError("OpenAI response did not contain output_text")
    return parse_decision(output_text)


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
        previous_id = f"{trial_id}:{stages[stages.index(stage) - 1]}"
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


def _record_trial(
    scientific_state: dict[str, Any],
    campaign_config: dict[str, Any],
    legacy_state: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    trial_id = str(legacy_state.get("current_trial_id"))
    if trial_id in scientific_state["entities"]["trials"]:
        return scientific_state
    trial = {
        "id": trial_id,
        "status": summary.get("status", "completed"),
        "stage_experiment_ids": [f"{trial_id}:{stage}" for stage in campaign_config["campaign"]["stages"]],
        "purpose": {"question_id": f"q_{campaign_config['campaign']['id']}_objective", "hypothesis_ids": [f"hyp_{campaign_config['campaign']['id']}_objective"]},
        "configuration": legacy_state.get("trial_configs", {}),
        "outcome": summary,
        "provenance": {"created_by": "autonomous_controller", "source": "campaign_trial_summary"},
    }
    return record_entity(scientific_state, "trials", trial_id, trial, actor="autonomous_controller")


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
            failures = int(proposed_state.get("controller_state", {}).get("decision_rejections", 0)) + 1
            safe_stop = failures >= int(campaign_config.get("campaign", {}).get("max_decision_retries", 3))
            proposed_state = update_controller_state(
                proposed_state,
                {
                    "decision_rejections": failures,
                    "last_rejected_candidate": {
                        "candidate": candidate,
                        "reasons": list(validation.reasons),
                        "created_at": _now(),
                    },
                    "status": "autonomous_safe_stop" if safe_stop else "running",
                    "safe_stop_reason": "repeated invalid candidates" if safe_stop else None,
                },
            )
            return proposed_state, legacy_state, "rejected candidate: " + "; ".join(validation.reasons)
        # Candidate records are semantic state and therefore get provenance.
        candidate.setdefault("provenance", {"created_by": "llm", "decision": decision["decision"]})
        if candidate["id"] not in proposed_state["entities"]["candidate_experiments"]:
            proposed_state = record_entity(proposed_state, "candidate_experiments", candidate["id"], candidate, actor="llm")
        if decision["decision"] == "propose_trial":
            patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
            next_legacy = legacy.start_trial(campaign_config, loop_config, trial_patch=patch, dry_run=False)
            proposed_state = update_controller_state(
                proposed_state,
                {"status": "running", "active_trial_id": next_legacy.get("current_trial_id"), "last_decision": decision["decision"], "last_decision_at": _now(), "decision_rejections": 0, "last_rejection_reason": None, "safe_stop_reason": None},
            )
            return proposed_state, next_legacy, f"launched candidate {candidate['id']}"
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
