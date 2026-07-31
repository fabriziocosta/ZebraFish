"""Deterministic validation policy for autonomous candidate launches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain_guidance import DomainGuidanceError, load_domain_contract


@dataclass(frozen=True)
class CandidateValidation:
    valid: bool
    reasons: tuple[str, ...]
    leaf_paths: tuple[str, ...] = ()


def patch_leaf_paths(patch: dict[str, Any], *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            paths.extend(patch_leaf_paths(value, prefix=path))
        else:
            paths.append(path)
    return paths


def _lookup(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _patch_for_stage(candidate: dict[str, Any], stage: str) -> dict[str, Any]:
    patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
    if not isinstance(patch, dict):
        return {}
    if stage in patch and isinstance(patch[stage], dict):
        return patch[stage]
    if any(key in patch for key in (candidate.get("allowed_stages") or [])):
        return {}
    if any(key in patch for key in ("10C", "13C", "12T", "15T")):
        return {}
    # A single-stage candidate may provide the patch directly.
    return patch


def _strict_protocol_enabled(campaign_config: dict[str, Any]) -> bool:
    protocol = str(campaign_config.get("campaign", {}).get("evaluation_protocol", ""))
    return bool(campaign_config.get("campaign", {}).get("require_protocol_compliance", False)) or protocol.startswith("three_seed")


def _flatten_values(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, nested in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten_values(nested, prefix=path))
    return result


def _structured_rule(value: Any, *, field: str) -> bool:
    if isinstance(value, dict):
        return (
            isinstance(value.get("metric"), str)
            and isinstance(value.get("comparison"), str)
            and isinstance(value.get("direction"), str)
            and isinstance(value.get("minimum_effect"), (int, float))
            and not isinstance(value.get("minimum_effect"), bool)
        )
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, dict) and _structured_rule(item, field=field) for item in value)
    return False


def _validate_domain_expectations(
    candidate: dict[str, Any],
    campaign_config: dict[str, Any],
    state: dict[str, Any],
) -> list[str]:
    guidance = campaign_config.get("domain_guidance", {})
    if not isinstance(guidance, dict) or not guidance.get("enabled"):
        return []
    expectations = candidate.get("domain_expectations")
    if not isinstance(expectations, list) or not expectations:
        return ["candidate requires domain_expectations for the active domain contract"]
    contract_path = guidance.get("contract_path")
    try:
        contract = load_domain_contract(contract_path)
    except (DomainGuidanceError, OSError, TypeError) as exc:
        return [f"active domain contract is unavailable: {exc}"]
    constraints = {str(item["id"]): item for item in contract["constraints"]}
    candidate_family_id = str(guidance.get("candidate_family_id") or contract["id"])
    reasons: list[str] = []
    seen: set[str] = set()
    for expectation in expectations:
        if not isinstance(expectation, dict):
            reasons.append("domain expectations must be mappings")
            continue
        constraint_id = str(expectation.get("constraint_id", ""))
        if constraint_id not in constraints:
            reasons.append(f"unknown domain constraint: {constraint_id or '<missing>'}")
            continue
        if constraint_id in seen:
            reasons.append(f"duplicate domain expectation: {constraint_id}")
        seen.add(constraint_id)
        if expectation.get("comparison") != "paired_baseline":
            reasons.append(f"domain expectation {constraint_id} must use paired_baseline")
        if expectation.get("direction") not in {"increase", "decrease", "preserve_or_improve"}:
            reasons.append(f"domain expectation {constraint_id} has invalid direction")
        if expectation.get("role") != constraints[constraint_id].get("role"):
            reasons.append(f"domain expectation {constraint_id} role does not match the protected contract")
    missing = sorted(set(constraints) - seen)
    if missing:
        reasons.append(f"candidate omits active domain constraints: {', '.join(missing)}")
    compatible_calibrations = [
        calibration
        for calibration in state.get("entities", {}).get("domain_calibrations", {}).values()
        if isinstance(calibration, dict)
        and calibration.get("status") == "frozen"
        and calibration.get("contract_hash") == contract.get("_hash")
        and calibration.get("candidate_family_id") == candidate_family_id
    ]
    candidate_kind = str(candidate.get("candidate_kind", "intervention"))
    if candidate_kind == "intervention" and not compatible_calibrations:
        reasons.append("a frozen domain baseline calibration is required before an intervention candidate")
    if candidate_kind == "baseline_calibration" and compatible_calibrations:
        reasons.append("a frozen calibration already exists for the active domain contract")
    return reasons


def validate_candidate(
    candidate: dict[str, Any],
    *,
    loop_config: dict[str, Any],
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    active_process: bool = False,
) -> CandidateValidation:
    reasons: list[str] = []
    strict_protocol = _strict_protocol_enabled(campaign_config)
    if not isinstance(candidate, dict):
        return CandidateValidation(False, ("candidate must be a mapping",))
    candidate_kind = str(candidate.get("candidate_kind", "intervention"))
    if candidate_kind not in {"intervention", "baseline_calibration"}:
        reasons.append(f"unsupported candidate_kind: {candidate_kind}")
    stages = list(campaign_config.get("campaign", {}).get("stages", []))
    if len(stages) < 1:
        reasons.append("campaign has no stages")
    if candidate.get("status") not in {None, "proposed", "validated", "selected_for_execution"}:
        reasons.append(f"candidate status is not launchable: {candidate.get('status')!r}")
    if not candidate.get("id") and not candidate.get("candidate_id"):
        reasons.append("candidate requires id")
    if not candidate.get("purpose") and not candidate.get("title"):
        reasons.append("candidate requires purpose")
    if not candidate.get("question_id") and not candidate.get("addresses", {}).get("question_id"):
        reasons.append("candidate must address an existing question")
    question_id = candidate.get("question_id") or candidate.get("addresses", {}).get("question_id")
    if question_id and question_id not in state.get("entities", {}).get("questions", {}):
        # During initialization a campaign may have no questions yet.  The
        # LLM is allowed to create one in the same state transaction, so a
        # valid explicit question is still required before launch.
        reasons.append(f"unknown question: {question_id}")
    hypothesis_ids = candidate.get("hypothesis_ids") or candidate.get("addresses", {}).get("hypothesis_ids", [])
    if not hypothesis_ids:
        reasons.append("candidate must address at least one hypothesis")
    for hypothesis_id in hypothesis_ids:
        if hypothesis_id not in state.get("entities", {}).get("hypotheses", {}):
            reasons.append(f"unknown hypothesis: {hypothesis_id}")
    configured_seeds = campaign_config.get("campaign", {}).get("replicate_seeds", [0, 1, 2])
    try:
        configured_seeds = [int(value) for value in configured_seeds]
    except (TypeError, ValueError):
        configured_seeds = [0, 1, 2]
    candidate_seeds = candidate.get("replicate_seeds", configured_seeds)
    if not isinstance(candidate_seeds, list) or len(candidate_seeds) != len(set(candidate_seeds)):
        reasons.append("replicate_seeds must be a list of unique seed values")
    elif len(candidate_seeds) < int(campaign_config.get("campaign", {}).get("minimum_replicates", 3)):
        reasons.append("candidate does not contain the required number of fixed-seed replicates")
    elif [int(value) for value in candidate_seeds] != configured_seeds:
        reasons.append("candidate replicate seeds must match the campaign protocol")
    if active_process:
        reasons.append("another process is active")
    if len(stages) != len(set(stages)):
        reasons.append("campaign stages must be unique")
    allowed_stages = candidate.get("allowed_stages")
    if allowed_stages is not None:
        if not isinstance(allowed_stages, list) or not allowed_stages:
            reasons.append("allowed_stages must be a non-empty list")
        else:
            unknown_allowed = [stage for stage in allowed_stages if stage not in stages]
            if unknown_allowed:
                reasons.append(f"candidate allowed_stages are not campaign stages: {', '.join(map(str, unknown_allowed))}")
            if len(allowed_stages) != len(set(allowed_stages)):
                reasons.append("candidate allowed_stages must be unique")
    if campaign_config.get("campaign", {}).get("enforce_candidate_lineage", False):
        base_experiment = candidate.get("base_experiment")
        if not base_experiment:
            reasons.append("candidate requires base_experiment")
        elif not (
            base_experiment in state.get("entities", {}).get("experiments", {})
            or base_experiment in state.get("entities", {}).get("trials", {})
        ):
            reasons.append(f"unknown base_experiment: {base_experiment}")
    if strict_protocol:
        if not candidate.get("base_experiment"):
            reasons.append("protocol candidate requires exact base_experiment")
        if not candidate.get("base_stage"):
            reasons.append("protocol candidate requires exact base_stage")
        elif candidate.get("base_stage") not in stages:
            reasons.append("protocol candidate base_stage is not a campaign stage")
        if not isinstance(candidate.get("fixed_variables"), dict):
            reasons.append("protocol candidate requires fixed_variables mapping")
        if not isinstance(candidate.get("varied_variables"), dict):
            reasons.append("protocol candidate requires varied_variables mapping")
        elif candidate_kind == "intervention" and not candidate.get("varied_variables"):
            reasons.append("intervention candidate requires at least one varied variable")
        if not _structured_rule(candidate.get("expected_outcomes"), field="expected_outcomes"):
            reasons.append("expected_outcomes must contain structured metric rules")
        if not _structured_rule(candidate.get("falsification_criteria"), field="falsification_criteria"):
            reasons.append("falsification_criteria must contain structured metric rules")
        if not candidate.get("resolved_base_configuration_hash"):
            reasons.append("protocol candidate requires resolved_base_configuration_hash")
        if not candidate.get("source_checkpoint") and not candidate.get("source_checkpoint_id"):
            reasons.append("protocol candidate requires an exact source checkpoint")
        if not isinstance(candidate.get("baseline"), str) or not candidate.get("baseline"):
            reasons.append("protocol candidate requires an exact paired baseline reference")
        reasons.extend(_validate_domain_expectations(candidate, campaign_config, state))

    total_leaves: list[str] = []
    patch = candidate.get("configuration_patch", candidate.get("trial_patch", {}))
    if not isinstance(patch, dict):
        reasons.append("configuration_patch must be a mapping")
        patch = {}
    unknown_stage_keys = [key for key in patch if key not in stages]
    if unknown_stage_keys:
        reasons.append(f"patch references unknown stage(s): {', '.join(map(str, unknown_stage_keys))}")
    for stage in stages:
        stage_patch = _patch_for_stage(candidate, stage)
        if not stage_patch:
            continue
        stage_cfg = loop_config.get("experiments", {}).get(stage, {})
        allowed = [str(path) for path in stage_cfg.get("allowed_patch_paths", [])]
        for leaf in patch_leaf_paths(stage_patch):
            total_leaves.append(f"{stage}.{leaf}")
            if not any(leaf == path or leaf.startswith(f"{path}.") for path in allowed):
                reasons.append(f"non-allowlisted parameter: {stage}.{leaf}")
            if leaf.startswith("model_config.") and leaf not in {"model_config.dropout", "model_config.normalization", "model_config.attention_dropout"}:
                reasons.append(f"architecture/model parameter is not autonomous-safe: {stage}.{leaf}")
            ranges = campaign_config.get("parameter_ranges", {}).get(stage)
            if not isinstance(ranges, dict):
                ranges = stage_cfg.get("parameter_ranges", {})
            range_spec = _lookup(ranges, leaf)
            value = _lookup(stage_patch, leaf)
            if strict_protocol and range_spec is None:
                reasons.append(f"parameter has no explicit range/allowlist leaf: {stage}.{leaf}")
            if isinstance(range_spec, dict) and value is not None:
                minimum = range_spec.get("min")
                maximum = range_spec.get("max")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    reasons.append(f"parameter must be numeric for range validation: {stage}.{leaf}")
                else:
                    if minimum is not None and value < minimum:
                        reasons.append(f"parameter below minimum: {stage}.{leaf}")
                    if maximum is not None and value > maximum:
                        reasons.append(f"parameter above maximum: {stage}.{leaf}")
    max_leaves = int(campaign_config.get("campaign", {}).get("max_patch_leaf_count", 2))
    if len(total_leaves) == 0 and not candidate.get("baseline"):
        reasons.append("candidate has no configuration patch")
    if len(total_leaves) > max_leaves:
        reasons.append(f"patch changes {len(total_leaves)} leaves; maximum is {max_leaves}")

    cost = candidate.get("cost") if isinstance(candidate.get("cost"), dict) else {}
    gpu_budget_applicable = campaign_config.get("campaign", {}).get("gpu_budget_applicable", True) is not False
    estimated_cost = cost.get("estimated_gpu_hours", candidate.get("estimated_gpu_hours"))
    if estimated_cost is None:
        reasons.append("candidate requires estimated_gpu_hours")
    else:
        try:
            estimated_cost = float(estimated_cost)
        except (TypeError, ValueError):
            reasons.append("estimated_gpu_hours must be numeric")
        else:
            maximum = campaign_config.get("campaign", {}).get("max_single_trial_gpu_hours") if gpu_budget_applicable else None
            if maximum is not None and estimated_cost > float(maximum):
                reasons.append("candidate exceeds maximum single-trial GPU budget")
            remaining = campaign_config.get("campaign", {}).get("remaining_gpu_hours") if gpu_budget_applicable else None
            replicate_count = len(candidate_seeds) if isinstance(candidate_seeds, list) else 1
            total_estimated = estimated_cost * replicate_count
            reservations = state.get("controller_state", {}).get("launch_reservations", {})
            reserved = sum(
                float(item.get("estimated_gpu_hours", 0.0))
                for item in reservations.values()
                if isinstance(item, dict) and item.get("status") in {"reserved", "launched"}
            )
            consumed = 0.0
            for experiment in state.get("entities", {}).get("experiments", {}).values():
                if not isinstance(experiment, dict):
                    continue
                resources = experiment.get("execution", {}).get("resources", {}) if isinstance(experiment.get("execution"), dict) else {}
                value = resources.get("actual_gpu_hours")
                if isinstance(value, (int, float)) and float(value) >= 0:
                    consumed += float(value)
            total_budget = remaining
            if total_budget is None and gpu_budget_applicable:
                total_budget = campaign_config.get("campaign", {}).get("compute_budget_gpu_hours")
            if total_budget is not None and consumed + reserved + total_estimated > float(total_budget):
                reasons.append("candidate exceeds remaining GPU budget")

    required = ("expected_outcomes", "falsification_criteria", "risks")
    for field in required:
        if not candidate.get(field):
            reasons.append(f"candidate requires {field}")
    if strict_protocol and isinstance(candidate.get("fixed_variables"), dict):
        base_id = candidate.get("base_experiment")
        base = state.get("entities", {}).get("experiments", {}).get(base_id, {})
        resolved = base.get("configuration", {}).get("resolved_configuration", {}) if isinstance(base, dict) else {}
        if not isinstance(resolved, dict) or not resolved:
            reasons.append("base experiment lacks a resolved configuration for protocol comparison")
        else:
            from src.experiment_protocol import config_hash
            if candidate.get("resolved_base_configuration_hash") != config_hash(resolved):
                reasons.append("resolved_base_configuration_hash does not match the immutable base configuration")
            fixed = _flatten_values(candidate["fixed_variables"])
            resolved_flat = _flatten_values(resolved)
            for path, expected in fixed.items():
                if path not in resolved_flat:
                    reasons.append(f"fixed variable is absent from base configuration: {path}")
                elif resolved_flat[path] != expected:
                    reasons.append(f"fixed variable does not match base configuration: {path}")
        varied = _flatten_values(candidate.get("varied_variables", {})) if isinstance(candidate.get("varied_variables"), dict) else {}
        patch_leaves = {path.split(".", 1)[1] for path in total_leaves if "." in path}
        if varied and set(varied) != patch_leaves:
            reasons.append("varied_variables must exactly describe every changed configuration leaf")
        if candidate_kind == "baseline_calibration" and (varied or patch_leaves):
            reasons.append("baseline calibration candidate cannot change configuration leaves")
    return CandidateValidation(not reasons, tuple(dict.fromkeys(reasons)), tuple(total_leaves))
