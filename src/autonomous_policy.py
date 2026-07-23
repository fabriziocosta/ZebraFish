"""Deterministic validation policy for autonomous candidate launches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def validate_candidate(
    candidate: dict[str, Any],
    *,
    loop_config: dict[str, Any],
    campaign_config: dict[str, Any],
    state: dict[str, Any],
    active_process: bool = False,
) -> CandidateValidation:
    reasons: list[str] = []
    if not isinstance(candidate, dict):
        return CandidateValidation(False, ("candidate must be a mapping",))
    stages = list(campaign_config.get("campaign", {}).get("stages", []))
    if len(stages) < 1:
        reasons.append("campaign has no stages")
    if candidate.get("status") not in {None, "proposed", "validated"}:
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
    if active_process:
        reasons.append("another process is active")
    if len(stages) != len(set(stages)):
        reasons.append("campaign stages must be unique")

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

    estimated_cost = candidate.get("cost", {}).get("estimated_gpu_hours", candidate.get("estimated_gpu_hours"))
    if estimated_cost is None:
        reasons.append("candidate requires estimated_gpu_hours")
    else:
        try:
            estimated_cost = float(estimated_cost)
        except (TypeError, ValueError):
            reasons.append("estimated_gpu_hours must be numeric")
        else:
            maximum = campaign_config.get("campaign", {}).get("max_single_trial_gpu_hours")
            if maximum is not None and estimated_cost > float(maximum):
                reasons.append("candidate exceeds maximum single-trial GPU budget")
            remaining = campaign_config.get("campaign", {}).get("remaining_gpu_hours")
            if remaining is not None and estimated_cost > float(remaining):
                reasons.append("candidate exceeds remaining GPU budget")

    required = ("expected_outcomes", "falsification_criteria", "risks")
    for field in required:
        if not candidate.get(field):
            reasons.append(f"candidate requires {field}")
    return CandidateValidation(not reasons, tuple(dict.fromkeys(reasons)), tuple(total_leaves))
