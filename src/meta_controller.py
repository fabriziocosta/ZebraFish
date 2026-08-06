"""Supervisory controller for daily, bounded self-healing of the experiment loop.

The meta-controller is deliberately separate from the scientific campaign
controller.  It may diagnose and repair operational code only through a fixed
allowlist and a verification suite.  Scientific objective changes are always
recorded as proposals.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable
import uuid

from src.scientific_state import (
    ScientificStateError,
    load_state,
    record_entity,
    save_state,
    transactional_update,
    update_controller_state,
)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


META_DECISIONS = {"no_action", "patch", "campaign_control", "proposal", "safe_stop"}
SEVERITIES = {"info", "warning", "critical"}
CAMPAIGN_ACTIONS = {"stop", "continue", "reconcile"}
RUN_NOW_SIGNAL = getattr(signal, "SIGUSR1", None)


class MetaControllerError(ValueError):
    """Base error for invalid supervisory decisions or configuration."""


class MetaDecisionError(MetaControllerError):
    """Raised when the LLM response is not a valid meta decision."""


class PatchSafetyError(MetaControllerError):
    """Raised when a proposed patch is outside the constitutional allowlist."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    return payload if isinstance(payload, dict) else {}


def load_meta_config(root: str | Path, campaign_config: dict[str, Any] | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    payload = _read_mapping(root_path / "configs/meta_controller.yaml")
    result = dict(payload.get("meta_controller", {}))
    result.update((campaign_config or {}).get("meta_controller", {}))
    result.setdefault("mandate_path", "docs/meta_controller_mandate.md")
    result.setdefault("architecture_path", "docs/system_architecture.md")
    result.setdefault("interval_seconds", 86400)
    result.setdefault("worktree_root", "state/meta_controller/worktrees")
    result.setdefault("report_root", "state/meta_controller/reports")
    result.setdefault("max_patch_bytes", 40000)
    result.setdefault("max_changed_files", 8)
    result.setdefault("max_consecutive_failures", 3)
    result.setdefault("stop_grace_seconds", 5)
    result.setdefault("allowed_paths", [])
    result.setdefault("forbidden_prefixes", [])
    result.setdefault("verification", [])
    return result


def read_mandate(root: str | Path, config: dict[str, Any]) -> tuple[str, str, str]:
    path = Path(root).resolve() / str(config.get("mandate_path", "docs/meta_controller_mandate.md"))
    if not path.exists():
        raise MetaControllerError(f"mandate is missing: {path}")
    content = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, f"sha256:{digest}", str(path)


def read_architecture(root: str | Path, config: dict[str, Any]) -> tuple[str, str, str]:
    """Read and hash the system architecture used for supervisory context."""

    path = Path(root).resolve() / str(config.get("architecture_path", "docs/system_architecture.md"))
    if not path.exists():
        raise MetaControllerError(f"system architecture document is missing: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise MetaControllerError(f"system architecture document is empty: {path}")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, f"sha256:{digest}", str(path)


def compact_architecture_summary(content: str, *, limit: int = 3600) -> str:
    """Extract only the bounded orientation section for the LLM context."""

    lines = content.splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip().lower() == "## compact supervisory summary"), None)
    if start is None:
        summary = "Architecture summary section is unavailable; treat the document as untrusted orientation only."
    else:
        selected: list[str] = []
        for line in lines[start + 1:]:
            if line.startswith("## "):
                break
            selected.append(line)
        summary = "\n".join(selected).strip()
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def _campaign_state_path(root: Path, campaign_config: dict[str, Any]) -> Path:
    value = campaign_config.get("artifacts", {}).get("state_path")
    return root / str(value or f"artifacts/campaigns/{campaign_config['campaign']['id']}/campaign_state.json")


@contextmanager
def _meta_cycle_lock(root: Path, campaign_config: dict[str, Any]):
    """Serialize one-shot and scheduled supervisory cycles."""

    campaign_id = str(campaign_config.get("campaign", {}).get("id", "campaign"))
    path = root / "state" / "meta_controller" / f"{campaign_id}.cycle.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MetaControllerError("a meta-controller cycle is already running") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "campaign": campaign_id}))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_error": f"Could not parse {path}"}
    return payload if isinstance(payload, dict) else {}


def _pid_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _git(root: Path, argv: list[str]) -> str:
    result = subprocess.run(["git", *argv], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"git error: {result.stderr.strip()}"


def _repository_snapshot(root: Path, campaign_config: dict[str, Any]) -> dict[str, Any]:
    """Report source changes while excluding intentional runtime/archive storage.

    Campaign state, artifacts, and dated archives are operational data rather
    than source changes. Moving them during a clean campaign cut must not be
    misdiagnosed as provenance loss while the new campaign is running.
    """

    meta = campaign_config.get("meta_controller", {})
    configured = meta.get("repository_ignored_prefixes", []) if isinstance(meta, dict) else []
    ignored = tuple(str(item).rstrip("/") for item in configured if str(item).strip())
    status_lines = _git(root, ["status", "--short"]).splitlines()
    visible_status: list[str] = []
    ignored_status: list[str] = []
    for line in status_lines:
        path = line[2:].strip() if len(line) >= 2 else line.strip()
        paths = [part.strip() for part in path.split(" -> ")]
        if any(any(candidate == prefix or candidate.startswith(prefix + "/") for prefix in ignored) for candidate in paths):
            ignored_status.append(line)
        else:
            visible_status.append(line)
    diff_args = ["diff", "--stat", "--", "."]
    diff_args.extend(f":(exclude){prefix}/**" for prefix in ignored if prefix in {"state", "artifacts", "archive"})
    diff_args.extend(f":(exclude){prefix}" for prefix in ignored if prefix not in {"state", "artifacts", "archive"})
    return {
        "revision": _git(root, ["rev-parse", "HEAD"]),
        "status": "\n".join(visible_status),
        "diff_stat": _git(root, diff_args),
        "ignored_operational_changes": len(ignored_status),
        "ignored_prefixes": list(ignored),
    }


def collect_snapshot(
    root: str | Path,
    campaign_config: dict[str, Any],
    *,
    architecture_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect bounded supervisory context without writing or launching anything."""

    root_path = Path(root).resolve()
    state_path = root_path / str(campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    state = load_state(state_path) if state_path.exists() else {"entities": {}, "controller_state": {}}
    campaign_state_path = _campaign_state_path(root_path, campaign_config)
    campaign_state = _read_json(campaign_state_path)
    controller = state.get("controller_state", {})
    meta_state = controller.get("meta_controller", {}) if isinstance(controller.get("meta_controller"), dict) else {}
    pid = campaign_state.get("active_launch_state", {}).get("pid")
    observations = list(state.get("entities", {}).get("observations", {}).values())[-20:]
    domain_evaluations = list(state.get("entities", {}).get("domain_evaluations", {}).values())[-5:]
    reports = list(state.get("entities", {}).get("meta_controller_runs", {}).values())[-5:]
    return {
        "campaign": {
            "id": campaign_config.get("campaign", {}).get("id"),
            "status": campaign_state.get("status"),
            "stage": campaign_state.get("current_stage"),
            "trial_id": campaign_state.get("current_trial_id"),
            "state_path": str(campaign_state_path),
        },
        "process": {"pid": pid, "running": _pid_running(pid)},
        "scientific_controller": {
            "status": controller.get("status"),
            "safe_stop_reason": controller.get("safe_stop_reason"),
            "last_poll_at": controller.get("last_poll_at"),
        },
        "meta_controller": meta_state,
        "observations": observations,
        "known_evidence_ids": [str(item.get("id")) for item in observations if isinstance(item, dict) and item.get("id")],
        "domain_guidance": [
            {
                "id": item.get("id"),
                "contract_id": item.get("contract_id"),
                "objective_eligibility": item.get("objective_eligibility"),
                "constraints": [
                    {
                        "id": constraint.get("id"),
                        "role": constraint.get("role"),
                        "status": constraint.get("status"),
                    }
                    for constraint in item.get("constraints", [])
                ],
                "umap_used_for_decision": False,
            }
            for item in domain_evaluations if isinstance(item, dict)
        ],
        "architecture": architecture_context or {
            "status": "not_loaded",
            "summary": "Architecture context was not loaded for this snapshot.",
        },
        "recent_meta_reports": reports,
        "repository": _repository_snapshot(root_path, campaign_config),
    }


def _schema() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "meta_controller_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mandate_version", "architecture_version", "decision", "diagnosis", "actions", "proposal_only_changes", "rollback_plan", "unresolved_risks"],
                "properties": {
                    "mandate_version": {"type": "string"},
                    "architecture_version": {"type": "string"},
                    "decision": {"type": "string", "enum": sorted(META_DECISIONS)},
                    "diagnosis": {
                        "type": "object", "additionalProperties": False,
                        "required": ["summary", "severity", "evidence_references", "root_causes"],
                        "properties": {
                            "summary": {"type": "string"},
                            "severity": {"type": "string", "enum": sorted(SEVERITIES)},
                            "evidence_references": {"type": "array", "items": {"type": "string"}},
                            "root_causes": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "actions": {
                        "type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["kind", "summary", "patch", "verification_names", "campaign_action"],
                            "properties": {
                                "kind": {"type": "string", "enum": ["patch", "campaign_control", "no_action"]},
                                "summary": {"type": "string"},
                                "patch": {"type": "string"},
                                "verification_names": {"type": "array", "items": {"type": "string"}},
                                "campaign_action": {"type": "string", "enum": ["none", "stop", "continue", "reconcile"]},
                            },
                        },
                    },
                    "proposal_only_changes": {"type": "array", "items": {"type": "string"}},
                    "rollback_plan": {"type": "string"},
                    "unresolved_risks": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    }


def parse_decision(payload: str | dict[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise MetaDecisionError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise MetaDecisionError("meta decision must be a JSON object")
    required = {"mandate_version", "architecture_version", "decision", "diagnosis", "actions", "proposal_only_changes", "rollback_plan", "unresolved_risks"}
    missing = sorted(required - result.keys())
    if missing:
        raise MetaDecisionError(f"meta decision missing fields: {', '.join(missing)}")
    if result["decision"] not in META_DECISIONS:
        raise MetaDecisionError(f"unsupported meta decision: {result['decision']!r}")
    diagnosis = result["diagnosis"]
    if not isinstance(diagnosis, dict) or diagnosis.get("severity") not in SEVERITIES or not isinstance(diagnosis.get("evidence_references"), list):
        raise MetaDecisionError("diagnosis must include a valid severity and evidence_references")
    if not isinstance(result["actions"], list) or not isinstance(result["proposal_only_changes"], list):
        raise MetaDecisionError("actions and proposal_only_changes must be arrays")
    for action in result["actions"]:
        if not isinstance(action, dict) or action.get("kind") not in {"patch", "campaign_control", "no_action"}:
            raise MetaDecisionError("invalid remediation action")
        if not isinstance(action.get("verification_names"), list) or action.get("campaign_action") not in {"none", *CAMPAIGN_ACTIONS}:
            raise MetaDecisionError("invalid remediation action fields")
        if not isinstance(action.get("patch"), str) or not isinstance(action.get("summary"), str):
            raise MetaDecisionError("action patch and summary must be strings")
    return result


def _allowed(path: str, config: dict[str, Any]) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    if any(normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/") for prefix in config.get("forbidden_prefixes", [])):
        return False
    return any(fnmatch.fnmatch(normalized, str(pattern)) for pattern in config.get("allowed_paths", []))


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:].strip())
        elif line.startswith("--- a/"):
            paths.append(line[6:].strip())
    return sorted(set(paths))


def validate_patch(patch: str, config: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(patch, str) or not patch.strip():
        raise PatchSafetyError("patch is empty")
    if len(patch.encode("utf-8")) > int(config.get("max_patch_bytes", 40000)):
        raise PatchSafetyError("patch exceeds maximum byte size")
    paths = patch_paths(patch)
    if not paths:
        raise PatchSafetyError("patch contains no unified-diff file paths")
    if len(paths) > int(config.get("max_changed_files", 8)):
        raise PatchSafetyError("patch changes too many files")
    unsafe = [path for path in paths if not _allowed(path, config)]
    if unsafe:
        raise PatchSafetyError(f"patch contains non-allowlisted paths: {', '.join(unsafe)}")
    if any(token in patch for token in (".env", "secrets", "artifacts/", "state/")):
        raise PatchSafetyError("patch contains a forbidden sensitive path")
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    lowered = added_lines.lower()
    unsafe_code_patterns = (
        "os.system(",
        "subprocess.popen(",
        "subprocess.call(",
        "subprocess.run(",
        "shell=true",
        "eval(",
        "exec(",
        "__import__(",
        "openai_api_key",
        "getenv(\"openai",
    )
    if any(pattern in lowered for pattern in unsafe_code_patterns):
        raise PatchSafetyError("patch contains an unsafe subprocess, dynamic-code, or secret-access pattern")
    # A repair may improve how the objective is observed, but it may not
    # silently redefine what the campaign means.  Those changes are proposals.
    protected_scientific_tokens = (
        "primary_metric",
        "optimization_direction",
        "guardrail",
        "minimums:",
        "dataset_semantics",
        "architecture",
        "model_family",
        "domain_guidance",
        "domain_expectations",
        "contract_path",
        "threshold_strategy",
        "umap_decision_role",
    )
    config_paths = [path for path in paths if path.startswith("configs/experiment_campaigns/")]
    if config_paths and any(token in lowered for token in protected_scientific_tokens):
        raise PatchSafetyError("patch attempts to alter a protected scientific campaign field")
    # Removing tests, skipping tests, or weakening assertions is never a
    # permissible self-healing operation.
    if any(path.startswith("tests/") for path in paths):
        removed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
        weakened = ("skip", "xfail", "assert true", "pass #", "return true")
        if any(any(token in line.lower() for token in weakened) for line in removed):
            raise PatchSafetyError("patch weakens or removes verification coverage")
    if any(path in {"configs/meta_controller.yaml", "docs/meta_controller_mandate.md", "docs/system_architecture.md"} for path in paths):
        raise PatchSafetyError("constitutional mandate, system architecture, and meta-controller policy are not autonomously mutable")
    if any(
        path.startswith("tests/")
        and any(token in lowered for token in ("verification", "default_verifications", "pytest.skip", "pytestmark"))
        for path in paths
    ):
        raise PatchSafetyError("meta-controller cannot alter its verification policy")
    if "src/meta_controller.py" in paths and any(
        token in lowered for token in ("subprocess", "eval(", "exec(", "__import__", "os.environ", "secrets")
    ):
        raise PatchSafetyError("meta-controller patch contains a forbidden self-modification construct")
    return tuple(paths)


def required_verification_names(paths: Iterable[str], config: dict[str, Any]) -> tuple[str, ...]:
    """Select verification deterministically from changed paths."""

    configured = _verification_map(config)
    names: list[str] = []
    defaults = [str(name) for name in config.get("default_verifications", [])]
    for path in paths:
        if path.startswith(("src/scientific_state.py", "src/autonomous_campaign.py", "src/agent_campaign_loop.py")):
            names.extend(defaults)
            break
    names.extend(defaults)
    for path in paths:
        if path.startswith("dashboard/") and "dashboard-build" in configured:
            names.append("dashboard-build")
        if path.startswith("src/observation_engine.py") and "observation-tests" in configured:
            names.append("observation-tests")
        if path.startswith("src/domain_guidance.py") and "domain-guidance-tests" in configured:
            names.append("domain-guidance-tests")
        if path.startswith("tests/") and "full-python-tests" in configured:
            names.append("full-python-tests")
    return tuple(dict.fromkeys(name for name in names if name in configured))


def _reverse_unified_patch(patch: str) -> str:
    """Generate a text reverse diff suitable for ``git apply``."""

    lines = patch.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        if line.startswith("--- a/"):
            result.append("+++ b/" + line[len("--- a/"):])
        elif line.startswith("+++ b/"):
            result.append("--- a/" + line[len("+++ b/"):])
        elif line.startswith("+") and not line.startswith("+++"):
            result.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            result.append("+" + line[1:])
        else:
            result.append(line)
    return "".join(result)


def _verification_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): item for item in config.get("verification", []) if isinstance(item, dict) and item.get("name")}


def validate_verifications(names: Iterable[str], config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    available = _verification_map(config)
    selected = []
    for name in names:
        if name not in available:
            raise PatchSafetyError(f"verification command is not allowlisted: {name}")
        item = dict(available[name])
        argv = item.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(token, str) for token in argv):
            raise PatchSafetyError(f"invalid verification command: {name}")
        selected.append(item)
    return tuple(selected)


def run_verifications(root: Path, commands: Iterable[dict[str, Any]], runner: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    results = []
    for command in commands:
        cwd = root / str(command.get("cwd", "."))
        argv = list(command["argv"])
        if argv and argv[0] == ".venv/bin/python":
            argv[0] = sys.executable
        if runner is None:
            completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
            result = {"name": command["name"], "argv": argv, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
        else:
            completed = runner(argv, cwd=cwd)
            result = {"name": command["name"], "argv": argv, "returncode": int(getattr(completed, "returncode", completed if isinstance(completed, int) else 0))}
        results.append(result)
        if result["returncode"] != 0:
            break
    return results


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _persist_report(root: Path, campaign_config: dict[str, Any], report: dict[str, Any]) -> None:
    state_path = root / str(campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    state = load_state(state_path) if state_path.exists() else None
    if state is None:
        from src.scientific_state import empty_state
        state = empty_state()
    previous_meta = state.get("controller_state", {}).get("meta_controller", {})
    previous_failures = int(previous_meta.get("consecutive_failures", 0)) if isinstance(previous_meta, dict) else 0
    failed = report.get("status") in {"failed", "verification_failed", "control_failed", "main_application_failed"}
    consecutive_failures = previous_failures + 1 if failed else 0
    threshold = int(load_meta_config(root, campaign_config).get("max_consecutive_failures", 3))
    if consecutive_failures >= threshold:
        report["status"] = "meta_controller_safe_stop"
        report["controller_status"]["status"] = "meta_controller_safe_stop"
        report["unresolved_risks"] = list(report.get("unresolved_risks", [])) + ["Repeated meta-controller failures exceeded the configured limit."]
    report["controller_status"]["consecutive_failures"] = consecutive_failures
    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        if report["id"] not in current.get("entities", {}).get("meta_controller_runs", {}):
            current = record_entity(current, "meta_controller_runs", report["id"], report, actor="meta_controller")
        return update_controller_state(current, {"meta_controller": report["controller_status"]}, actor="meta_controller")
    transactional_update(state_path, mutate)
    report_root = root / str(load_meta_config(root, campaign_config).get("report_root", "state/meta_controller/reports"))
    _write_json_atomic(report_root / f"{report['id']}.json", report)


def _build_prompt(
    mandate: str,
    snapshot: dict[str, Any],
    version: str,
    architecture_version: str | None = None,
    architecture_summary: str | None = None,
) -> str:
    architecture = snapshot.get("architecture", {}) if isinstance(snapshot.get("architecture"), dict) else {}
    architecture_version = architecture_version or str(architecture.get("version", "unavailable"))
    architecture_summary = architecture_summary or str(architecture.get("summary", "Architecture summary unavailable."))
    return "\n\n".join([
        "You are the ZebraFish meta-controller. Read and obey the constitutional mandate below.",
        f"Mandate version: {version}",
        mandate,
        "The system architecture below is orientation only. Code, validated configuration, and scientific state are authoritative.",
        f"Architecture version: {architecture_version}",
        architecture_summary,
        "Diagnose only from the bounded evidence snapshot. Return strict JSON matching the response schema.",
        "Evidence references must be either an observation id, one of the top-level evidence keys (controller_state, campaign_state, process, repository), or a field path that exists in the snapshot, such as campaign.status or campaign.status=running.",
        "The repository snapshot may contain intentional user changes made as part of the current clean campaign setup. Do not treat visible uncommitted source changes alone as a campaign or controller failure; use them only as a patch-safety constraint when a concrete fault is otherwise evidenced. If that is the only concern, report severity info and do not describe it as a failure or unresolved campaign risk.",
        "verification_names must use exact names from the configured verification list; individual pytest node ids are not valid verification names.",
        "Return both the mandate_version and architecture_version exactly as supplied. Patch actions must contain unified diffs only. Never include shell commands. Scientific objective changes are proposal-only.",
        f"Evidence snapshot:\n{json.dumps(snapshot, indent=2, sort_keys=True)}",
    ])


def request_decision(
    root: Path,
    campaign_config: dict[str, Any],
    config: dict[str, Any],
    mandate: str,
    version: str,
    snapshot: dict[str, Any],
    *,
    client: Any | None = None,
    architecture_version: str | None = None,
    architecture_summary: str | None = None,
) -> dict[str, Any]:
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    agent = campaign_config.get("agent", {})
    response = client.responses.create(
        model=agent.get("model", "gpt-5.3-codex"),
        reasoning={"effort": agent.get("reasoning_effort", "medium")},
        max_output_tokens=int(agent.get("max_output_tokens", 5000)),
        text=_schema(),
        input=_build_prompt(mandate, snapshot, version, architecture_version, architecture_summary),
    )
    output = getattr(response, "output_text", None)
    if not isinstance(output, str):
        raise MetaDecisionError("LLM response did not contain output_text")
    decision = parse_decision(output)
    if decision["mandate_version"] != version:
        raise MetaDecisionError("LLM returned a mandate version different from the loaded mandate")
    expected_architecture = architecture_version or str(snapshot.get("architecture", {}).get("version", "unavailable"))
    if decision["architecture_version"] != expected_architecture:
        raise MetaDecisionError("LLM returned an architecture version different from the loaded architecture document")
    return decision


def _validate_decision_evidence(decision: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Ensure every diagnosis reference resolves to the supplied snapshot.

    The model historically returned useful field-level references such as
    ``campaign.status=running``.  Treating those as opaque ids caused valid
    diagnoses to fail even though the referenced field was present in the
    bounded snapshot.  Keep the strict evidence boundary, but resolve dotted
    paths and optionally verify an ``=value`` assertion against the snapshot.
    """

    known = {str(value) for value in snapshot.get("known_evidence_ids", [])}
    top_level_references = {"controller_state", "campaign_state", "process", "repository"}

    def resolve(path: str) -> tuple[bool, Any]:
        value: Any = snapshot
        components = path.replace("[", ".").replace("]", "").split(".")
        for component in components:
            if not component:
                return False, None
            if isinstance(value, dict):
                if component not in value:
                    return False, None
                value = value[component]
            elif isinstance(value, list) and component.isdigit():
                index = int(component)
                if index >= len(value):
                    return False, None
                value = value[index]
            else:
                return False, None
        return True, value

    for raw_reference in decision.get("diagnosis", {}).get("evidence_references", []):
        reference = str(raw_reference).strip()
        if reference in known or reference in top_level_references:
            continue

        path, separator, expected = reference.partition("=")
        path = path.strip()
        available, actual = resolve(path)
        if not available:
            raise MetaDecisionError(f"diagnosis references unavailable evidence: {raw_reference}")
        if separator:
            expected = expected.strip()
            rendered_actual = json.dumps(actual, sort_keys=True, separators=(",", ":"))
            normalized_expected = expected
            try:
                normalized_expected = json.dumps(json.loads(expected), sort_keys=True, separators=(",", ":"))
            except json.JSONDecodeError:
                pass
            if normalized_expected != rendered_actual and expected != str(actual):
                raise MetaDecisionError(f"diagnosis evidence assertion does not match snapshot: {raw_reference}")


def apply_patch_in_worktree(root: Path, config: dict[str, Any], patch: str, run_id: str, *, verification_names: Iterable[str] | None = None, runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    paths = validate_patch(patch, config)
    deterministic_names = list(required_verification_names(paths, config))
    available_names = set(_verification_map(config))
    # The deterministic path-based suites are authoritative.  A model may
    # suggest a pytest node id even though the policy allowlist contains only
    # suite names; discard such extras while retaining all required suites.
    requested_names = [str(name) for name in (verification_names or []) if str(name) in available_names]
    names = list(dict.fromkeys(deterministic_names + requested_names))
    if not names:
        raise PatchSafetyError("patch action must select at least one allowlisted verification")
    commands = validate_verifications([str(name) for name in names], config)
    baseline_revision = _git(root, ["rev-parse", "HEAD"])
    baseline_status = _git(root, ["status", "--short", "--", *paths])
    if baseline_status:
        raise PatchSafetyError(f"targeted files have pre-existing user changes: {baseline_status}")
    worktree = root / str(config.get("worktree_root", "state/meta_controller/worktrees")) / run_id
    worktree.parent.mkdir(parents=True, exist_ok=True)
    try:
        created = subprocess.run(["git", "worktree", "add", "--detach", str(worktree), baseline_revision], cwd=root, capture_output=True, text=True, check=False)
        if created.returncode != 0:
            raise PatchSafetyError(f"could not create isolated worktree: {created.stderr.strip()}")
        checked = subprocess.run(["git", "apply", "--check", "-"], cwd=worktree, input=patch, capture_output=True, text=True, check=False)
        if checked.returncode != 0:
            raise PatchSafetyError(f"patch does not apply cleanly: {checked.stderr.strip()}")
        applied = subprocess.run(["git", "apply", "-"], cwd=worktree, input=patch, capture_output=True, text=True, check=False)
        if applied.returncode != 0:
            raise PatchSafetyError(f"patch application failed: {applied.stderr.strip()}")
        # Dependency trees are intentionally outside git.  Reuse the local
        # read-only installations inside the isolated worktree when present.
        for dependency in (".venv", "dashboard/node_modules"):
            source = root / dependency
            target = worktree / dependency
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=source.is_dir())
        verification = run_verifications(worktree, commands, runner=runner)
        if any(item["returncode"] != 0 for item in verification):
            return {"status": "verification_failed", "paths": paths, "worktree": str(worktree), "verification": verification}
        status = _git(root, ["status", "--short", "--", *paths])
        if status != baseline_status:
            return {
                "status": "overlapping_user_changes",
                "paths": paths,
                "worktree": str(worktree),
                "verification": verification,
                "overlap": {"before": baseline_status, "after": status},
                "baseline_revision": baseline_revision,
            }
        if _git(root, ["rev-parse", "HEAD"]) != baseline_revision:
            return {"status": "main_revision_changed", "paths": paths, "worktree": str(worktree), "verification": verification, "baseline_revision": baseline_revision}
        apply_main = subprocess.run(["git", "apply", "-"], cwd=root, input=patch, capture_output=True, text=True, check=False)
        if apply_main.returncode != 0:
            return {"status": "main_application_failed", "paths": paths, "worktree": str(worktree), "verification": verification, "error": apply_main.stderr.strip(), "baseline_revision": baseline_revision}
        post_health = _git(root, ["diff", "--check", "--", *paths])
        if post_health:
            reverse_patch = _reverse_unified_patch(patch)
            subprocess.run(["git", "apply", "-R", "-"], cwd=root, input=patch, capture_output=True, text=True, check=False)
            return {"status": "post_application_health_failed", "paths": paths, "worktree": str(worktree), "verification": verification, "error": post_health, "reverse_patch": reverse_patch, "baseline_revision": baseline_revision}
        reverse_patch = _reverse_unified_patch(patch)
        return {
            "status": "applied",
            "paths": paths,
            "worktree": str(worktree),
            "verification": verification,
            "baseline_revision": baseline_revision,
            "patch": patch,
            "reverse_patch": reverse_patch,
            "rollback_patch": reverse_patch,
            "rollback_mode": "apply",
            "post_application_health": "passed",
        }
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root, capture_output=True, text=True, check=False)


def run_once(
    root: str | Path,
    campaign_config: dict[str, Any],
    *,
    client: Any | None = None,
    invocation_source: str = "cli",
    execution_mode: str = "once",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    try:
        with _meta_cycle_lock(root_path, campaign_config):
            return _run_once(
                root_path,
                campaign_config,
                client=client,
                invocation_source=invocation_source,
                execution_mode=execution_mode,
            )
    except MetaControllerError as exc:
        if "already running" not in str(exc):
            raise
        return {
            "id": f"meta_busy_{os.getpid()}",
            "campaign": campaign_config.get("campaign", {}).get("id"),
            "status": "already_running",
            "diagnosis": {"summary": str(exc), "severity": "info", "evidence_references": [], "root_causes": []},
            "actions": [],
            "invocation_source": invocation_source,
        }


def _run_once(
    root: str | Path,
    campaign_config: dict[str, Any],
    *,
    client: Any | None = None,
    invocation_source: str = "cli",
    execution_mode: str = "once",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_meta_config(root_path, campaign_config)
    started = utc_now()
    run_id = f"meta_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    mandate_version = "unavailable"
    mandate_path: str | None = None
    architecture_version = "unavailable"
    architecture_path: str | None = None
    controller_status = {
        "status": "running",
        "run_id": run_id,
        "active_run_id": run_id,
        "pid": os.getpid(),
        "mode": execution_mode,
        "run_now_supported": execution_mode == "loop" and RUN_NOW_SIGNAL is not None,
        "started_at": started,
        "invocation_source": invocation_source,
    }
    _set_meta_status(root_path, campaign_config, controller_status)
    try:
        mandate, mandate_version, mandate_path = read_mandate(root_path, config)
        architecture, architecture_version, architecture_path = read_architecture(root_path, config)
        architecture_context = {
            "version": architecture_version,
            "path": architecture_path,
            "summary": compact_architecture_summary(architecture),
            "read_only_orientation": True,
        }
        snapshot = collect_snapshot(root_path, campaign_config, architecture_context=architecture_context)
        decision = request_decision(
            root_path,
            campaign_config,
            config,
            mandate,
            mandate_version,
            snapshot,
            client=client,
            architecture_version=architecture_version,
            architecture_summary=architecture_context["summary"],
        )
        _validate_decision_evidence(decision, snapshot)
        actions: list[dict[str, Any]] = []
        for action in decision["actions"]:
            action_result = {"kind": action["kind"], "summary": action["summary"], "campaign_action": action["campaign_action"], "status": "recorded"}
            if action["kind"] == "patch":
                action_result["result"] = apply_patch_in_worktree(root_path, config, action["patch"], run_id, verification_names=action["verification_names"])
                action_result["status"] = action_result["result"]["status"]
            elif action["kind"] == "campaign_control" and action["campaign_action"] != "none":
                action_result["result"] = _execute_campaign_control(root_path, campaign_config, action["campaign_action"], config, client=client)
                action_result["status"] = "recorded" if action_result["result"].get("returncode", 1) == 0 else "control_failed"
            actions.append(action_result)
        failed = any(item.get("status") not in {"recorded", "applied"} for item in actions)
        status = "verification_failed" if failed else "completed"
        if decision["decision"] == "safe_stop":
            status = "meta_controller_safe_stop"
        report = {
            "id": run_id, "campaign": campaign_config.get("campaign", {}).get("id"), "status": status,
            "invocation_source": invocation_source,
            "mandate_version": mandate_version, "mandate_path": mandate_path,
            "architecture_version": architecture_version, "architecture_path": architecture_path,
            "started_at": started, "completed_at": utc_now(),
            "diagnosis": decision["diagnosis"], "evidence_references": decision["diagnosis"].get("evidence_references", []),
            "actions": actions, "proposal_only_changes": decision["proposal_only_changes"],
            "rollback_plan": decision["rollback_plan"], "unresolved_risks": decision["unresolved_risks"],
            "provenance": {"created_by": "meta_controller", "mandate_version": mandate_version, "architecture_version": architecture_version, "snapshot": "compact", "invocation_source": invocation_source},
            "controller_status": {"status": status, "last_run_id": run_id, "last_run_at": utc_now(), "mandate_version": mandate_version, "architecture_version": architecture_version, "summary": decision["diagnosis"].get("summary", ""), "next_run_at": None, "pid": None if execution_mode == "once" else os.getpid(), "mode": execution_mode, "run_now_supported": execution_mode == "loop" and RUN_NOW_SIGNAL is not None, "invocation_source": invocation_source},
        }
    except Exception as exc:
        try:
            _, mandate_version, mandate_path = read_mandate(root_path, config)
        except Exception:
            mandate_version, mandate_path = "unavailable", None
        try:
            _, architecture_version, architecture_path = read_architecture(root_path, config)
        except Exception:
            architecture_version, architecture_path = "unavailable", None
        report = {
            "id": run_id, "campaign": campaign_config.get("campaign", {}).get("id"), "status": "failed",
            "invocation_source": invocation_source,
            "mandate_version": mandate_version, "mandate_path": mandate_path,
            "architecture_version": architecture_version, "architecture_path": architecture_path,
            "started_at": started, "completed_at": utc_now(),
            "diagnosis": {"summary": f"Meta-controller could not complete: {type(exc).__name__}: {exc}", "severity": "critical", "evidence_references": [], "root_causes": []},
            "evidence_references": [], "actions": [], "proposal_only_changes": [], "rollback_plan": "No patch was applied.", "unresolved_risks": [str(exc)],
            "provenance": {"created_by": "meta_controller", "error": type(exc).__name__, "mandate_version": mandate_version, "architecture_version": architecture_version, "invocation_source": invocation_source},
            "controller_status": {"status": "failed", "last_run_id": run_id, "last_run_at": utc_now(), "mandate_version": mandate_version, "architecture_version": architecture_version, "summary": str(exc), "next_run_at": None, "pid": None if execution_mode == "once" else os.getpid(), "mode": execution_mode, "run_now_supported": execution_mode == "loop" and RUN_NOW_SIGNAL is not None, "invocation_source": invocation_source},
        }
    _persist_report(root_path, campaign_config, report)
    return report


def _execute_campaign_control(root: Path, campaign_config: dict[str, Any], action: str, config: dict[str, Any], *, client: Any | None = None) -> dict[str, Any]:
    """Execute only existing deterministic campaign lifecycle operations."""

    from src import autonomous_campaign, agent_campaign_loop

    output = __import__("io").StringIO()
    if action == "stop":
        code = agent_campaign_loop.terminate_campaign(
            campaign_config,
            reason="meta-controller requested stop from deterministic evidence",
            force_after=float(config.get("stop_grace_seconds", 5)),
            stream=output,
        )
        recovery_code = None
        recovery_detail = ""
        recovery_running = False
        if code == 0:
            # A productive stop is not an endpoint. Re-enter the autonomous
            # campaign initializer immediately so it can select and launch a
            # bounded replacement trial using the same validated policy.
            recovery_code = autonomous_campaign.run_autonomous_campaign(
                campaign_config,
                once=True,
                new_trial=True,
                client=client,
                stream=output,
            )
            recovery_detail = output.getvalue()[-4000:]
            recovery_running = bool(agent_campaign_loop.campaign_live_status(campaign_config).get("running"))
            if not recovery_running and recovery_code == 0:
                recovery_code = 2
                recovery_detail += "\nautonomous recovery completed without launching a replacement stage"
    elif action == "continue":
        code = agent_campaign_loop.resume_suspended_campaign(campaign_config, stream=output)
        recovery_code = None
        recovery_detail = ""
        recovery_running = bool(agent_campaign_loop.campaign_live_status(campaign_config).get("running"))
    elif action == "reconcile":
        code = autonomous_campaign.run_autonomous_campaign(
            campaign_config,
            once=True,
            # Let autonomous reconciliation inspect completed predecessor
            # artifacts before it considers starting a fresh trial.
            new_trial=False,
            client=client,
            stream=output,
        )
        recovery_code = None
        recovery_detail = ""
        recovery_running = bool(agent_campaign_loop.campaign_live_status(campaign_config).get("running"))
    else:
        raise MetaControllerError(f"unsupported campaign control action: {action}")
    effective_code = recovery_code if action == "stop" and recovery_code is not None else code
    return {
        "action": action,
        "returncode": int(effective_code),
        "stop_returncode": int(code),
        "recovery_returncode": recovery_code,
        "replacement_running": recovery_running,
        "detail": recovery_detail or output.getvalue()[-4000:],
    }


def _meta_state(root: Path, campaign_config: dict[str, Any]) -> dict[str, Any]:
    state_path = root / str(campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    return load_state(state_path) if state_path.exists() else {"controller_state": {}}


def _meta_loop_owns_state(root: Path, campaign_config: dict[str, Any], pid: int) -> bool:
    """Return whether ``pid`` is still the loop recorded as meta-controller."""

    meta = _meta_state(root, campaign_config).get("controller_state", {}).get("meta_controller", {})
    return isinstance(meta, dict) and meta.get("pid") == pid


def _set_meta_status(
    root: Path,
    campaign_config: dict[str, Any],
    updates: dict[str, Any],
    *,
    actor: str = "meta_controller",
) -> None:
    state_path = root / str(campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    transactional_update(
        state_path,
        lambda state: update_controller_state(state, {"meta_controller": updates}, actor=actor),
    )


def stop_loop(root: str | Path, campaign_config: dict[str, Any], *, reason: str = "dashboard stop requested") -> bool:
    root_path = Path(root).resolve()
    state = _meta_state(root_path, campaign_config)
    meta = state.get("controller_state", {}).get("meta_controller", {})
    pid = meta.get("pid")
    if not _pid_running(pid):
        _set_meta_status(root_path, campaign_config, {**meta, "status": "stopped", "stop_reason": reason, "pid": None})
        return False
    _set_meta_status(root_path, campaign_config, {**meta, "status": "stopping", "stop_reason": reason})
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def request_run_now(root: str | Path, campaign_config: dict[str, Any], *, requested_by: str = "dashboard_user") -> dict[str, Any]:
    """Wake a compatible scheduler, or report that a one-shot may be spawned."""

    root_path = Path(root).resolve()
    state = _meta_state(root_path, campaign_config)
    meta = state.get("controller_state", {}).get("meta_controller", {})
    meta = meta if isinstance(meta, dict) else {}
    pid = meta.get("pid")
    if _pid_running(pid):
        if meta.get("mode") == "loop" and meta.get("run_now_supported") and RUN_NOW_SIGNAL is not None:
            try:
                os.kill(int(pid), RUN_NOW_SIGNAL)
            except OSError as exc:
                raise MetaControllerError(f"could not wake meta-controller scheduler: {exc}") from exc
            _set_meta_status(
                root_path,
                campaign_config,
                {
                    **meta,
                    "status": "run_now_requested",
                    "run_now_requested_at": utc_now(),
                    "run_now_requested_by": requested_by,
                },
                actor=requested_by,
            )
            return {"status": "requested", "mode": "scheduler_wakeup", "pid": int(pid)}
        return {"status": "already_running", "mode": meta.get("mode", "once"), "pid": int(pid)}
    return {"status": "not_running", "mode": "one_shot_required"}


def run_loop(root: str | Path, campaign_config: dict[str, Any], *, client: Any | None = None, sleep_fn: Callable[[float], None] = time.sleep) -> int:
    root_path = Path(root).resolve()
    config = load_meta_config(root_path, campaign_config)
    pid = os.getpid()
    meta = _meta_state(root_path, campaign_config).get("controller_state", {}).get("meta_controller", {})
    if _pid_running(meta.get("pid")) and int(meta.get("pid")) != pid:
        return 1
    wake_event = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    def _run_now(_signum: int, _frame: Any) -> None:
        wake_event.set()

    signal.signal(signal.SIGTERM, _stop)
    if RUN_NOW_SIGNAL is not None:
        signal.signal(RUN_NOW_SIGNAL, _run_now)
    _set_meta_status(root_path, campaign_config, {"status": "running", "pid": pid, "mode": "loop", "run_now_supported": RUN_NOW_SIGNAL is not None, "started_at": utc_now(), "interval_seconds": int(config["interval_seconds"])})
    try:
        while True:
            requested_now = wake_event.is_set()
            wake_event.clear()
            report = run_once(
                root_path,
                campaign_config,
                client=client,
                invocation_source="dashboard_user" if requested_now else "scheduled",
                execution_mode="loop",
            )
            if report["status"] == "meta_controller_safe_stop":
                if _meta_loop_owns_state(root_path, campaign_config, pid):
                    _set_meta_status(
                        root_path,
                        campaign_config,
                        {
                            "status": "meta_controller_safe_stop",
                            "pid": None,
                            "next_run_at": None,
                            "stopped_at": utc_now(),
                            "stop_reason": "repeated supervisory failures",
                        },
                    )
                return 2
            next_at = datetime.now(timezone.utc).timestamp() + int(config["interval_seconds"])
            _set_meta_status(root_path, campaign_config, {"status": "running", "pid": pid, "mode": "loop", "run_now_supported": RUN_NOW_SIGNAL is not None, "last_run_id": report["id"], "last_run_at": report["completed_at"], "next_run_at": datetime.fromtimestamp(next_at, timezone.utc).isoformat(timespec="seconds"), "mandate_version": report["mandate_version"], "architecture_version": report.get("architecture_version", "unavailable"), "summary": report["diagnosis"].get("summary", ""), "last_invocation_source": report.get("invocation_source", "scheduled")})
            if sleep_fn is time.sleep:
                deadline = time.monotonic() + int(config["interval_seconds"])
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or wake_event.wait(min(30.0, remaining)):
                        break
                    if _meta_loop_owns_state(root_path, campaign_config, pid):
                        _set_meta_status(
                            root_path,
                            campaign_config,
                            {
                                "status": "running",
                                "pid": pid,
                                "mode": "loop",
                                "run_now_supported": RUN_NOW_SIGNAL is not None,
                                "last_run_id": report["id"],
                                "last_run_at": report["completed_at"],
                                "next_run_at": datetime.fromtimestamp(next_at, timezone.utc).isoformat(timespec="seconds"),
                                "mandate_version": report["mandate_version"],
                                "architecture_version": report.get("architecture_version", "unavailable"),
                                "summary": report["diagnosis"].get("summary", ""),
                                "last_invocation_source": report.get("invocation_source", "scheduled"),
                            },
                        )
            else:
                sleep_fn(int(config["interval_seconds"]))
    except KeyboardInterrupt:
        # A superseding loop may have taken ownership while this process was
        # shutting down.  Do not let the old process overwrite its live state.
        if _meta_loop_owns_state(root_path, campaign_config, pid):
            _set_meta_status(root_path, campaign_config, {"status": "stopped", "pid": None, "stopped_at": utc_now(), "stop_reason": "stop requested"})
        return 0
    except Exception as exc:
        # Preserve an actionable state if an unexpected scheduler-level error
        # occurs outside run_once's bounded error report. Without this, the
        # last successful cycle leaves a false running status forever.
        if _meta_loop_owns_state(root_path, campaign_config, pid):
            _set_meta_status(
                root_path,
                campaign_config,
                {
                    "status": "failed",
                    "pid": None,
                    "failed_at": utc_now(),
                    "next_run_at": None,
                    "stop_reason": f"scheduler exited unexpectedly: {type(exc).__name__}: {exc}",
                    "summary": f"Meta-controller scheduler exited unexpectedly: {type(exc).__name__}: {exc}",
                },
            )
        return 1


def cli(argv: list[str], *, root: str | Path) -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous supervisory meta-controller.")
    parser.add_argument("campaign_config")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--continue", dest="continue_loop", action="store_true")
    mode.add_argument("--stop", action="store_true")
    parser.add_argument("--invocation-source", default="cli")
    args = parser.parse_args(argv)
    from src.agent_campaign_loop import load_campaign_config
    campaign_config = load_campaign_config(args.campaign_config)
    if args.once:
        report = run_once(root, campaign_config, invocation_source=args.invocation_source)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "completed" else 2
    if args.stop:
        return 0 if stop_loop(root, campaign_config) else 1
    return run_loop(root, campaign_config) if args.start or args.continue_loop else 2


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:], root=Path(__file__).resolve().parents[1]))
