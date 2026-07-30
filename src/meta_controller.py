"""Supervisory controller for daily, bounded self-healing of the experiment loop.

The meta-controller is deliberately separate from the scientific campaign
controller.  It may diagnose and repair operational code only through a fixed
allowlist and a verification suite.  Scientific objective changes are always
recorded as proposals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable
import uuid

from src.scientific_state import (
    ScientificStateError,
    load_state,
    record_entity,
    save_state,
    update_controller_state,
)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


META_DECISIONS = {"no_action", "patch", "campaign_control", "proposal", "safe_stop"}
SEVERITIES = {"info", "warning", "critical"}
CAMPAIGN_ACTIONS = {"stop", "continue", "reconcile"}


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


def _campaign_state_path(root: Path, campaign_config: dict[str, Any]) -> Path:
    value = campaign_config.get("artifacts", {}).get("state_path")
    return root / str(value or f"artifacts/campaigns/{campaign_config['campaign']['id']}/campaign_state.json")


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


def collect_snapshot(root: str | Path, campaign_config: dict[str, Any]) -> dict[str, Any]:
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
        "recent_meta_reports": reports,
        "repository": {
            "revision": _git(root_path, ["rev-parse", "HEAD"]),
            "status": _git(root_path, ["status", "--short"]),
            "diff_stat": _git(root_path, ["diff", "--stat"]),
        },
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
                "required": ["mandate_version", "decision", "diagnosis", "actions", "proposal_only_changes", "rollback_plan", "unresolved_risks"],
                "properties": {
                    "mandate_version": {"type": "string"},
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
    required = {"mandate_version", "decision", "diagnosis", "actions", "proposal_only_changes", "rollback_plan", "unresolved_risks"}
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
    return tuple(paths)


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
    state = record_entity(state, "meta_controller_runs", report["id"], report, actor="meta_controller")
    state = update_controller_state(state, {"meta_controller": report["controller_status"]}, actor="meta_controller")
    save_state(state_path, state)
    report_root = root / str(load_meta_config(root, campaign_config).get("report_root", "state/meta_controller/reports"))
    _write_json_atomic(report_root / f"{report['id']}.json", report)


def _build_prompt(mandate: str, snapshot: dict[str, Any], version: str) -> str:
    return "\n\n".join([
        "You are the ZebraFish meta-controller. Read and obey the constitutional mandate below.",
        f"Mandate version: {version}",
        mandate,
        "Diagnose only from the bounded evidence snapshot. Return strict JSON matching the response schema.",
        "Patch actions must contain unified diffs only. Never include shell commands. Scientific objective changes are proposal-only.",
        f"Evidence snapshot:\n{json.dumps(snapshot, indent=2, sort_keys=True)}",
    ])


def request_decision(root: Path, campaign_config: dict[str, Any], config: dict[str, Any], mandate: str, version: str, snapshot: dict[str, Any], *, client: Any | None = None) -> dict[str, Any]:
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    agent = campaign_config.get("agent", {})
    response = client.responses.create(
        model=agent.get("model", "gpt-5.3-codex"),
        reasoning={"effort": agent.get("reasoning_effort", "medium")},
        max_output_tokens=int(agent.get("max_output_tokens", 5000)),
        text=_schema(),
        input=_build_prompt(mandate, snapshot, version),
    )
    output = getattr(response, "output_text", None)
    if not isinstance(output, str):
        raise MetaDecisionError("LLM response did not contain output_text")
    decision = parse_decision(output)
    if decision["mandate_version"] != version:
        raise MetaDecisionError("LLM returned a mandate version different from the loaded mandate")
    return decision


def apply_patch_in_worktree(root: Path, config: dict[str, Any], patch: str, run_id: str, *, verification_names: Iterable[str] | None = None, runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    paths = validate_patch(patch, config)
    names = list(verification_names or config.get("default_verifications", []))
    if not names:
        raise PatchSafetyError("patch action must select at least one allowlisted verification")
    commands = validate_verifications([str(name) for name in names], config)
    worktree = root / str(config.get("worktree_root", "state/meta_controller/worktrees")) / run_id
    worktree.parent.mkdir(parents=True, exist_ok=True)
    try:
        created = subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=root, capture_output=True, text=True, check=False)
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
        if status:
            return {"status": "overlapping_user_changes", "paths": paths, "worktree": str(worktree), "verification": verification, "overlap": status}
        apply_main = subprocess.run(["git", "apply", "-"], cwd=root, input=patch, capture_output=True, text=True, check=False)
        if apply_main.returncode != 0:
            return {"status": "main_application_failed", "paths": paths, "worktree": str(worktree), "verification": verification, "error": apply_main.stderr.strip()}
        return {"status": "applied", "paths": paths, "worktree": str(worktree), "verification": verification, "rollback_patch": patch}
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root, capture_output=True, text=True, check=False)


def run_once(root: str | Path, campaign_config: dict[str, Any], *, client: Any | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_meta_config(root_path, campaign_config)
    started = utc_now()
    run_id = f"meta_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    controller_status = {"status": "running", "run_id": run_id, "started_at": started}
    try:
        mandate, mandate_version, mandate_path = read_mandate(root_path, config)
        snapshot = collect_snapshot(root_path, campaign_config)
        decision = request_decision(root_path, campaign_config, config, mandate, mandate_version, snapshot, client=client)
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
            "mandate_version": mandate_version, "mandate_path": mandate_path,
            "started_at": started, "completed_at": utc_now(),
            "diagnosis": decision["diagnosis"], "evidence_references": decision["diagnosis"].get("evidence_references", []),
            "actions": actions, "proposal_only_changes": decision["proposal_only_changes"],
            "rollback_plan": decision["rollback_plan"], "unresolved_risks": decision["unresolved_risks"],
            "provenance": {"created_by": "meta_controller", "mandate_version": mandate_version, "snapshot": "compact"},
            "controller_status": {"status": status, "last_run_id": run_id, "last_run_at": utc_now(), "mandate_version": mandate_version, "summary": decision["diagnosis"].get("summary", ""), "next_run_at": None},
        }
    except Exception as exc:
        try:
            _, mandate_version, mandate_path = read_mandate(root_path, config)
        except Exception:
            mandate_version, mandate_path = "unavailable", None
        report = {
            "id": run_id, "campaign": campaign_config.get("campaign", {}).get("id"), "status": "failed",
            "mandate_version": mandate_version, "mandate_path": mandate_path,
            "started_at": started, "completed_at": utc_now(),
            "diagnosis": {"summary": f"Meta-controller could not complete: {type(exc).__name__}: {exc}", "severity": "critical", "evidence_references": [], "root_causes": []},
            "evidence_references": [], "actions": [], "proposal_only_changes": [], "rollback_plan": "No patch was applied.", "unresolved_risks": [str(exc)],
            "provenance": {"created_by": "meta_controller", "error": type(exc).__name__},
            "controller_status": {"status": "failed", "last_run_id": run_id, "last_run_at": utc_now(), "mandate_version": mandate_version, "summary": str(exc), "next_run_at": None},
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
        live_before = agent_campaign_loop.campaign_live_status(campaign_config).get("running")
        code = autonomous_campaign.run_autonomous_campaign(
            campaign_config,
            once=True,
            new_trial=not live_before,
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


def _set_meta_status(root: Path, campaign_config: dict[str, Any], updates: dict[str, Any]) -> None:
    state_path = root / str(campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    state = _meta_state(root, campaign_config)
    state = update_controller_state(state, {"meta_controller": updates}, actor="meta_controller")
    save_state(state_path, state)


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


def run_loop(root: str | Path, campaign_config: dict[str, Any], *, client: Any | None = None, sleep_fn: Callable[[float], None] = time.sleep) -> int:
    root_path = Path(root).resolve()
    config = load_meta_config(root_path, campaign_config)
    pid = os.getpid()
    meta = _meta_state(root_path, campaign_config).get("controller_state", {}).get("meta_controller", {})
    if _pid_running(meta.get("pid")) and int(meta.get("pid")) != pid:
        return 1
    def _stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    _set_meta_status(root_path, campaign_config, {"status": "running", "pid": pid, "started_at": utc_now(), "interval_seconds": int(config["interval_seconds"])})
    try:
        while True:
            report = run_once(root_path, campaign_config, client=client)
            if report["status"] == "meta_controller_safe_stop":
                return 2
            next_at = datetime.now(timezone.utc).timestamp() + int(config["interval_seconds"])
            _set_meta_status(root_path, campaign_config, {"status": "running", "pid": pid, "last_run_id": report["id"], "last_run_at": report["completed_at"], "next_run_at": datetime.fromtimestamp(next_at, timezone.utc).isoformat(timespec="seconds"), "mandate_version": report["mandate_version"], "summary": report["diagnosis"].get("summary", "")})
            sleep_fn(int(config["interval_seconds"]))
    except KeyboardInterrupt:
        _set_meta_status(root_path, campaign_config, {"status": "stopped", "pid": None, "stopped_at": utc_now(), "stop_reason": "stop requested"})
        return 0


def cli(argv: list[str], *, root: str | Path) -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous supervisory meta-controller.")
    parser.add_argument("campaign_config")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--continue", dest="continue_loop", action="store_true")
    mode.add_argument("--stop", action="store_true")
    args = parser.parse_args(argv)
    from src.agent_campaign_loop import load_campaign_config
    campaign_config = load_campaign_config(args.campaign_config)
    if args.once:
        report = run_once(root, campaign_config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "completed" else 2
    if args.stop:
        return 0 if stop_loop(root, campaign_config) else 1
    return run_loop(root, campaign_config) if args.start or args.continue_loop else 2


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:], root=Path(__file__).resolve().parents[1]))
