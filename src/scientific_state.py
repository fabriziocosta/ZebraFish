"""Persistent scientific state for autonomous experiment campaigns.

The state file is deliberately small and auditable.  Large histories and
plots remain in run artifacts; this module stores the entities and relations
that the autonomous controller needs to reason about.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable
import uuid

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - JSON fallback is useful in minimal envs.
    yaml = None


STATE_VERSION = 1
ENTITY_COLLECTIONS = (
    "components",
    "datasets",
    "trials",
    "experiments",
    "observations",
    "hypotheses",
    "beliefs",
    "questions",
    "candidate_experiments",
)
RELATION_TYPES = {
    "supports",
    "contradicts",
    "refines",
    "supersedes",
    "alternative_to",
    "derived_from",
    "tests",
    "produced",
    "replicates",
    "extends",
    "differs_from",
    "reuses_checkpoint",
    "motivates",
    "rules_out",
    "prioritises",
    "blocks",
    "uses_component",
    "trained_on",
    "evaluated_on",
    "modifies_parameter",
}
IMMUTABLE_COLLECTIONS = {"experiments", "observations"}


class ScientificStateError(ValueError):
    """Base error for invalid scientific state or operations."""


class ImmutableEntityError(ScientificStateError):
    """Raised when an immutable historical record is changed."""


class OperationConflictError(ScientificStateError):
    """Raised when an operation's expected value is stale."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "project": {},
        "entities": {collection: {} for collection in ENTITY_COLLECTIONS},
        "relations": [],
        "controller_state": {},
        "audit_log": [],
    }


def ensure_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    state = deepcopy(payload) if isinstance(payload, dict) else empty_state()
    state.setdefault("version", STATE_VERSION)
    state.setdefault("project", {})
    entities = state.setdefault("entities", {})
    for collection in ENTITY_COLLECTIONS:
        if not isinstance(entities.get(collection), dict):
            entities[collection] = {}
    if not isinstance(state.get("relations"), list):
        state["relations"] = []
    if not isinstance(state.get("controller_state"), dict):
        state["controller_state"] = {}
    if not isinstance(state.get("audit_log"), list):
        state["audit_log"] = []
    return state


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ScientificStateError("scientific state must be a mapping")
    normalized = ensure_state(state)
    if normalized["version"] != STATE_VERSION:
        raise ScientificStateError(
            f"unsupported scientific state version {normalized['version']!r}; expected {STATE_VERSION}"
        )
    for relation in normalized["relations"]:
        if not isinstance(relation, dict):
            raise ScientificStateError("relations must contain mappings")
        relation_type = str(relation.get("type", ""))
        if relation_type not in RELATION_TYPES:
            raise ScientificStateError(f"unsupported relation type: {relation_type!r}")
        for required in ("source", "target"):
            if not relation.get(required):
                raise ScientificStateError(f"relation missing {required}: {relation!r}")
    for collection, records in normalized["entities"].items():
        if not isinstance(records, dict):
            raise ScientificStateError(f"entities.{collection} must be a mapping")
        for entity_id, record in records.items():
            if not isinstance(record, dict):
                raise ScientificStateError(f"{collection}.{entity_id} must be a mapping")
            if collection in IMMUTABLE_COLLECTIONS and record.get("status") in {
                "completed",
                "failed",
                "terminated",
                "semantic_early_stopped",
            }:
                _validate_provenance(record, f"{collection}.{entity_id}")
    return normalized


def _validate_provenance(record: dict[str, Any], label: str) -> None:
    provenance = record.get("provenance")
    if provenance is None:
        # Migrated records may have provenance at execution level, but all
        # newly-created records must expose it at the record boundary.
        if not record.get("execution", {}).get("provenance"):
            raise ScientificStateError(f"immutable record {label} is missing provenance")


def load_state(path: str | Path = "state/scientific_state.yaml") -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return empty_state()
    text = target.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return validate_state(ensure_state(payload))


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    normalized = validate_state(state)
    target.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        rendered = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
    else:
        rendered = json.dumps(normalized, indent=2, sort_keys=False)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(target)


def get_path(state: dict[str, Any], path: str) -> Any:
    value: Any = state
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _parent_for_path(state: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    components = [component for component in path.split(".") if component]
    if not components:
        raise ScientificStateError("operation path cannot be empty")
    parent: Any = state
    for component in components[:-1]:
        if not isinstance(parent, dict) or component not in parent:
            raise ScientificStateError(f"operation path does not exist: {path}")
        parent = parent[component]
    if not isinstance(parent, dict):
        raise ScientificStateError(f"operation parent is not a mapping: {path}")
    return parent, components[-1]


def _immutable_path(path: str) -> bool:
    components = path.split(".")
    return len(components) >= 3 and components[0] == "entities" and components[1] in IMMUTABLE_COLLECTIONS


def _record_audit(state: dict[str, Any], *, actor: str, operation: dict[str, Any], now: str) -> None:
    state["audit_log"].append(
        {
            "id": f"op_{uuid.uuid4().hex[:12]}",
            "created_at": now,
            "actor": actor,
            "operation": deepcopy(operation),
        }
    )


def _validate_semantic_record(collection: str, record: dict[str, Any]) -> None:
    if collection in {"hypotheses", "beliefs", "questions", "candidate_experiments"}:
        if not record.get("provenance"):
            raise ScientificStateError(f"{collection} records require provenance")
        if not record.get("created_at"):
            raise ScientificStateError(f"{collection} records require created_at")


def apply_operations(
    state: dict[str, Any],
    operations: Iterable[dict[str, Any]],
    *,
    actor: str = "llm",
    now: str | None = None,
) -> dict[str, Any]:
    """Apply validated operation patches transactionally to a copied state."""

    candidate = validate_state(deepcopy(state))
    timestamp = now or utc_now()
    operations_list = list(operations)
    for operation in operations_list:
        if not isinstance(operation, dict):
            raise ScientificStateError("each state operation must be a mapping")
        kind = str(operation.get("operation", ""))
        path = str(operation.get("path", ""))
        if kind == "create":
            parent, key = _parent_for_path(candidate, path)
            if key in parent:
                raise OperationConflictError(f"entity already exists at {path}")
            value = deepcopy(operation.get("value"))
            if not isinstance(value, dict):
                raise ScientificStateError(f"create operation requires mapping value: {path}")
            collection = path.split(".")[2] if path.startswith("entities.") and len(path.split(".")) > 2 else ""
            value.setdefault("created_at", timestamp)
            if collection in ENTITY_COLLECTIONS:
                _validate_semantic_record(collection, value)
            parent[key] = value
        elif kind in {"update", "transition"}:
            if _immutable_path(path):
                raise ImmutableEntityError(f"immutable historical path cannot be updated: {path}")
            parent, key = _parent_for_path(candidate, path)
            expected = operation.get("expected_old", operation.get("old_value", None))
            if "expected_old" in operation or "old_value" in operation:
                if parent.get(key) != expected:
                    raise OperationConflictError(
                        f"stale operation at {path}: expected {expected!r}, found {parent.get(key)!r}"
                    )
            parent[key] = deepcopy(operation.get("value"))
        elif kind == "append":
            target = get_path(candidate, path)
            if not isinstance(target, list):
                raise ScientificStateError(f"append target must be a list: {path}")
            target.append(deepcopy(operation.get("value")))
        elif kind == "relation":
            relation = deepcopy(operation.get("value", operation))
            relation.pop("operation", None)
            relation_type = str(relation.get("type", ""))
            if relation_type not in RELATION_TYPES:
                raise ScientificStateError(f"unsupported relation type: {relation_type!r}")
            if not relation.get("source") or not relation.get("target"):
                raise ScientificStateError("relation operations require source and target")
            relation.setdefault("created_at", timestamp)
            candidate["relations"].append(relation)
        else:
            raise ScientificStateError(f"unsupported operation: {kind!r}")
        _record_audit(candidate, actor=actor, operation=operation, now=timestamp)
    return validate_state(candidate)


def record_entity(
    state: dict[str, Any],
    collection: str,
    entity_id: str,
    record: dict[str, Any],
    *,
    actor: str = "controller",
    now: str | None = None,
) -> dict[str, Any]:
    if collection not in ENTITY_COLLECTIONS:
        raise ScientificStateError(f"unknown entity collection: {collection}")
    timestamp = now or utc_now()
    value = deepcopy(record)
    value.setdefault("id", entity_id)
    value.setdefault("created_at", timestamp)
    value.setdefault("provenance", {"created_by": actor, "created_at": timestamp})
    return apply_operations(
        state,
        [{"operation": "create", "path": f"entities.{collection}.{entity_id}", "value": value}],
        actor=actor,
        now=timestamp,
    )


def update_controller_state(
    state: dict[str, Any],
    updates: dict[str, Any],
    *,
    actor: str = "controller",
    now: str | None = None,
) -> dict[str, Any]:
    """Update mutable controller fields and record one audit entry."""

    candidate = validate_state(deepcopy(state))
    timestamp = now or utc_now()
    candidate["controller_state"].update(deepcopy(updates))
    _record_audit(
        candidate,
        actor=actor,
        operation={"operation": "controller_update", "value": deepcopy(updates)},
        now=timestamp,
    )
    return candidate


def compact_context(state: dict[str, Any], *, limit: int = 8) -> dict[str, Any]:
    """Return the bounded scientific context sent to an LLM."""

    entities = state.get("entities", {})
    def recent(collection: str) -> list[dict[str, Any]]:
        return [
            {"id": entity_id, **record}
            for entity_id, record in list(entities.get(collection, {}).items())[-limit:]
        ]

    return {
        "project": state.get("project", {}),
        "controller_state": state.get("controller_state", {}),
        "hypotheses": recent("hypotheses"),
        "beliefs": recent("beliefs"),
        "questions": recent("questions"),
        "observations": recent("observations"),
        "candidate_experiments": recent("candidate_experiments"),
        "trials": recent("trials"),
    }
