"""Persistent scientific state for autonomous experiment campaigns.

The state file is deliberately small and auditable.  Large histories and
plots remain in run artifacts; this module stores the entities and relations
that the autonomous controller needs to reason about.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Iterable, Iterator
import uuid

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - JSON fallback is useful in minimal envs.
    yaml = None


STATE_VERSION = 2
PROTOCOL_VERSION = "scientific-loop-v3"
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
    "belief_updates",
    "replicate_groups",
    "meta_controller_runs",
    "candidate_events",
    "replicate_group_events",
    "lockbox_confirmations",
    "split_manifests",
    "launch_events",
    "domain_constraints",
    "domain_calibrations",
    "domain_evaluations",
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
    "evaluates_constraint",
}
IMMUTABLE_COLLECTIONS = {
    "experiments",
    "observations",
    "trials",
    "belief_updates",
    "replicate_groups",
    "meta_controller_runs",
    "candidate_events",
    "replicate_group_events",
    "lockbox_confirmations",
    "split_manifests",
    "launch_events",
    "domain_constraints",
    "domain_calibrations",
    "domain_evaluations",
}


class ScientificStateError(ValueError):
    """Base error for invalid scientific state or operations."""


class ImmutableEntityError(ScientificStateError):
    """Raised when an immutable historical record is changed."""


class OperationConflictError(ScientificStateError):
    """Raised when an operation's expected value is stale."""


class StateRevisionConflictError(ScientificStateError):
    """Raised when a writer attempts to replace a newer scientific state."""


class StateLockError(ScientificStateError):
    """Raised when the scientific state lock cannot be acquired."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "writer_protocol": PROTOCOL_VERSION,
        "state_revision": 0,
        "last_writer": None,
        "updated_at": None,
        "project": {},
        "entities": {collection: {} for collection in ENTITY_COLLECTIONS},
        "relations": [],
        "controller_state": {},
        "audit_log": [],
    }


def ensure_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return empty_state()
    if not isinstance(payload, dict):
        raise ScientificStateError("scientific state must be a mapping")
    state = deepcopy(payload)
    version = state.get("version", 1)
    if version == 1:
        state = migrate_state_payload(state)
    elif version != STATE_VERSION:
        raise ScientificStateError(f"unsupported scientific state version {version!r}; expected {STATE_VERSION}")
    state.setdefault("state_revision", 0)
    state.setdefault("protocol_version", "legacy-v2" if version == 1 else PROTOCOL_VERSION)
    state.setdefault("writer_protocol", state.get("protocol_version"))
    state.setdefault("last_writer", None)
    state.setdefault("updated_at", None)
    state.setdefault("project", {})
    if not isinstance(state["project"], dict):
        raise ScientificStateError("project must be a mapping")
    entities = state.setdefault("entities", {})
    if not isinstance(entities, dict):
        raise ScientificStateError("entities must be a mapping")
    for collection in ENTITY_COLLECTIONS:
        if collection not in entities:
            entities[collection] = {}
        elif not isinstance(entities[collection], dict):
            raise ScientificStateError(f"entities.{collection} must be a mapping")
    if "relations" not in state:
        state["relations"] = []
    elif not isinstance(state["relations"], list):
        raise ScientificStateError("relations must be a list")
    if "controller_state" not in state:
        state["controller_state"] = {}
    elif not isinstance(state["controller_state"], dict):
        raise ScientificStateError("controller_state must be a mapping")
    if "audit_log" not in state:
        state["audit_log"] = []
    elif not isinstance(state["audit_log"], list):
        raise ScientificStateError("audit_log must be a list")
    return state


def migrate_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 payload in memory without discarding historical records."""

    migrated = deepcopy(payload)
    if migrated.get("version", 1) != 1:
        return migrated
    entities = migrated.setdefault("entities", {})
    if not isinstance(entities, dict):
        raise ScientificStateError("v1 entities must be a mapping")
    entities.setdefault("belief_updates", {})
    entities.setdefault("replicate_groups", {})
    migrated["version"] = STATE_VERSION
    migrated.setdefault("state_revision", 0)
    migrated.setdefault("protocol_version", "legacy-v2")
    migrated.setdefault("writer_protocol", "legacy-v2")
    migrated.setdefault("last_writer", "state_migration_v1")
    migrated.setdefault("updated_at", None)
    return migrated


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ScientificStateError("scientific state must be a mapping")
    normalized = ensure_state(state)
    if not isinstance(normalized["state_revision"], int) or normalized["state_revision"] < 0:
        raise ScientificStateError("state_revision must be a non-negative integer")
    entity_collections: dict[str, list[str]] = {}
    for collection, records in normalized["entities"].items():
        for entity_id in records:
            entity_collections.setdefault(str(entity_id), []).append(collection)
    relation_pairs: set[tuple[str, str, str, str, str]] = set()
    relation_directions: dict[tuple[str, str], set[str]] = {}
    for relation in normalized["relations"]:
        if not isinstance(relation, dict):
            raise ScientificStateError("relations must contain mappings")
        relation_type = str(relation.get("type", ""))
        if relation_type not in RELATION_TYPES:
            raise ScientificStateError(f"unsupported relation type: {relation_type!r}")
        for required in ("source", "target"):
            if not relation.get(required):
                raise ScientificStateError(f"relation missing {required}: {relation!r}")
        source = str(relation["source"])
        target = str(relation["target"])
        source_collections = entity_collections.get(source, [])
        target_collections = entity_collections.get(target, [])
        source_type = relation.get("source_type")
        target_type = relation.get("target_type")
        if source_type is None:
            if len(source_collections) != 1:
                raise ScientificStateError(f"relation source type is ambiguous or unknown: {relation!r}")
            source_type = source_collections[0]
            relation["source_type"] = source_type
        if target_type is None:
            if len(target_collections) != 1:
                raise ScientificStateError(f"relation target type is ambiguous or unknown: {relation!r}")
            target_type = target_collections[0]
            relation["target_type"] = target_type
        if source_type not in normalized["entities"] or source not in normalized["entities"].get(source_type, {}):
            raise ScientificStateError(f"relation references unknown typed source: {relation!r}")
        if target_type not in normalized["entities"] or target not in normalized["entities"].get(target_type, {}):
            raise ScientificStateError(f"relation references unknown typed target: {relation!r}")
        key = (relation_type, str(source_type), source, str(target_type), target)
        if key in relation_pairs:
            raise ScientificStateError(f"duplicate relation: {relation!r}")
        relation_pairs.add(key)
        relation_directions.setdefault((str(source_type), source, str(target_type), target), set()).add(relation_type)
    for endpoints, relation_types in relation_directions.items():
        if "supports" in relation_types and "contradicts" in relation_types:
            raise ScientificStateError(f"relation pair is both supporting and contradicting: {endpoints!r}")
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
            if collection == "candidate_experiments" and record.get("status") == "selected_for_execution":
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
    with state_lock(target):
        text = target.read_text(encoding="utf-8")
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return validate_state(ensure_state(payload))


def _state_writer_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def state_lock(path: str | Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Serialize scientific-state reads and writes using an advisory lock."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_target = _lock_path(target)
    handle = lock_target.open("a+", encoding="utf-8")
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while time.monotonic() < deadline:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps({"pid": os.getpid(), "writer": _state_writer_id()}))
                handle.flush()
                yield
                break
            except BlockingIOError:
                time.sleep(0.05)
        else:
            raise StateLockError(f"timed out acquiring scientific state lock: {lock_target}")
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _save_state_unlocked(target: Path, state: dict[str, Any], *, expected_revision: int | None = None) -> None:
    disk_revision = 0
    if target.exists():
        text = target.read_text(encoding="utf-8")
        disk_payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
        disk_revision = int(ensure_state(disk_payload).get("state_revision", 0))
    supplied_revision = state.get("state_revision", 0) if expected_revision is None else expected_revision
    if int(supplied_revision) != disk_revision:
        raise StateRevisionConflictError(
            f"stale scientific state revision: expected {supplied_revision}, found {disk_revision}"
        )
    normalized = validate_state(state)
    normalized["state_revision"] = disk_revision + 1
    normalized["last_writer"] = _state_writer_id()
    normalized["writer_protocol"] = PROTOCOL_VERSION
    normalized["protocol_version"] = PROTOCOL_VERSION
    normalized["updated_at"] = utc_now()
    if yaml is not None:
        rendered = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
    else:
        rendered = json.dumps(normalized, indent=2, sort_keys=False)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(str(target.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    state.clear()
    state.update(normalized)


def save_state(path: str | Path, state: dict[str, Any], *, expected_revision: int | None = None) -> None:
    target = Path(path)
    with state_lock(target):
        _save_state_unlocked(target, state, expected_revision=expected_revision)


@contextmanager
def state_transaction(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read, mutate, validate, and commit one scientific-state transaction."""

    target = Path(path)
    with state_lock(target):
        if target.exists():
            text = target.read_text(encoding="utf-8")
            payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
            state = validate_state(ensure_state(payload))
        else:
            state = empty_state()
        revision = int(state.get("state_revision", 0))
        yield state
        normalized = validate_state(state)
        normalized["state_revision"] = revision
        _save_state_unlocked(target, normalized, expected_revision=revision)


def transactional_update(
    path: str | Path,
    mutator: Callable[[dict[str, Any]], Any],
    *,
    retries: int = 3,
) -> Any:
    """Run one locked state mutation, retrying only revision conflicts."""

    last_error: Exception | None = None
    for _attempt in range(max(1, int(retries))):
        try:
            with state_transaction(path) as state:
                result = mutator(state)
                if isinstance(result, dict) and result is not state:
                    state.clear()
                    state.update(result)
                return result
        except StateRevisionConflictError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise StateRevisionConflictError("scientific-state transaction failed without a conflict detail")


def merge_nonconflicting_states(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge controller snapshots without rewriting immutable evidence.

    The campaign and meta-controller can both update mutable controller
    metadata. Historical entities are unioned and any divergent immutable
    record is a hard conflict rather than an overwrite.
    """

    base = validate_state(deepcopy(current))
    proposed = validate_state(deepcopy(incoming))
    for collection, records in proposed.get("entities", {}).items():
        destination = base.setdefault("entities", {}).setdefault(collection, {})
        for entity_id, record in records.items():
            if entity_id not in destination:
                destination[entity_id] = deepcopy(record)
            elif collection in IMMUTABLE_COLLECTIONS and destination[entity_id] != record:
                raise StateRevisionConflictError(f"immutable scientific record conflict: {collection}.{entity_id}")
    existing_relations = {json.dumps(item, sort_keys=True, default=str) for item in base.get("relations", [])}
    for relation in proposed.get("relations", []):
        rendered = json.dumps(relation, sort_keys=True, default=str)
        if rendered not in existing_relations:
            base["relations"].append(deepcopy(relation))
            existing_relations.add(rendered)
    existing_audits = {str(item.get("id")) for item in base.get("audit_log", []) if isinstance(item, dict)}
    for audit in proposed.get("audit_log", []):
        if isinstance(audit, dict) and str(audit.get("id")) not in existing_audits:
            base["audit_log"].append(deepcopy(audit))
    incoming_controller = proposed.get("controller_state", {})
    if isinstance(incoming_controller, dict):
        for key, value in incoming_controller.items():
            if key == "meta_controller" and isinstance(value, dict) and isinstance(base.get("controller_state", {}).get(key), dict):
                # Campaign/controller snapshots contain the meta-controller
                # state that existed when they were loaded.  They must not
                # overwrite a newer live loop status (or PID) when the
                # campaign later merges its own scientific updates.
                continue
            else:
                base["controller_state"][key] = deepcopy(value)
    return validate_state(base)


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
    if collection in {"hypotheses", "beliefs", "questions", "candidate_experiments", "meta_controller_runs"}:
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
            components = path.split(".")
            if len(components) >= 3 and components[0] == "entities" and components[1] == "candidate_experiments":
                candidate_record = candidate["entities"]["candidate_experiments"].get(components[2], {})
                if isinstance(candidate_record, dict) and candidate_record.get("status") == "selected_for_execution":
                    raise ImmutableEntityError(f"selected candidate cannot be updated: {path}")
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


def append_belief_update(
    state: dict[str, Any],
    update_id: str,
    update: dict[str, Any],
    *,
    actor: str = "controller",
    now: str | None = None,
) -> dict[str, Any]:
    """Append one immutable belief update event.

    Belief records contain the current view; this collection contains the
    append-only history needed to audit how that view was reached.
    """

    value = deepcopy(update)
    timestamp = now or utc_now()
    value.setdefault("id", update_id)
    value.setdefault("created_at", timestamp)
    value.setdefault("provenance", {"created_by": actor, "created_at": timestamp})
    required = ("previous_belief", "new_belief", "triggering_observations", "rationale", "method")
    missing = [field for field in required if field not in value]
    if missing:
        raise ScientificStateError(f"belief update is missing required fields: {', '.join(missing)}")
    return record_entity(state, "belief_updates", update_id, value, actor=actor, now=timestamp)


def append_lifecycle_event(
    state: dict[str, Any],
    collection: str,
    event_id: str,
    *,
    subject_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    actor: str = "controller",
    now: str | None = None,
) -> dict[str, Any]:
    """Append an immutable event for a frozen scientific object."""

    if collection not in {"candidate_events", "replicate_group_events", "launch_events"}:
        raise ScientificStateError(f"unsupported lifecycle-event collection: {collection}")
    timestamp = now or utc_now()
    return record_entity(
        state,
        collection,
        event_id,
        {
            "id": event_id,
            "subject_id": subject_id,
            "event_type": event_type,
            "payload": deepcopy(payload or {}),
            "created_at": timestamp,
            "provenance": {"created_by": actor, "created_at": timestamp},
        },
        actor=actor,
        now=timestamp,
    )


def reserve_launch(
    state: dict[str, Any],
    *,
    reservation_id: str,
    candidate_id: str,
    trial_id: str,
    estimated_gpu_hours: float,
    replicate_group_id: str | None = None,
    replicate_index: int | None = None,
    actor: str = "campaign_controller",
    now: str | None = None,
) -> dict[str, Any]:
    """Create an idempotent launch reservation in controller state."""

    if estimated_gpu_hours < 0:
        raise ScientificStateError("estimated_gpu_hours cannot be negative")
    candidate = state.get("entities", {}).get("candidate_experiments", {}).get(candidate_id)
    if not isinstance(candidate, dict):
        raise ScientificStateError(f"launch reservation references unknown candidate: {candidate_id}")
    timestamp = now or utc_now()
    candidate_state = validate_state(deepcopy(state))
    reservations = candidate_state.setdefault("controller_state", {}).setdefault("launch_reservations", {})
    existing = reservations.get(reservation_id)
    if existing is not None:
        if existing.get("candidate_id") != candidate_id or existing.get("trial_id") != trial_id:
            raise OperationConflictError(f"launch reservation id is already bound to another launch: {reservation_id}")
        return candidate_state
    for existing_id, record in reservations.items():
        if not isinstance(record, dict) or record.get("status") not in {"reserved", "launched"}:
            continue
        if record.get("candidate_id") == candidate_id and record.get("trial_id") == trial_id:
            return candidate_state
    reservations[reservation_id] = {
        "id": reservation_id,
        "candidate_id": candidate_id,
        "trial_id": trial_id,
        "estimated_gpu_hours": float(estimated_gpu_hours),
        "replicate_group_id": replicate_group_id,
        "replicate_index": replicate_index,
        "status": "reserved",
        "created_at": timestamp,
        "provenance": {"created_by": actor, "created_at": timestamp},
    }
    _record_audit(
        candidate_state,
        actor=actor,
        operation={"operation": "launch_reservation", "reservation_id": reservation_id, "candidate_id": candidate_id, "trial_id": trial_id},
        now=timestamp,
    )
    return candidate_state


def update_launch_reservation(
    state: dict[str, Any],
    reservation_id: str,
    *,
    status: str,
    actual_gpu_hours: float | None = None,
    actor: str = "campaign_controller",
    now: str | None = None,
) -> dict[str, Any]:
    if status not in {"reserved", "launched", "completed", "released", "failed"}:
        raise ScientificStateError(f"unsupported launch reservation status: {status}")
    candidate = validate_state(deepcopy(state))
    reservations = candidate.get("controller_state", {}).get("launch_reservations", {})
    if reservation_id not in reservations:
        raise ScientificStateError(f"unknown launch reservation: {reservation_id}")
    record = reservations[reservation_id]
    record["status"] = status
    if actual_gpu_hours is not None:
        record["actual_gpu_hours"] = float(actual_gpu_hours)
    record["updated_at"] = now or utc_now()
    _record_audit(candidate, actor=actor, operation={"operation": "launch_reservation_update", "reservation_id": reservation_id, "status": status}, now=now or utc_now())
    return candidate


def compact_context(state: dict[str, Any], *, limit: int = 8) -> dict[str, Any]:
    """Return the bounded scientific context sent to an LLM."""

    entities = state.get("entities", {})
    def recent(collection: str) -> list[dict[str, Any]]:
        return [
            {"id": entity_id, **record}
            for entity_id, record in list(entities.get(collection, {}).items())[-limit:]
        ]

    domain_evaluations = []
    for entity_id, record in list(entities.get("domain_evaluations", {}).items())[-limit:]:
        domain_evaluations.append(
            {
                "id": entity_id,
                "experiment_id": record.get("stage_experiment_id") or record.get("experiment_id"),
                "contract_id": record.get("contract_id"),
                "contract_hash": record.get("contract_hash"),
                "objective_eligibility": record.get("objective_eligibility"),
                "umap_used_for_decision": False,
                "constraints": [
                    {
                        "id": constraint.get("id"),
                        "title": constraint.get("title"),
                        "role": constraint.get("role"),
                        "status": constraint.get("status"),
                        "checks": [
                            {
                                "metric": check.get("metric"),
                                "status": check.get("status"),
                                "delta": check.get("delta"),
                                "reason": check.get("reason"),
                            }
                            for check in constraint.get("checks", [])
                        ],
                    }
                    for constraint in record.get("constraints", [])
                ],
            }
        )

    return {
        "project": state.get("project", {}),
        "controller_state": state.get("controller_state", {}),
        "hypotheses": recent("hypotheses"),
        "beliefs": recent("beliefs"),
        "questions": recent("questions"),
        "observations": recent("observations"),
        "domain_guidance": {
            "constraints": recent("domain_constraints"),
            "calibrations": recent("domain_calibrations"),
            "evaluations": domain_evaluations,
            "decision_policy": "hard identifiability guardrails, then secondary biological geometry; UMAP is visualization only",
        },
        "candidate_experiments": recent("candidate_experiments"),
        "trials": recent("trials"),
    }
