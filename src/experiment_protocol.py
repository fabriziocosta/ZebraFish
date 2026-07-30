"""Deterministic experiment-protocol utilities.

This module keeps scientific comparison rules independent from the LLM.  It
does not decide which intervention is interesting; it defines what evidence
is sufficient to compare one intervention with another.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import sqrt
import math
import platform
import sys
from pathlib import Path
import subprocess
from statistics import mean, stdev
from datetime import datetime, timezone
from typing import Any, Iterable


PROTOCOL_VERSION = "replicate-lockbox-v1"
DEFAULT_SEEDS = (0, 1, 2)


def _t_critical_975(degrees_of_freedom: int) -> float:
    # Exact two-sided 95% Student-t critical values for the small samples used
    # by the default replicate protocol; normal approximation is only used
    # beyond the table.
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
    if degrees_of_freedom <= 0:
        return float("inf")
    if degrees_of_freedom in values:
        return values[degrees_of_freedom]
    larger = min((key for key in values if key > degrees_of_freedom), default=30)
    smaller = max((key for key in values if key < degrees_of_freedom), default=1)
    if larger == smaller:
        return values[smaller]
    weight = (degrees_of_freedom - smaller) / (larger - smaller)
    return values[smaller] + weight * (values[larger] - values[smaller])


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible scientific data."""

    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return stable_hash(config)


def split_manifest(
    *,
    dataset_hash: str,
    split_seed: int,
    train_instance_ids: Iterable[Any],
    validation_instance_ids: Iterable[Any],
    holdout_instance_ids: Iterable[Any],
    lockbox_instance_ids: Iterable[Any],
    fractions: dict[str, float],
) -> dict[str, Any]:
    """Create the immutable identity of a comparison split."""

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_hash": dataset_hash,
        "split_seed": int(split_seed),
        "train_instance_ids": sorted(map(str, train_instance_ids)),
        "validation_instance_ids": sorted(map(str, validation_instance_ids)),
        "holdout_instance_ids": sorted(map(str, holdout_instance_ids)),
        "lockbox_instance_ids": sorted(map(str, lockbox_instance_ids)),
        "fractions": {str(key): float(value) for key, value in fractions.items()},
    }
    return {"id": f"split_{stable_hash(payload)[:20]}", "hash": stable_hash(payload), **payload}


def current_code_revision(root: str | Path = ".") -> str | None:
    """Return the checkout revision used for an experiment, when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    source_stage: str,
    source_experiment: str | None = None,
    model_family: str = "unknown",
    architecture_hash: str = "unknown",
    dataset_hash: str = "unknown",
    preprocessing_hash: str = "unknown",
    resolved_config: dict[str, Any] | None = None,
    code_revision: str | None = None,
    format_version: str = "unknown",
    compatibility_status: str = "verified",
) -> dict[str, Any]:
    """Create the immutable identity record for a reused checkpoint."""

    path = Path(checkpoint_path)
    return {
        "checkpoint_id": f"checkpoint_{stable_hash({'path': str(path), 'hash': file_hash(path)})[:16]}",
        "checkpoint_path": str(path),
        "checkpoint_hash": file_hash(path),
        "source_stage": source_stage,
        "source_experiment": source_experiment,
        "model_family": model_family,
        "architecture_hash": architecture_hash,
        "dataset_hash": dataset_hash,
        "preprocessing_hash": preprocessing_hash,
        "resolved_config_hash": config_hash(resolved_config or {}),
        "code_revision": code_revision,
        "format_version": format_version,
        "compatibility_status": compatibility_status,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def actual_resource_usage(
    *,
    started_at: str | None,
    completed_at: str | None,
    gpu_count: float = 1.0,
    queue_seconds: float | None = None,
    peak_memory_gb: float | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Derive actual usage from timestamps rather than estimates."""

    actual_gpu_hours: float | None = None
    wall_seconds: float | None = None
    if started_at and completed_at:
        try:
            start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
            wall_seconds = max(0.0, (end - start).total_seconds())
            actual_gpu_hours = wall_seconds / 3600.0 * max(0.0, float(gpu_count))
        except (TypeError, ValueError):
            pass
    return {
        "wall_seconds": wall_seconds,
        "actual_gpu_hours": actual_gpu_hours,
        "gpu_count": gpu_count,
        "queue_seconds": queue_seconds,
        "peak_memory_gb": peak_memory_gb,
        "failure_reason": failure_reason,
        "protocol_version": PROTOCOL_VERSION,
    }


def checkpoint_signature(
    *,
    checkpoint_path: str | Path,
    model_family: str,
    architecture_hash: str,
    dataset_hash: str,
    preprocessing_hash: str,
    code_revision: str,
    format_version: str = "unknown",
) -> dict[str, Any]:
    return {
        "checkpoint_hash": file_hash(checkpoint_path),
        "model_family": model_family,
        "architecture_hash": architecture_hash,
        "dataset_hash": dataset_hash,
        "preprocessing_hash": preprocessing_hash,
        "code_revision": code_revision,
        "format_version": format_version,
    }


def checkpoint_compatible(
    source: dict[str, Any],
    required: dict[str, Any],
    *,
    allowed_code_revisions: Iterable[str] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Compare the scientific identity of a checkpoint, not only its path."""

    reasons: list[str] = []
    for field in ("model_family", "architecture_hash", "dataset_hash", "preprocessing_hash", "format_version"):
        if source.get(field) != required.get(field):
            reasons.append(f"checkpoint {field} mismatch")
    allowed_revisions = set(str(value) for value in allowed_code_revisions)
    if allowed_revisions and source.get("code_revision") not in allowed_revisions:
        reasons.append("checkpoint code revision is not compatible")
    if not source.get("checkpoint_hash"):
        reasons.append("checkpoint hash is unavailable")
    return not reasons, tuple(reasons)


@dataclass(frozen=True)
class ReplicateAggregate:
    count: int
    values: tuple[float, ...]
    mean: float | None
    standard_deviation: float | None
    confidence_interval_95: tuple[float, float] | None
    minimum: float | None
    maximum: float | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "values": list(self.values),
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "confidence_interval_95": list(self.confidence_interval_95) if self.confidence_interval_95 else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "status": self.status,
        }


def aggregate_replicates(values: Iterable[float], *, minimum_replicates: int = 3) -> ReplicateAggregate:
    finite = tuple(float(value) for value in values if value is not None and math.isfinite(float(value)))
    count = len(finite)
    if not finite:
        return ReplicateAggregate(count, finite, None, None, None, None, None, "no_data")
    average = mean(finite)
    deviation = stdev(finite) if count > 1 else None
    interval = None
    if deviation is not None:
        margin = _t_critical_975(count - 1) * deviation / sqrt(count)
        interval = (average - margin, average + margin)
    status = "replicate_complete" if count >= minimum_replicates else "insufficient_replicates"
    return ReplicateAggregate(count, finite, average, deviation, interval, min(finite), max(finite), status)


def compare_paired(
    candidate_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    minimum_effect: float,
    alpha: float = 0.05,
    minimum_replicates: int = 3,
) -> dict[str, Any]:
    """Evaluate a preregistered paired candidate-vs-baseline contrast."""

    candidate = [float(value) for value in candidate_values if math.isfinite(float(value))]
    baseline = [float(value) for value in baseline_values if math.isfinite(float(value))]
    if len(candidate) != len(baseline) or len(candidate) < minimum_replicates:
        return {"status": "insufficient_replicates", "count": min(len(candidate), len(baseline)), "minimum_effect": minimum_effect}
    differences = [left - right for left, right in zip(candidate, baseline)]
    average = mean(differences)
    deviation = stdev(differences) if len(differences) > 1 else None
    if deviation is None:
        interval = None
    else:
        if abs(float(alpha) - 0.05) > 1e-12:
            raise ValueError("paired comparison currently supports alpha=0.05 only")
        margin = _t_critical_975(len(differences) - 1) * deviation / sqrt(len(differences))
        interval = (average - margin, average + margin)
    lower = interval[0] if interval else average
    status = "replicate_supported" if lower >= float(minimum_effect) else "statistically_unresolved"
    return {
        "status": status,
        "count": len(differences),
        "differences": differences,
        "mean_difference": average,
        "standard_deviation": deviation,
        "confidence_interval_95": interval,
        "minimum_effect": float(minimum_effect),
        "alpha": float(alpha),
    }


def lockbox_confirmation(
    *,
    candidate_id: str,
    baseline_values: Iterable[float],
    candidate_values: Iterable[float],
    minimum_effect: float,
    guardrail_values: Iterable[float] | None = None,
    guardrail_minimum: float | None = None,
    minimum_replicates: int = 3,
) -> dict[str, Any]:
    """Evaluate a frozen candidate using values from the protected split."""

    comparison = compare_paired(
        candidate_values,
        baseline_values,
        minimum_effect=minimum_effect,
        minimum_replicates=minimum_replicates,
    )
    guardrail = None
    if guardrail_values is not None and guardrail_minimum is not None:
        guardrail = aggregate_guardrail(
            guardrail_values,
            minimum=guardrail_minimum,
            minimum_replicates=minimum_replicates,
        )
    confirmed = comparison.get("status") == "replicate_supported" and (
        guardrail is None or guardrail["all_replicates_pass"]
    )
    return {
        "id": f"lockbox_{stable_hash({'candidate_id': candidate_id, 'comparison': comparison, 'guardrail': guardrail})[:20]}",
        "candidate_id": candidate_id,
        "status": "lockbox_confirmed" if confirmed else "lockbox_unresolved",
        "comparison": comparison,
        "guardrail": guardrail,
        "protected": True,
        "provenance": {"created_by": "lockbox_evaluator", "protocol_version": PROTOCOL_VERSION},
    }


def aggregate_guardrail(values: Iterable[float], *, minimum: float, minimum_replicates: int = 3) -> dict[str, Any]:
    aggregate = aggregate_replicates(values, minimum_replicates=minimum_replicates)
    finite = list(aggregate.values)
    return {
        "metric": "guardrail",
        "threshold": minimum,
        "all_replicates_pass": bool(finite) and len(finite) >= minimum_replicates and all(value >= minimum for value in finite),
        "minimum_observed": min(finite) if finite else None,
        "aggregate": aggregate.as_dict(),
    }


def resource_record(
    *,
    started_at: str | None,
    completed_at: str | None,
    reserved_gpu_hours: float | None,
    actual_gpu_hours: float | None,
    queue_seconds: float | None = None,
    peak_memory_gb: float | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "started_at": started_at,
        "completed_at": completed_at,
        "reserved_gpu_hours": reserved_gpu_hours,
        "actual_gpu_hours": actual_gpu_hours,
        "queue_seconds": queue_seconds,
        "peak_memory_gb": peak_memory_gb,
        "failure_reason": failure_reason,
    }


def environment_metadata() -> dict[str, Any]:
    """Return reproducibility metadata without probing or mutating training."""

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "protocol_version": PROTOCOL_VERSION,
    }
