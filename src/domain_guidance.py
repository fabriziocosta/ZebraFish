"""Deterministic, campaign-specific scientific domain evaluation.

The evaluator deliberately operates on original latent vectors and prediction
tables. Two-dimensional projections are retained as visual artifacts only and
are never accepted as numeric decision evidence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from src.experiment_protocol import compare_paired, file_hash, stable_hash

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


DOMAIN_EVALUATION_VERSION = "domain-guidance-v1"
CONSTRAINT_KINDS = {"classification_guardrail", "pairwise_separability", "related_geometry"}
CONSTRAINT_ROLES = {"hard_guardrail", "secondary_evidence"}
CHECK_DIRECTIONS = {"increase", "decrease", "preserve_or_improve"}


class DomainGuidanceError(ValueError):
    """Raised when a domain contract or evaluation input is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def validate_domain_contract(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DomainGuidanceError("domain contract must be a mapping")
    contract = deepcopy(payload)
    if contract.get("version") != 1:
        raise DomainGuidanceError("domain contract version must be 1")
    if not isinstance(contract.get("id"), str) or not contract["id"]:
        raise DomainGuidanceError("domain contract requires a non-empty id")
    applies_to = contract.get("applies_to")
    if not isinstance(applies_to, dict) or not isinstance(applies_to.get("stages"), list) or not applies_to.get("target"):
        raise DomainGuidanceError("domain contract requires applies_to.stages and applies_to.target")
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, dict):
        raise DomainGuidanceError("domain contract requires evaluation settings")
    for field in ("split", "outer_unit", "inner_unit"):
        if not isinstance(evaluation.get(field), str) or not evaluation[field]:
            raise DomainGuidanceError(f"domain contract evaluation requires {field}")
    if int(evaluation.get("baseline_replicates", 0)) < 3:
        raise DomainGuidanceError("baseline calibration requires at least three replicates")
    constraints = contract.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise DomainGuidanceError("domain contract requires at least one constraint")
    seen: set[str] = set()
    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise DomainGuidanceError("domain constraints must be mappings")
        constraint_id = constraint.get("id")
        if not isinstance(constraint_id, str) or not constraint_id or constraint_id in seen:
            raise DomainGuidanceError("domain constraint ids must be unique non-empty strings")
        seen.add(constraint_id)
        if constraint.get("kind") not in CONSTRAINT_KINDS:
            raise DomainGuidanceError(f"unsupported domain constraint kind: {constraint.get('kind')!r}")
        if constraint.get("role") not in CONSTRAINT_ROLES:
            raise DomainGuidanceError(f"unsupported domain constraint role: {constraint.get('role')!r}")
        labels = constraint.get("labels")
        expected_labels = 2 if constraint["kind"] in {"pairwise_separability", "related_geometry"} else 1
        if not isinstance(labels, list) or len(labels) < expected_labels or len(labels) != len(set(map(str, labels))):
            raise DomainGuidanceError(f"constraint {constraint_id} has invalid labels")
        checks = constraint.get("checks")
        if not isinstance(checks, list) or not checks:
            raise DomainGuidanceError(f"constraint {constraint_id} requires checks")
        metric_names: set[str] = set()
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get("metric"), str):
                raise DomainGuidanceError(f"constraint {constraint_id} has an invalid metric check")
            if check["metric"] in metric_names:
                raise DomainGuidanceError(f"constraint {constraint_id} repeats metric {check['metric']}")
            metric_names.add(check["metric"])
            if check.get("direction") not in CHECK_DIRECTIONS:
                raise DomainGuidanceError(f"constraint {constraint_id} has an invalid direction")
            for number_field in ("minimum_effect", "tolerance", "absolute_minimum", "absolute_maximum"):
                if number_field in check and (
                    not isinstance(check[number_field], (int, float)) or isinstance(check[number_field], bool)
                ):
                    raise DomainGuidanceError(f"constraint {constraint_id} {number_field} must be numeric")
    return contract


def load_domain_contract(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise DomainGuidanceError(f"domain contract does not exist: {target}")
    text = target.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    contract = validate_domain_contract(payload)
    contract["_source_path"] = str(target)
    contract["_hash"] = domain_contract_hash(contract)
    return contract


def domain_contract_hash(contract: dict[str, Any]) -> str:
    canonical = {key: value for key, value in contract.items() if not str(key).startswith("_")}
    return stable_hash(canonical)


def derive_experimental_run_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    path = Path(text)
    parent = path.parent
    return parent.name or path.name or "unknown"


def aligned_evaluation_metadata(
    metadata: pd.DataFrame,
    *,
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    probabilities: np.ndarray | None = None,
    probability_labels: Iterable[str] = (),
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    true_values = np.asarray(list(y_true), dtype=object)
    pred_values = np.asarray(list(y_pred), dtype=object)
    if len(frame) != len(true_values) or len(frame) != len(pred_values):
        raise DomainGuidanceError("metadata, true labels, and predictions must be row-aligned")
    frame["true_name"] = true_values.astype(str)
    frame["pred_name"] = pred_values.astype(str)
    if "experimental_run_id" not in frame:
        source = frame["image_condition_dir"] if "image_condition_dir" in frame else pd.Series([""] * len(frame))
        frame["experimental_run_id"] = source.map(derive_experimental_run_id)
    if probabilities is not None:
        matrix = np.asarray(probabilities, dtype=float)
        labels = [str(label) for label in probability_labels]
        if matrix.ndim != 2 or matrix.shape[0] != len(frame) or matrix.shape[1] != len(labels):
            raise DomainGuidanceError("probability matrix is not aligned with metadata and labels")
        for index, label in enumerate(labels):
            safe = "".join(character if character.isalnum() else "_" for character in label).strip("_")
            frame[f"proba_{safe}"] = matrix[:, index]
    return frame


def _standardize(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise DomainGuidanceError("latent features must be a two-dimensional matrix")
    center = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    standardized = (matrix - center) / scale
    if not np.isfinite(standardized).all():
        raise DomainGuidanceError("latent features contain non-finite values")
    return standardized


def _balanced_accuracy(true_values: np.ndarray, pred_values: np.ndarray, labels: list[str]) -> float | None:
    recalls = []
    for label in labels:
        mask = true_values == label
        if not mask.any():
            return None
        recalls.append(float(np.mean(pred_values[mask] == label)))
    return float(np.mean(recalls)) if recalls else None


def _classification_metrics(true_values: np.ndarray, pred_values: np.ndarray, labels: list[str]) -> dict[str, float | None]:
    supported = [label for label in labels if np.any(true_values == label)]
    if not supported:
        return {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in supported:
        true_positive = int(np.sum((true_values == label) & (pred_values == label)))
        false_negative = int(np.sum((true_values == label) & (pred_values != label)))
        false_positive = int(np.sum((true_values != label) & (pred_values == label)))
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        recalls.append(float(recall))
        f1s.append(float(f1))
    recalls_array = np.asarray(recalls, dtype=float)
    f1_array = np.asarray(f1s, dtype=float)
    predicted_counts = Counter(str(value) for value in pred_values)
    total = max(1, len(pred_values))
    probabilities = np.asarray([count / total for count in predicted_counts.values()], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    result: dict[str, float | None] = {
        "worst_class_recall": float(np.min(recalls_array)),
        "macro_class_recall": float(np.mean(recalls_array)),
        "macro_class_f1": float(np.mean(f1_array)),
        "prediction_coverage": float(len(set(pred_values).intersection(labels)) / len(labels)),
        "prediction_collapse_index": float(max(predicted_counts.values(), default=0) / total),
        "effective_predicted_classes": float(math.exp(entropy)),
    }
    for label, recall, f1 in zip(supported, recalls_array, f1_array):
        result[f"class_recall::{label}"] = float(recall)
        result[f"class_f1::{label}"] = float(f1)
    return result


def _pairwise_prediction_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    labels: list[str],
) -> dict[str, float | None]:
    mask = np.isin(true_values, labels)
    pair_true = true_values[mask]
    pair_pred = pred_values[mask]
    if not all(np.any(pair_true == label) for label in labels):
        return {"pairwise_balanced_accuracy": None, "symmetric_confusion_rate": None}
    confusion = sum(int(np.sum((pair_true == left) & (pair_pred == right))) for left, right in (labels, labels[::-1]))
    return {
        "pairwise_balanced_accuracy": _balanced_accuracy(pair_true, pair_pred, labels),
        "symmetric_confusion_rate": float(confusion / max(1, len(pair_true))),
    }


def _leave_one_compound_out(
    features: np.ndarray,
    true_values: np.ndarray,
    compounds: np.ndarray,
    labels: list[str],
    *,
    minimum_compounds_per_class: int,
) -> float | None:
    mask = np.isin(true_values, labels)
    pair_features = features[mask]
    pair_true = true_values[mask]
    pair_groups = compounds[mask]
    for label in labels:
        if len(set(pair_groups[pair_true == label])) < minimum_compounds_per_class:
            return None
    predictions: list[str] = []
    truths: list[str] = []
    for group in sorted(set(map(str, pair_groups))):
        test = pair_groups.astype(str) == group
        train = ~test
        if not test.any() or not all(np.any(pair_true[train] == label) for label in labels):
            continue
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="liblinear",
        )
        estimator.fit(pair_features[train], pair_true[train])
        predictions.extend(map(str, estimator.predict(pair_features[test])))
        truths.extend(map(str, pair_true[test]))
    if not truths or not all(label in truths for label in labels):
        return None
    return float(balanced_accuracy_score(truths, predictions))


def _centroid_distance(features: np.ndarray, true_values: np.ndarray, labels: list[str]) -> float | None:
    groups = [features[true_values == label] for label in labels]
    if any(len(group) < 2 for group in groups):
        return None
    centroids = [np.mean(group, axis=0) for group in groups]
    distance = float(np.linalg.norm(centroids[0] - centroids[1]))
    radii = [float(np.sqrt(np.mean(np.sum((group - centroid) ** 2, axis=1)))) for group, centroid in zip(groups, centroids)]
    pooled_radius = math.sqrt(sum(radius * radius for radius in radii) / len(radii))
    return distance / max(pooled_radius, 1e-12)


def _distance_rank_score(features: np.ndarray, true_values: np.ndarray, labels: list[str]) -> float | None:
    available = sorted(set(map(str, true_values)))
    if any(label not in available for label in labels) or len(available) < 3:
        return None
    centers = {label: np.mean(features[true_values == label], axis=0) for label in available}
    distances: list[tuple[tuple[str, str], float]] = []
    for index, left in enumerate(available):
        for right in available[index + 1 :]:
            distances.append(((left, right), float(np.linalg.norm(centers[left] - centers[right]))))
    distances.sort(key=lambda item: item[1])
    target = tuple(sorted(labels))
    rank = next((index for index, (pair, _) in enumerate(distances) if tuple(sorted(pair)) == target), None)
    if rank is None:
        return None
    return 1.0 if len(distances) == 1 else float(1.0 - rank / (len(distances) - 1))


def _neighbourhood_pair_affinity(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: list[str],
    *,
    neighbours: int,
) -> float | None:
    mask = np.isin(true_values, labels)
    if int(mask.sum()) < 4:
        return None
    k = min(max(1, int(neighbours)), len(features) - 1)
    nearest = NearestNeighbors(n_neighbors=k + 1).fit(features)
    indices = nearest.kneighbors(features, return_distance=False)[:, 1:]
    scores: list[float] = []
    for row_index in np.flatnonzero(mask):
        own = str(true_values[row_index])
        counterpart = labels[1] if own == labels[0] else labels[0]
        neighbour_labels = true_values[indices[row_index]]
        non_self = neighbour_labels[neighbour_labels != own]
        if len(non_self):
            scores.append(float(np.mean(non_self == counterpart)))
    return float(np.mean(scores)) if scores else None


def _neighbourhood_purity(
    features: np.ndarray,
    true_values: np.ndarray,
    labels: list[str],
    *,
    neighbours: int,
) -> float | None:
    mask = np.isin(true_values, labels)
    if int(mask.sum()) < 4 or len(features) < 2:
        return None
    k = min(max(1, int(neighbours)), len(features) - 1)
    nearest = NearestNeighbors(n_neighbors=k + 1).fit(features)
    indices = nearest.kneighbors(features, return_distance=False)[:, 1:]
    scores = [
        float(np.mean(np.isin(true_values[indices[row_index]], labels)))
        for row_index in np.flatnonzero(mask)
    ]
    return float(np.mean(scores)) if scores else None


def _hierarchical_indices(metadata: pd.DataFrame, evaluation: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    outer = str(evaluation["outer_unit"])
    inner = str(evaluation["inner_unit"])
    if outer not in metadata or inner not in metadata:
        return rng.integers(0, len(metadata), size=len(metadata))
    sampled: list[int] = []
    outer_values = sorted(set(metadata[outer].astype(str)))
    for sampled_outer in rng.choice(outer_values, size=len(outer_values), replace=True):
        outer_rows = metadata.index[metadata[outer].astype(str) == str(sampled_outer)].to_numpy()
        inner_values = sorted(set(metadata.loc[outer_rows, inner].astype(str)))
        for sampled_inner in rng.choice(inner_values, size=len(inner_values), replace=True):
            rows = outer_rows[metadata.loc[outer_rows, inner].astype(str).to_numpy() == str(sampled_inner)]
            if len(rows):
                sampled.extend(map(int, rng.choice(rows, size=len(rows), replace=True)))
    return np.asarray(sampled, dtype=int)


def _bootstrap_interval(
    metadata: pd.DataFrame,
    evaluation: dict[str, Any],
    metric: Callable[[np.ndarray], float | None],
) -> list[float] | None:
    iterations = max(0, int(evaluation.get("bootstrap_iterations", 0)))
    if iterations < 20 or len(metadata) < 4:
        return None
    rng = np.random.default_rng(int(evaluation.get("bootstrap_seed", 1729)))
    values: list[float] = []
    for _ in range(iterations):
        indices = _hierarchical_indices(metadata, evaluation, rng)
        value = metric(indices)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if len(values) < max(10, iterations // 4):
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _metric_record(
    value: float | None,
    *,
    interval: list[float] | None = None,
    status: str | None = None,
    method: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "confidence_interval_95": interval,
        "status": status or ("measured" if value is not None else "unresolved"),
        "method": method,
    }


def _baseline_metric(calibration: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not isinstance(calibration, dict) or calibration.get("status") != "frozen":
        return None
    record = calibration.get("metrics", {}).get(key)
    return record if isinstance(record, dict) else None


def _assess_check(
    metric: dict[str, Any],
    check: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    value = metric.get("value")
    if value is None:
        return {"status": "unresolved", "reason": "metric unavailable", "baseline": baseline}
    if "absolute_minimum" in check and float(value) < float(check["absolute_minimum"]):
        return {
            "status": "fail",
            "reason": f"value is below absolute hard floor {float(check['absolute_minimum']):.3g}",
            "baseline": baseline,
        }
    if "absolute_maximum" in check and float(value) > float(check["absolute_maximum"]):
        return {
            "status": "fail",
            "reason": f"value exceeds absolute hard ceiling {float(check['absolute_maximum']):.3g}",
            "baseline": baseline,
        }
    if baseline is None or baseline.get("mean") is None:
        return {"status": "calibrating", "reason": "frozen baseline profile unavailable", "baseline": None}
    baseline_value = float(baseline["mean"])
    delta = float(value) - baseline_value
    direction = str(check["direction"])
    minimum_effect = float(check.get("minimum_effect", 0.0))
    tolerance = float(check.get("tolerance", 0.0))
    if direction == "increase":
        passed = delta >= minimum_effect
    elif direction == "decrease":
        passed = -delta >= minimum_effect
    else:
        metric_name = str(check["metric"])
        minimize = any(token in metric_name for token in ("distance", "confusion", "collapse"))
        passed = delta <= tolerance if minimize else delta >= -tolerance
    return {
        "status": "pass" if passed else "fail",
        "baseline": baseline,
        "delta": delta,
        "reason": f"baseline-relative {direction} rule",
    }


def _constraint_support(
    frame: pd.DataFrame,
    true_values: np.ndarray,
    labels: list[str],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    outer = str(evaluation["outer_unit"])
    inner = str(evaluation["inner_unit"])
    minimum_compounds = int(evaluation.get("minimum_compounds_per_class", 2))
    minimum_runs = int(evaluation.get("minimum_runs_per_class", 2))
    by_class: dict[str, Any] = {}
    sufficient = outer in frame and inner in frame
    for label in labels:
        rows = frame.loc[true_values == label]
        compounds = int(rows[outer].nunique()) if outer in rows else 0
        runs = int(rows[inner].nunique()) if inner in rows else 0
        label_sufficient = (
            len(rows) > 0
            and compounds >= minimum_compounds
            and runs >= minimum_runs
        )
        sufficient = sufficient and label_sufficient
        by_class[label] = {
            "rows": int(len(rows)),
            "compounds": compounds,
            "experimental_runs": runs,
            "sufficient": label_sufficient,
        }
    return {
        "sufficient": bool(sufficient),
        "minimum_compounds_per_class": minimum_compounds,
        "minimum_runs_per_class": minimum_runs,
        "by_class": by_class,
    }


def evaluate_domain_guidance(
    *,
    experiment_id: str,
    latent_features: np.ndarray,
    metadata: pd.DataFrame,
    y_true: Iterable[str],
    y_pred: Iterable[str],
    contract: dict[str, Any],
    split_hash: str | None = None,
    evaluation_protocol: str = "legacy_single_seed",
    calibration_profile: dict[str, Any] | None = None,
    training_seed: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    contract = validate_domain_contract(contract)
    contract_hash = domain_contract_hash(contract)
    evaluation = contract["evaluation"]
    frame = metadata.reset_index(drop=True).copy()
    true_values = np.asarray(list(map(str, y_true)), dtype=object)
    pred_values = np.asarray(list(map(str, y_pred)), dtype=object)
    features = np.asarray(latent_features, dtype=float)
    if len(frame) != len(true_values) or len(frame) != len(pred_values) or len(frame) != len(features):
        raise DomainGuidanceError("latent features, metadata, and predictions must be row-aligned")
    if "experimental_run_id" not in frame:
        source = frame["image_condition_dir"] if "image_condition_dir" in frame else pd.Series([""] * len(frame))
        frame["experimental_run_id"] = source.map(derive_experimental_run_id)
    controls = set(map(str, evaluation.get("control_labels", [])))
    keep = np.ones(len(frame), dtype=bool)
    if bool(evaluation.get("exclude_controls", True)):
        keep &= ~np.isin(true_values, sorted(controls))
        if "is_control" in frame:
            control_mask = frame["is_control"].map(
                lambda value: (
                    bool(value)
                    if isinstance(value, (bool, np.bool_))
                    else str(value).strip().lower() in {"1", "true", "yes", "y"}
                )
            ).to_numpy(dtype=bool)
            keep &= ~control_mask
    frame = frame.loc[keep].reset_index(drop=True)
    true_values = true_values[keep]
    pred_values = pred_values[keep]
    features = _standardize(features[keep])
    labels = sorted({str(label) for constraint in contract["constraints"] for label in constraint["labels"]})
    class_counts = {label: int(np.sum(true_values == label)) for label in labels}
    metrics: dict[str, dict[str, Any]] = {}
    constraints: list[dict[str, Any]] = []

    classification = _classification_metrics(true_values, pred_values, labels)
    for name, value in classification.items():
        key = f"classification::{name}"
        interval = _bootstrap_interval(
            frame,
            evaluation,
            lambda indices, metric_name=name: _classification_metrics(
                true_values[indices], pred_values[indices], labels
            ).get(metric_name),
        ) if value is not None and not name.startswith("class_") else None
        metrics[key] = _metric_record(value, interval=interval, method="cluster_aware_classification")

    compounds = frame[evaluation["outer_unit"]].astype(str).to_numpy() if evaluation["outer_unit"] in frame else np.arange(len(frame)).astype(str)
    for constraint in contract["constraints"]:
        constraint_labels = list(map(str, constraint["labels"]))
        support = _constraint_support(frame, true_values, constraint_labels, evaluation)
        local: dict[str, dict[str, Any]] = {}
        if constraint["kind"] == "classification_guardrail":
            for check in constraint["checks"]:
                local[check["metric"]] = metrics.get(
                    f"classification::{check['metric']}",
                    _metric_record(None, method="cluster_aware_classification"),
                )
        elif constraint["kind"] == "pairwise_separability":
            pair_metrics = _pairwise_prediction_metrics(true_values, pred_values, constraint_labels)
            pair_metrics["leave_one_compound_out_balanced_accuracy"] = _leave_one_compound_out(
                features,
                true_values,
                compounds,
                constraint_labels,
                minimum_compounds_per_class=int(evaluation.get("minimum_compounds_per_class", 2)),
            )
            for check in constraint["checks"]:
                name = str(check["metric"])
                value = pair_metrics.get(name)
                if name in {"pairwise_balanced_accuracy", "symmetric_confusion_rate"} and value is not None:
                    interval = _bootstrap_interval(
                        frame,
                        evaluation,
                        lambda indices, metric_name=name: _pairwise_prediction_metrics(
                            true_values[indices], pred_values[indices], constraint_labels
                        ).get(metric_name),
                    )
                else:
                    interval = None
                local[name] = _metric_record(
                    value,
                    interval=interval,
                    method="leave_one_compound_out_linear_probe" if name.startswith("leave_one") else "pairwise_confusion",
                )
        else:
            geometry_values = {
                "normalized_centroid_distance": _centroid_distance(features, true_values, constraint_labels),
                "related_distance_rank_score": _distance_rank_score(features, true_values, constraint_labels),
                "neighbourhood_pair_affinity": _neighbourhood_pair_affinity(
                    features,
                    true_values,
                    constraint_labels,
                    neighbours=int(evaluation.get("neighbourhood_k", 7)),
                ),
                "neighbourhood_purity": _neighbourhood_purity(
                    features,
                    true_values,
                    constraint_labels,
                    neighbours=int(evaluation.get("neighbourhood_k", 7)),
                ),
            }
            callbacks: dict[str, Callable[[np.ndarray], float | None]] = {
                "normalized_centroid_distance": lambda indices: _centroid_distance(features[indices], true_values[indices], constraint_labels),
                "related_distance_rank_score": lambda indices: _distance_rank_score(features[indices], true_values[indices], constraint_labels),
                "neighbourhood_pair_affinity": lambda indices: _neighbourhood_pair_affinity(
                    features[indices],
                    true_values[indices],
                    constraint_labels,
                    neighbours=int(evaluation.get("neighbourhood_k", 7)),
                ),
                "neighbourhood_purity": lambda indices: _neighbourhood_purity(
                    features[indices],
                    true_values[indices],
                    constraint_labels,
                    neighbours=int(evaluation.get("neighbourhood_k", 7)),
                ),
            }
            for check in constraint["checks"]:
                name = str(check["metric"])
                value = geometry_values.get(name)
                interval = (
                    _bootstrap_interval(frame, evaluation, callbacks[name])
                    if value is not None and name != "neighbourhood_pair_affinity"
                    else None
                )
                local[name] = _metric_record(value, interval=interval, method="original_standardized_latent_space")

        assessments = []
        for check in constraint["checks"]:
            name = str(check["metric"])
            key = f"{constraint['id']}::{name}"
            metrics[key] = local[name]
            assessment = _assess_check(local[name], check, _baseline_metric(calibration_profile, key))
            if not support["sufficient"]:
                assessment = {
                    **assessment,
                    "status": "unresolved",
                    "reason": "insufficient independent compounds or experimental runs",
                }
            assessments.append(
                {
                    "metric": name,
                    "value": local[name].get("value"),
                    "confidence_interval_95": local[name].get("confidence_interval_95"),
                    "method": local[name].get("method"),
                    **assessment,
                }
            )
        statuses = {item["status"] for item in assessments}
        status = (
            "fail" if "fail" in statuses
            else "unresolved" if "unresolved" in statuses
            else "calibrating" if "calibrating" in statuses
            else "pass"
        )
        constraints.append(
            {
                "id": constraint["id"],
                "title": constraint.get("title", constraint["id"]),
                "kind": constraint["kind"],
                "role": constraint["role"],
                "labels": constraint_labels,
                "support": support,
                "status": status,
                "checks": assessments,
            }
        )

    hard_statuses = [item["status"] for item in constraints if item["role"] == "hard_guardrail"]
    protocol_compliant = str(evaluation_protocol).startswith(("three_seed", "replicate"))
    overall = (
        "legacy_descriptive" if not protocol_compliant
        else "hard_guardrail_failed" if "fail" in hard_statuses
        else "unresolved" if "unresolved" in hard_statuses
        else "calibrating" if "calibrating" in hard_statuses
        else "eligible"
    )
    timestamp = created_at or _now()
    return {
        "id": f"domain_eval_{stable_hash({'experiment': experiment_id, 'contract': contract_hash, 'split': split_hash})[:20]}",
        "version": DOMAIN_EVALUATION_VERSION,
        "experiment_id": experiment_id,
        "contract_id": contract["id"],
        "contract_hash": contract_hash,
        "contract_path": contract.get("_source_path"),
        "split": evaluation["split"],
        "split_hash": split_hash,
        "evaluation_protocol": evaluation_protocol,
        "training_seed": training_seed,
        "objective_eligibility": overall,
        "umap_used_for_decision": False,
        "unit_of_analysis": {
            "outer": evaluation["outer_unit"],
            "inner": evaluation["inner_unit"],
            "frame_rows_are_independent": False,
        },
        "sample_coverage": {
            "rows": len(frame),
            "class_counts": class_counts,
            "compounds": int(frame[evaluation["outer_unit"]].nunique()) if evaluation["outer_unit"] in frame else None,
            "experimental_runs": int(frame[evaluation["inner_unit"]].nunique()) if evaluation["inner_unit"] in frame else None,
        },
        "metrics": metrics,
        "constraints": constraints,
        "calibration_id": calibration_profile.get("id") if isinstance(calibration_profile, dict) else None,
        "created_at": timestamp,
        "provenance": {
            "created_by": "domain_guidance_evaluator",
            "deterministic": True,
            "latent_space": "original_standardized",
            "projection_role": "visualization_only",
        },
    }


def calibrate_domain_baseline(
    reports: Iterable[dict[str, Any]],
    *,
    contract: dict[str, Any],
    candidate_family_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    contract = validate_domain_contract(contract)
    expected = int(contract["evaluation"].get("baseline_replicates", 3))
    compatible = [
        report for report in reports
        if isinstance(report, dict)
        and report.get("contract_hash") == domain_contract_hash(contract)
        and str(report.get("evaluation_protocol", "")).startswith(("three_seed", "replicate"))
    ]
    split_hashes = {report.get("split_hash") for report in compatible}
    if len(compatible) < expected:
        raise DomainGuidanceError(f"baseline calibration requires {expected} protocol-compliant reports")
    if len(split_hashes) != 1 or None in split_hashes:
        raise DomainGuidanceError("baseline calibration reports must share one non-empty split hash")
    selected = sorted(
        compatible,
        key=lambda report: (
            int(report.get("replicate_index", 10**9)),
            int(report.get("training_seed", 10**9))
            if report.get("training_seed") is not None
            else 10**9,
            str(report.get("id", "")),
        ),
    )[:expected]
    training_seeds = [report.get("training_seed") for report in selected]
    if any(seed is None for seed in training_seeds) or len(set(training_seeds)) != expected:
        raise DomainGuidanceError(
            f"baseline calibration requires {expected} distinct recorded training seeds"
        )
    metric_names = set.intersection(*(set(report.get("metrics", {})) for report in selected))
    metrics: dict[str, Any] = {}
    for name in sorted(metric_names):
        values = [
            report["metrics"][name].get("value")
            for report in selected
            if isinstance(report.get("metrics", {}).get(name), dict)
        ]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if len(finite) != expected:
            continue
        average = float(np.mean(finite))
        deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        margin = 4.303 * deviation / math.sqrt(len(finite)) if len(finite) == 3 else 1.96 * deviation / math.sqrt(len(finite))
        metrics[name] = {
            "values": finite,
            "mean": average,
            "standard_deviation": deviation,
            "confidence_interval_95": [average - margin, average + margin],
        }
    payload = {
        "version": DOMAIN_EVALUATION_VERSION,
        "contract_id": contract["id"],
        "contract_hash": domain_contract_hash(contract),
        "candidate_family_id": candidate_family_id,
        "split_hash": next(iter(split_hashes)),
        "baseline_evaluation_ids": [report["id"] for report in selected],
        "training_seeds": [int(seed) for seed in training_seeds],
        "replicate_count": len(selected),
        "metrics": metrics,
        "status": "frozen",
        "created_at": created_at or _now(),
        "provenance": {"created_by": "domain_guidance_calibrator", "deterministic": True},
    }
    payload["id"] = f"domain_calibration_{stable_hash(payload)[:20]}"
    payload["hash"] = stable_hash(payload)
    return payload


def aggregate_domain_evaluations(
    reports: Iterable[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    contract = validate_domain_contract(contract)
    expected = int(contract["evaluation"].get("baseline_replicates", 3))
    ordered = sorted(
        [report for report in reports if isinstance(report, dict)],
        key=lambda report: (
            int(report.get("replicate_index", 10**9)),
            int(report.get("training_seed", 10**9)) if report.get("training_seed") is not None else 10**9,
            str(report.get("id", "")),
        ),
    )
    if len(ordered) < expected:
        return {
            "status": "unresolved",
            "reason": "insufficient domain-evaluation replicates",
            "replicate_count": len(ordered),
            "required_replicates": expected,
            "constraints": [],
            "hard_guardrails_pass": False,
        }
    expected_contract_hash = domain_contract_hash(contract)
    calibration_split_hash = calibration.get("split_hash")
    calibration_seeds = [int(seed) for seed in calibration.get("training_seeds", [])]
    candidate_seeds = [
        int(report["training_seed"])
        for report in ordered[:expected]
        if report.get("training_seed") is not None
    ]
    incompatible = [
        report.get("id")
        for report in ordered[:expected]
        if report.get("contract_hash") != expected_contract_hash
        or report.get("split_hash") != calibration_split_hash
        or not str(report.get("evaluation_protocol", "")).startswith(("three_seed", "replicate"))
    ]
    if (
        calibration.get("status") != "frozen"
        or calibration.get("contract_hash") != expected_contract_hash
        or not calibration_split_hash
        or len(candidate_seeds) != expected
        or candidate_seeds != calibration_seeds
        or incompatible
    ):
        return {
            "status": "unresolved",
            "reason": "domain reports are incompatible with the frozen contract, split, or protocol",
            "replicate_count": min(len(ordered), expected),
            "required_replicates": expected,
            "incompatible_evaluation_ids": incompatible,
            "constraints": [],
            "hard_guardrails_pass": False,
        }
    constraints: list[dict[str, Any]] = []
    for constraint in contract["constraints"]:
        checks = []
        for check in constraint["checks"]:
            key = f"{constraint['id']}::{check['metric']}"
            report_checks = []
            support_sufficient = True
            for report in ordered[:expected]:
                report_constraint = next(
                    (
                        item for item in report.get("constraints", [])
                        if item.get("id") == constraint["id"]
                    ),
                    {},
                )
                support_sufficient = support_sufficient and bool(
                    report_constraint.get("support", {}).get("sufficient", False)
                )
                report_check = next(
                    (
                        item for item in report_constraint.get("checks", [])
                        if item.get("metric") == check["metric"]
                    ),
                    {},
                )
                report_checks.append(report_check)
            absolute_failures = [
                item for item in report_checks
                if item.get("status") == "fail"
                and str(item.get("reason", "")).startswith(("value is below absolute", "value exceeds absolute"))
            ]
            if absolute_failures:
                checks.append({
                    "metric": check["metric"],
                    "status": "fail",
                    "reason": "absolute hard guardrail failed in at least one replicate",
                    "failed_replicates": len(absolute_failures),
                })
                continue
            if not support_sufficient:
                checks.append({
                    "metric": check["metric"],
                    "status": "unresolved",
                    "reason": "at least one replicate lacks sufficient independent compounds or experimental runs",
                })
                continue
            candidate_values = [
                report.get("metrics", {}).get(key, {}).get("value")
                for report in ordered[:expected]
            ]
            baseline_values = calibration.get("metrics", {}).get(key, {}).get("values", [])
            if any(value is None for value in candidate_values) or len(baseline_values) < expected:
                checks.append({
                    "metric": check["metric"],
                    "status": "unresolved",
                    "reason": "candidate or baseline replicate metric is unavailable",
                })
                continue
            direction = str(check["direction"])
            candidate_numeric = [float(value) for value in candidate_values]
            baseline_numeric = [float(value) for value in baseline_values[:expected]]
            if direction == "decrease" or (
                direction == "preserve_or_improve"
                and any(token in str(check["metric"]) for token in ("distance", "confusion", "collapse"))
            ):
                candidate_numeric = [-value for value in candidate_numeric]
                baseline_numeric = [-value for value in baseline_numeric]
            required = (
                -float(check.get("tolerance", 0.0))
                if direction == "preserve_or_improve"
                else float(check.get("minimum_effect", 0.0))
            )
            comparison = compare_paired(
                candidate_numeric,
                baseline_numeric,
                minimum_effect=required,
                minimum_replicates=expected,
            )
            interval = comparison.get("confidence_interval_95")
            if comparison.get("status") == "replicate_supported":
                status = "pass"
            elif interval is not None and float(interval[1]) < required:
                status = "fail"
            else:
                status = "unresolved"
            checks.append({
                "metric": check["metric"],
                "status": status,
                "direction": direction,
                "comparison": comparison,
            })
        statuses = {item["status"] for item in checks}
        constraint_status = "fail" if "fail" in statuses else "unresolved" if "unresolved" in statuses else "pass"
        constraints.append({
            "id": constraint["id"],
            "title": constraint.get("title", constraint["id"]),
            "role": constraint["role"],
            "status": constraint_status,
            "checks": checks,
        })
    hard = [item["status"] for item in constraints if item["role"] == "hard_guardrail"]
    hard_pass = bool(hard) and all(status == "pass" for status in hard)
    return {
        "status": "pass" if hard_pass else "fail" if "fail" in hard else "unresolved",
        "replicate_count": expected,
        "required_replicates": expected,
        "constraints": constraints,
        "hard_guardrails_pass": hard_pass,
        "calibration_id": calibration.get("id"),
        "contract_hash": domain_contract_hash(contract),
        "umap_used_for_decision": False,
    }


def domain_evaluation_observations(report: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for constraint in report.get("constraints", []):
        status = str(constraint.get("status", "unresolved"))
        observation_type = {
            "pass": "domain_constraint_pass",
            "fail": "domain_constraint_fail",
            "calibrating": "domain_constraint_calibrating",
            "unresolved": "domain_constraint_unresolved",
        }.get(status, "domain_constraint_unresolved")
        identity = {
            "evaluation": report.get("id"),
            "constraint": constraint.get("id"),
            "status": status,
        }
        observations.append(
            {
                "id": f"obs_{stable_hash(identity)[:16]}",
                "type": observation_type,
                "source_experiments": [str(report.get("stage_experiment_id") or report.get("experiment_id"))],
                "statement": f"{constraint.get('title', constraint.get('id'))}: {status}.",
                "direction": (
                    "supporting" if status == "pass"
                    else "contradicting" if status == "fail"
                    else "inconclusive"
                ),
                "measurements": {
                    "constraint_id": constraint.get("id"),
                    "role": constraint.get("role"),
                    "labels": constraint.get("labels", []),
                    "checks": constraint.get("checks", []),
                    "domain_evaluation_id": report.get("id"),
                },
                "detection": {
                    "method": "deterministic_domain_contract",
                    "rule": "baseline_relative_constraint",
                    "threshold": report.get("calibration_id") or "calibration_required",
                    "version": DOMAIN_EVALUATION_VERSION,
                },
                "reliability": 1.0 if status in {"pass", "fail"} else 0.8,
                "classification_status": status,
                "created_at": report.get("created_at") or _now(),
                "provenance": {
                    "created_by": "domain_guidance_evaluator",
                    "contract_hash": report.get("contract_hash"),
                    "domain_evaluation_id": report.get("id"),
                },
            }
        )
    return observations


def build_live_domain_diagnostic(
    *,
    experiment_id: str,
    epoch: int,
    latent_features: np.ndarray,
    metadata: pd.DataFrame,
    y_true: Iterable[str],
    y_pred: Iterable[str],
    contract: dict[str, Any],
    split_hash: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    live_contract = deepcopy(validate_domain_contract(contract))
    live_contract["evaluation"]["bootstrap_iterations"] = 0
    live_contract["evaluation"]["split"] = "validation"
    report = evaluate_domain_guidance(
        experiment_id=f"{experiment_id}:validation:epoch:{int(epoch)}",
        latent_features=latent_features,
        metadata=metadata,
        y_true=y_true,
        y_pred=y_pred,
        contract=live_contract,
        split_hash=split_hash,
        evaluation_protocol="live_validation_diagnostic",
        created_at=created_at,
    )
    monitoring = contract.get("evaluation", {}).get("live_monitoring", {})
    classification = report.get("metrics", {})
    coverage = classification.get("classification::prediction_coverage", {}).get("value")
    collapse = classification.get("classification::prediction_collapse_index", {}).get("value")
    worst_recall = classification.get("classification::worst_class_recall", {}).get("value")
    triggers: list[dict[str, Any]] = []
    if coverage is not None and coverage < float(monitoring.get("minimum_prediction_coverage", 0.5)):
        triggers.append({
            "type": "domain_live_class_coverage_failure",
            "metric": "prediction_coverage",
            "value": coverage,
            "threshold": monitoring.get("minimum_prediction_coverage", 0.5),
        })
    if collapse is not None and collapse > float(monitoring.get("maximum_prediction_collapse_index", 0.85)):
        triggers.append({
            "type": "domain_live_prediction_collapse",
            "metric": "prediction_collapse_index",
            "value": collapse,
            "threshold": monitoring.get("maximum_prediction_collapse_index", 0.85),
        })
    if worst_recall is not None and worst_recall < float(monitoring.get("minimum_worst_class_recall", 0.05)):
        triggers.append({
            "type": "domain_live_worst_class_failure",
            "metric": "worst_class_recall",
            "value": worst_recall,
            "threshold": monitoring.get("minimum_worst_class_recall", 0.05),
        })
    report.update(
        {
            "id": f"domain_live_{stable_hash({'experiment': experiment_id, 'epoch': int(epoch), 'contract': domain_contract_hash(contract)})[:20]}",
            "experiment_id": experiment_id,
            "epoch": int(epoch),
            "contract_hash": domain_contract_hash(contract),
            "evaluation_protocol": "live_validation_diagnostic",
            "objective_eligibility": "live_diagnostic_only",
            "live_triggers": triggers,
            "termination_eligible": bool(monitoring.get("termination_eligible", False)),
            "umap_used_for_decision": False,
        }
    )
    return report


def live_domain_observations(
    report: dict[str, Any],
    *,
    stage_experiment_id: str,
) -> list[dict[str, Any]]:
    observations = []
    for trigger in report.get("live_triggers", []):
        identity = {
            "report": report.get("id"),
            "trigger": trigger,
            "stage_experiment_id": stage_experiment_id,
        }
        observations.append(
            {
                "id": f"obs_{stable_hash(identity)[:16]}",
                "type": trigger.get("type"),
                "source_experiments": [stage_experiment_id],
                "statement": (
                    f"Live validation {trigger.get('metric')} is {trigger.get('value'):.3g}; "
                    f"the registered diagnostic threshold is {trigger.get('threshold'):.3g}."
                ),
                "direction": "contradicting",
                "measurements": {
                    **trigger,
                    "epoch": report.get("epoch"),
                    "domain_evaluation_id": report.get("id"),
                    "termination_eligible": report.get("termination_eligible", False),
                },
                "detection": {
                    "method": "deterministic_live_domain_rule",
                    "rule": trigger.get("type"),
                    "threshold": trigger.get("threshold"),
                    "version": DOMAIN_EVALUATION_VERSION,
                },
                "reliability": 0.9,
                "created_at": report.get("created_at") or _now(),
                "provenance": {
                    "created_by": "live_domain_guidance_evaluator",
                    "contract_hash": report.get("contract_hash"),
                    "validation_only": True,
                },
            }
        )
    return observations


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_domain_calibration(path: str | Path, calibration: dict[str, Any]) -> Path:
    if calibration.get("status") != "frozen" or not calibration.get("hash"):
        raise DomainGuidanceError("only a frozen, hashed domain calibration can be persisted")
    target = Path(path)
    _atomic_json(target, calibration)
    return target


def persist_live_domain_diagnostic(
    *,
    run_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, str]:
    root = Path(run_dir) / "domain_guidance" / "live"
    epoch = int(report.get("epoch", 0))
    epoch_path = root / f"epoch_{epoch:04d}.json"
    latest_path = root / "latest.json"
    _atomic_json(epoch_path, report)
    _atomic_json(latest_path, report)
    return {"epoch_path": str(epoch_path), "latest_path": str(latest_path)}


def persist_domain_evaluation(
    *,
    run_dir: str | Path,
    report: dict[str, Any],
    latent_features: np.ndarray,
    aligned_metadata: pd.DataFrame,
    projection_paths: Iterable[str | Path] = (),
    diagnostic_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(run_dir) / "domain_guidance"
    root.mkdir(parents=True, exist_ok=True)
    latent_path = root / "latent_embeddings.npz"
    metadata_path = root / "latent_metadata.csv"
    predictions_path = root / "domain_predictions.csv"
    report_path = root / "domain_evaluation.json"
    with tempfile.NamedTemporaryFile("wb", dir=root, prefix=".latent_embeddings.", suffix=".npz", delete=False) as handle:
        temporary_latent = Path(handle.name)
    try:
        np.savez_compressed(temporary_latent, latent_features=np.asarray(latent_features, dtype=np.float32))
        os.replace(temporary_latent, latent_path)
    finally:
        temporary_latent.unlink(missing_ok=True)
    aligned_metadata.to_csv(metadata_path, index=False)
    prediction_columns = [
        column for column in aligned_metadata.columns
        if column in {"compound", "experimental_run_id", "concentration_band", "dataset_split", "is_control", "true_name", "pred_name"}
        or column.startswith("proba_")
    ]
    aligned_metadata[prediction_columns].to_csv(predictions_path, index=False)
    persisted = deepcopy(report)
    persisted["artifacts"] = {
        "latent_embeddings": {"path": str(latent_path), "hash": file_hash(latent_path)},
        "metadata": {"path": str(metadata_path), "hash": file_hash(metadata_path)},
        "predictions": {"path": str(predictions_path), "hash": file_hash(predictions_path)},
        "visual_projections": [
            {"path": str(path), "hash": file_hash(path), "decision_role": "visualization_only"}
            for path in projection_paths if Path(path).exists()
        ],
        "diagnostics": [
            {"path": str(path), "hash": file_hash(path), "decision_role": "deterministic_evidence"}
            for path in diagnostic_paths if Path(path).exists()
        ],
    }
    _atomic_json(report_path, persisted)
    return {"report": persisted, "report_path": str(report_path), "report_hash": file_hash(report_path)}
