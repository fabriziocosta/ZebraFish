"""Deterministic extraction of scientific observations from run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def read_history(path: str | Path) -> list[dict[str, float]]:
    target = Path(path)
    rows: list[dict[str, float]] = []
    if not target.exists():
        return rows
    with target.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {key: value for key, value in ((_key, _number(_value)) for _key, _value in raw.items()) if value is not None}
            if row:
                rows.append(row)
    return rows


def read_summary_metrics(path: str | Path) -> dict[str, float]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    if {"target", "metric", "value"}.issubset(rows[0]):
        return {
            f"{row['target']}.{row['metric']}": value
            for row in rows
            if (value := _number(row.get("value"))) is not None
        }
    if {"metric", "value"}.issubset(rows[0]):
        return {
            str(row["metric"]): value
            for row in rows
            if (value := _number(row.get("value"))) is not None
        }
    return {}


@dataclass(frozen=True)
class DetectorConfig:
    plateau_window: int = 5
    plateau_min_delta: float = 0.002
    generalisation_gap_threshold: float = 0.10
    instability_zscore: float = 3.0
    regression_tolerance: float = 0.05
    runtime_anomaly_factor: float = 2.0


def _observation(
    observation_type: str,
    experiment_id: str,
    statement: str,
    measurements: dict[str, Any],
    *,
    rule: str,
    threshold: Any,
    reliability: float = 0.9,
    now: str | None = None,
) -> dict[str, Any]:
    timestamp = now or _now()
    return {
        "id": f"obs_{uuid.uuid4().hex[:12]}",
        "type": observation_type,
        "source_experiments": [experiment_id],
        "statement": statement,
        "measurements": measurements,
        "detection": {"method": "deterministic_rule", "rule": rule, "threshold": threshold},
        "reliability": reliability,
        "created_at": timestamp,
        "provenance": {"created_by": "observation_engine", "created_at": timestamp},
    }


def detect_non_finite(rows: Iterable[dict[str, Any]], experiment_id: str, *, now: str | None = None) -> list[dict[str, Any]]:
    bad: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                bad.append(key)
    if not bad:
        return []
    keys = sorted(set(bad))
    return [
        _observation(
            "non_finite_metric",
            experiment_id,
            f"Non-finite values were detected in {', '.join(keys)}.",
            {"metrics": keys, "count": len(bad)},
            rule="non_finite_value",
            threshold="finite",
            reliability=1.0,
            now=now,
        )
    ]


def detect_plateaus(
    rows: list[dict[str, float]],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    if len(rows) < max(3, config.plateau_window):
        return []
    observations: list[dict[str, Any]] = []
    window = rows[-config.plateau_window :]
    for metric in sorted(set().union(*(row.keys() for row in rows))):
        values = [row[metric] for row in window if metric in row and math.isfinite(row[metric])]
        if len(values) < max(3, config.plateau_window - 1):
            continue
        spread = max(values) - min(values)
        if spread <= config.plateau_min_delta:
            observations.append(
                _observation(
                    "loss_plateau" if "loss" in metric.lower() else "validation_plateau",
                    experiment_id,
                    f"{metric} changed by at most {spread:.6g} over the last {len(values)} recorded points.",
                    {"metric": metric, "window": len(values), "spread": spread, "last_value": values[-1]},
                    rule="metric_plateau",
                    threshold=config.plateau_min_delta,
                    now=now,
                )
            )
    return observations


def detect_generalisation_gaps(
    rows: list[dict[str, float]],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    last = rows[-1]
    observations: list[dict[str, Any]] = []
    for key, train_value in last.items():
        if not key.startswith("train_"):
            continue
        validation_key = "val_" + key[len("train_") :]
        validation_value = last.get(validation_key)
        if validation_value is None:
            continue
        gap = abs(train_value - validation_value)
        if gap >= config.generalisation_gap_threshold:
            observations.append(
                _observation(
                    "generalisation_gap",
                    experiment_id,
                    f"{key} and {validation_key} differ by {gap:.6g} at the final recorded point.",
                    {"train_metric": key, "validation_metric": validation_key, "train": train_value, "validation": validation_value, "gap": gap},
                    rule="generalisation_gap_above_threshold",
                    threshold=config.generalisation_gap_threshold,
                    now=now,
                )
            )
    return observations


def detect_instability(
    rows: list[dict[str, float]],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    if len(rows) < 4:
        return observations
    for metric in sorted(set().union(*(row.keys() for row in rows))):
        values = [row[metric] for row in rows if metric in row and math.isfinite(row[metric])]
        if len(values) < 4:
            continue
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        scale = pstdev(deltas)
        if scale == 0:
            continue
        tail = abs(deltas[-1] - mean(deltas[:-1]))
        if tail >= config.instability_zscore * scale:
            observations.append(
                _observation(
                    "unstable_metric",
                    experiment_id,
                    f"The final change in {metric} is unusually large relative to its prior trajectory.",
                    {"metric": metric, "final_delta": deltas[-1], "prior_delta_std": scale},
                    rule="trajectory_delta_outlier",
                    threshold=config.instability_zscore,
                    now=now,
                )
            )
    return observations


def detect_regression(
    score: float | None,
    comparable_scores: Iterable[float],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    scores = [float(value) for value in comparable_scores]
    if score is None or not scores:
        return []
    reference = median(scores)
    if score >= reference * (1.0 - config.regression_tolerance):
        return []
    return [
        _observation(
            "regression_against_baseline",
            experiment_id,
            f"The score {score:.6g} is below the comparable-trial median {reference:.6g}.",
            {"score": score, "comparable_median": reference, "comparable_count": len(scores)},
            rule="score_below_comparable_median",
            threshold=config.regression_tolerance,
            now=now,
        )
    ]


def detect_runtime_anomaly(
    runtime_hours: float | None,
    comparable_runtime_hours: Iterable[float],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    runtimes = [float(value) for value in comparable_runtime_hours if float(value) > 0]
    if runtime_hours is None or not runtimes:
        return []
    reference = median(runtimes)
    if runtime_hours <= reference * config.runtime_anomaly_factor:
        return []
    return [
        _observation(
            "anomalous_runtime",
            experiment_id,
            f"Runtime {runtime_hours:.3g} hours is unusually high compared with the median {reference:.3g} hours.",
            {"runtime_hours": runtime_hours, "comparable_median_hours": reference},
            rule="runtime_above_comparable_median",
            threshold=config.runtime_anomaly_factor,
            now=now,
        )
    ]


def generate_observations(
    experiment_id: str,
    *,
    run_dir: str | Path | None = None,
    history_path: str | Path | None = None,
    summary_metrics_path: str | Path | None = None,
    score: float | None = None,
    comparable_scores: Iterable[float] = (),
    runtime_hours: float | None = None,
    comparable_runtime_hours: Iterable[float] = (),
    config: DetectorConfig = DetectorConfig(),
    now: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(run_dir) if run_dir else None
    if history_path is None and root is not None:
        candidates = sorted(root.rglob("*history*.csv"), key=lambda path: path.stat().st_mtime)
        history_path = candidates[-1] if candidates else None
    if summary_metrics_path is None and root is not None:
        candidates = sorted(root.rglob("*summary_metrics*.csv"), key=lambda path: path.stat().st_mtime)
        summary_metrics_path = candidates[-1] if candidates else None
    rows = read_history(history_path) if history_path else []
    observations: list[dict[str, Any]] = []
    observations.extend(detect_non_finite(rows, experiment_id, now=now))
    observations.extend(detect_plateaus(rows, experiment_id, config, now=now))
    observations.extend(detect_generalisation_gaps(rows, experiment_id, config, now=now))
    observations.extend(detect_instability(rows, experiment_id, config, now=now))
    observations.extend(detect_regression(score, comparable_scores, experiment_id, config, now=now))
    observations.extend(detect_runtime_anomaly(runtime_hours, comparable_runtime_hours, experiment_id, config, now=now))
    if summary_metrics_path:
        metrics = read_summary_metrics(summary_metrics_path)
        for metric, value in metrics.items():
            if not math.isfinite(value):
                observations.extend(detect_non_finite([{metric: value}], experiment_id, now=now))
    return observations
