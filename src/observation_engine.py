"""Deterministic extraction of scientific observations from run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

from src.experiment_protocol import stable_hash


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
    detector_version: str = "observation-v2"
    live_monitor_enabled: bool = True
    live_trigger_types: tuple[str, ...] = ()
    live_min_persistent_polls: int = 2
    live_llm_cooldown_seconds: float = 3600.0
    live_termination_force_after: float = 5.0
    plateau_window: int = 5
    plateau_min_delta: float = 0.002
    generalisation_gap_threshold: float = 0.10
    instability_zscore: float = 3.0
    regression_tolerance: float = 0.05
    runtime_anomaly_factor: float = 2.0
    metric_specs: dict[str, Any] | None = None
    minimum_epochs_before_termination: int = 0
    insufficient_data_is_trigger: bool = False
    primary_metric: str = "compound.macro_f1"
    required_primary_metric: bool = True
    guardrail_metric: str = "action.accuracy"
    guardrail_minimum: float = 0.30
    slope_confidence_z: float = 1.96


def _metric_spec(config: DetectorConfig, metric: str) -> dict[str, Any] | None:
    specs = config.metric_specs or {}
    if metric in specs and isinstance(specs[metric], dict):
        return specs[metric]
    if specs:
        return None
    return {"direction": "minimize" if "loss" in metric.lower() else "maximize"}


def _observation(
    observation_type: str,
    experiment_id: str,
    statement: str,
    measurements: dict[str, Any],
    *,
    rule: str,
    threshold: Any,
    reliability: float = 0.9,
    detector_version: str = "observation-v2",
    now: str | None = None,
) -> dict[str, Any]:
    timestamp = now or _now()
    identity = {
        "detector_version": detector_version,
        "type": observation_type,
        "experiment": experiment_id,
        "rule": rule,
        "threshold": threshold,
        "measurements": measurements,
    }
    return {
        "id": f"obs_{stable_hash(identity)[:16]}",
        "type": observation_type,
        "source_experiments": [experiment_id],
        "statement": statement,
        "measurements": measurements,
        "detection": {"method": "deterministic_rule", "rule": rule, "threshold": threshold, "version": detector_version},
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
        spec = _metric_spec(config, metric)
        if spec is None or spec.get("role") in {"control", "auxiliary"}:
            continue
        values = [row[metric] for row in window if metric in row and math.isfinite(row[metric])]
        valid_range = spec.get("valid_range")
        if isinstance(valid_range, (list, tuple)) and len(valid_range) == 2:
            values = [value for value in values if float(valid_range[0]) <= value <= float(valid_range[1])]
        if len(values) < max(3, config.plateau_window - 1):
            continue
        spread = max(values) - min(values)
        reference = max(abs(mean(values)), 1e-12)
        relative_spread = spread / reference
        threshold = float(spec.get("plateau_threshold", config.plateau_min_delta))
        use_relative = bool(spec.get("relative_plateau", False))
        plateau = relative_spread <= threshold if use_relative else spread <= threshold
        if plateau:
            observations.append(
                _observation(
                    "loss_plateau" if "loss" in metric.lower() else "validation_plateau",
                    experiment_id,
                    f"{metric} changed by at most {spread:.6g} over the last {len(values)} recorded points.",
                    {"metric": metric, "window": len(values), "spread": spread, "last_value": values[-1]},
                    rule="metric_plateau",
                    threshold=threshold,
                    detector_version=config.detector_version,
                    now=now,
                )
            )
    return observations


def detect_trajectory_slope(
    rows: list[dict[str, float]],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    """Estimate a robust-enough linear trend with a small-sample uncertainty bound."""

    observations: list[dict[str, Any]] = []
    minimum = max(4, int(config.plateau_window))
    if len(rows) < minimum:
        return observations
    for metric in sorted(set().union(*(row.keys() for row in rows))):
        spec = _metric_spec(config, metric)
        if spec is None or spec.get("role") in {"control", "auxiliary"}:
            continue
        values = [row[metric] for row in rows[-minimum:] if metric in row and math.isfinite(row[metric])]
        if len(values) < minimum - 1:
            continue
        x = list(range(len(values)))
        x_mean = mean(x)
        y_mean = mean(values)
        denominator = sum((item - x_mean) ** 2 for item in x)
        slope = sum((item - x_mean) * (value - y_mean) for item, value in zip(x, values)) / denominator
        residuals = [value - (y_mean + slope * (item - x_mean)) for item, value in zip(x, values)]
        residual_std = math.sqrt(sum(value * value for value in residuals) / max(1, len(values) - 2))
        slope_se = residual_std / math.sqrt(denominator) if denominator else float("inf")
        direction = str(spec.get("direction", "minimize"))
        effective_slope = slope if direction == "maximize" else -slope
        lower_bound = effective_slope - float(config.slope_confidence_z) * slope_se
        if lower_bound >= 0:
            trend = "improving"
        elif effective_slope + float(config.slope_confidence_z) * slope_se <= 0:
            trend = "worsening"
        else:
            trend = "uncertain"
        observations.append(
            _observation(
                "trajectory_slope",
                experiment_id,
                f"{metric} has an estimated slope of {slope:.6g} over the last {len(values)} points; direction is {trend}.",
                {"metric": metric, "slope": slope, "slope_standard_error": slope_se, "effective_slope": effective_slope, "window": len(values), "trend": trend},
                rule="linear_slope_with_confidence_bound",
                threshold=config.slope_confidence_z,
                reliability=0.8 if trend != "uncertain" else 0.6,
                detector_version=config.detector_version,
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
        spec = _metric_spec(config, key)
        if spec is None:
            continue
        direction = str(spec.get("direction", "minimize"))
        signed_gap = validation_value - train_value if direction == "minimize" else train_value - validation_value
        gap = max(0.0, signed_gap)
        scale = max(abs(train_value), abs(validation_value), 1e-12)
        relative_gap = gap / scale
        threshold = float(spec.get("gap_threshold", config.generalisation_gap_threshold))
        use_relative = bool(spec.get("relative_gap", True))
        exceeds = relative_gap >= threshold if use_relative else gap >= threshold
        if exceeds:
            observations.append(
                _observation(
                    "generalisation_gap",
                    experiment_id,
                    f"{key} and {validation_key} differ by {gap:.6g} at the final recorded point.",
                    {"train_metric": key, "validation_metric": validation_key, "train": train_value, "validation": validation_value, "gap": gap, "relative_gap": relative_gap, "direction": direction},
                    rule="generalisation_gap_above_threshold",
                    threshold=threshold,
                    detector_version=config.detector_version,
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
        spec = _metric_spec(config, metric)
        if spec is None or spec.get("role") in {"control", "auxiliary"}:
            continue
        values = [row[metric] for row in rows if metric in row and math.isfinite(row[metric])]
        if len(values) < 4:
            continue
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        prior_deltas = deltas[:-1]
        if len(prior_deltas) < 2:
            continue
        scale = pstdev(prior_deltas)
        if scale == 0:
            continue
        tail = abs(deltas[-1] - mean(prior_deltas))
        if tail >= config.instability_zscore * scale:
            observations.append(
                _observation(
                    "unstable_metric",
                    experiment_id,
                    f"The final change in {metric} is unusually large relative to its prior trajectory.",
                    {"metric": metric, "final_delta": deltas[-1], "prior_delta_std": scale},
                    rule="trajectory_delta_outlier",
                    threshold=config.instability_zscore,
                    detector_version=config.detector_version,
                    now=now,
                )
            )
    return observations


def detect_regression(
    score: float | None,
    comparable_scores: Iterable[float],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    direction: str = "maximize",
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    scores = [float(value) for value in comparable_scores]
    if score is None or not scores:
        return []
    reference = median(scores)
    if direction == "minimize":
        regressed = score > reference * (1.0 + config.regression_tolerance)
    else:
        regressed = score < reference * (1.0 - config.regression_tolerance)
    if not regressed:
        return []
    return [
        _observation(
            "regression_against_baseline",
            experiment_id,
            f"The score {score:.6g} is worse than the comparable-trial median {reference:.6g} for a {direction} metric.",
            {"score": score, "comparable_median": reference, "comparable_count": len(scores), "direction": direction},
            rule="score_below_comparable_median",
            threshold=config.regression_tolerance,
            now=now,
        )
    ]


def detect_insufficient_data(
    rows: list[dict[str, float]],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    """Report missing monitoring evidence explicitly instead of treating it as success."""

    required = max(1, int(config.plateau_window))
    if len(rows) >= required:
        return []
    return [
        _observation(
            "insufficient_data",
            experiment_id,
            f"Only {len(rows)} history points are available; at least {required} are required for trajectory detectors.",
            {"observed_points": len(rows), "required_points": required},
            rule="minimum_history_window",
            threshold=required,
            reliability=1.0 if not rows else 0.95,
            now=now,
        )
    ] if config.insufficient_data_is_trigger else []


def detect_seed_sensitivity(
    scores: Iterable[float],
    experiment_id: str,
    *,
    threshold: float = 0.05,
    now: str | None = None,
) -> list[dict[str, Any]]:
    values = [float(value) for value in scores if value is not None and math.isfinite(float(value))]
    if len(values) < 2 or max(values) - min(values) <= threshold:
        return []
    return [_observation(
        "seed_sensitivity",
        experiment_id,
        f"Replicate scores span {max(values) - min(values):.6g}, exceeding the configured seed-sensitivity threshold.",
        {"scores": values, "spread": max(values) - min(values)},
        rule="replicate_score_spread",
        threshold=threshold,
        now=now,
    )]


def detect_trajectory_similarity(
    rows_a: list[dict[str, float]],
    rows_b: list[dict[str, float]],
    experiment_id: str,
    *,
    metric: str = "val_loss",
    threshold: float = 0.98,
    now: str | None = None,
) -> list[dict[str, Any]]:
    a = [row[metric] for row in rows_a if metric in row and math.isfinite(row[metric])]
    b = [row[metric] for row in rows_b if metric in row and math.isfinite(row[metric])]
    n = min(len(a), len(b))
    if n < 3:
        return []
    a, b = a[-n:], b[-n:]
    mean_a, mean_b = mean(a), mean(b)
    denom_a = math.sqrt(sum((value - mean_a) ** 2 for value in a))
    denom_b = math.sqrt(sum((value - mean_b) ** 2 for value in b))
    correlation = 1.0 if denom_a == 0 and denom_b == 0 else 0.0 if denom_a == 0 or denom_b == 0 else sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (denom_a * denom_b)
    if correlation < threshold:
        return []
    return [_observation(
        "trajectory_similarity",
        experiment_id,
        f"The {metric} trajectory is highly similar to the comparable trajectory (correlation {correlation:.6g}).",
        {"metric": metric, "correlation": correlation, "points": n},
        rule="trajectory_correlation_above_threshold",
        threshold=threshold,
        now=now,
    )]


def detect_dominance(
    score: float | None,
    comparable_scores: Iterable[float],
    experiment_id: str,
    *,
    margin: float = 0.0,
    now: str | None = None,
) -> list[dict[str, Any]]:
    values = [float(value) for value in comparable_scores if value is not None and math.isfinite(float(value))]
    if score is None or not values or float(score) <= max(values) + margin:
        return []
    return [_observation(
        "dominates_comparable_trials",
        experiment_id,
        f"The score {float(score):.6g} exceeds every comparable trial by at least {margin:.6g}.",
        {"score": float(score), "comparable_maximum": max(values), "comparable_count": len(values)},
        rule="score_above_comparable_maximum",
        threshold=margin,
        now=now,
    )]


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


def detect_primary_metric_availability(
    summary_metrics: dict[str, float],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    if not config.required_primary_metric or config.primary_metric in summary_metrics:
        return []
    return [_observation(
        "primary_metric_unavailable",
        experiment_id,
        f"The registered primary metric {config.primary_metric} is not present in the stage summary.",
        {"metric": config.primary_metric, "available_metrics": sorted(summary_metrics)},
        rule="registered_primary_metric_missing",
        threshold="required",
        reliability=1.0,
        now=now,
    )]


def detect_guardrail_status(
    summary_metrics: dict[str, float],
    experiment_id: str,
    config: DetectorConfig = DetectorConfig(),
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    if config.guardrail_metric not in summary_metrics:
        return []
    value = summary_metrics[config.guardrail_metric]
    if value >= float(config.guardrail_minimum):
        return []
    return [_observation(
        "guardrail_failure",
        experiment_id,
        f"Guardrail {config.guardrail_metric} is {value:.6g}, below the minimum {config.guardrail_minimum:.6g}.",
        {"metric": config.guardrail_metric, "value": value, "minimum": config.guardrail_minimum},
        rule="guardrail_below_minimum",
        threshold=config.guardrail_minimum,
        reliability=1.0,
        now=now,
    )]


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
    replicate_scores: Iterable[float] = (),
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
    observations.extend(detect_insufficient_data(rows, experiment_id, config, now=now))
    observations.extend(detect_plateaus(rows, experiment_id, config, now=now))
    observations.extend(detect_generalisation_gaps(rows, experiment_id, config, now=now))
    observations.extend(detect_instability(rows, experiment_id, config, now=now))
    primary_spec = _metric_spec(config, "compound.macro_f1") or {}
    observations.extend(detect_regression(score, comparable_scores, experiment_id, config, direction=str(primary_spec.get("direction", "maximize")), now=now))
    observations.extend(detect_seed_sensitivity(replicate_scores, experiment_id, now=now))
    observations.extend(detect_dominance(score, comparable_scores, experiment_id, now=now))
    observations.extend(detect_runtime_anomaly(runtime_hours, comparable_runtime_hours, experiment_id, config, now=now))
    if summary_metrics_path:
        metrics = read_summary_metrics(summary_metrics_path)
        observations.extend(detect_primary_metric_availability(metrics, experiment_id, config, now=now))
        observations.extend(detect_guardrail_status(metrics, experiment_id, config, now=now))
        for metric, value in metrics.items():
            if not math.isfinite(value):
                observations.extend(detect_non_finite([{metric: value}], experiment_id, now=now))
    observations.extend(detect_trajectory_slope(rows, experiment_id, config, now=now))
    return observations
