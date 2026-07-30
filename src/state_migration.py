"""Import existing campaign ledgers into the scientific state file."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

from src.observation_engine import generate_observations
from src.campaign_watchdog import inspect_campaign
from src.scientific_state import apply_operations, load_state, merge_nonconflicting_states, record_entity, transactional_update, update_controller_state


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _iter_trial_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    ignored = {"campaign.lock", "campaign.terminate.lock", "campaign_state.json", "trials.csv", "trials.jsonl", "leaderboard.csv"}
    result: list[Path] = []
    for path in root.iterdir():
        if path.is_dir() and path.name not in ignored and ((path / "trial_manifest.json").exists() or (path / "trial_summary.json").exists()):
            result.append(path)
    return sorted(result)


def _read_ledger_rows(root: Path) -> list[dict[str, Any]]:
    """Read legacy CSV/JSONL ledgers without treating them as canonical state."""

    rows: list[dict[str, Any]] = []
    csv_path = root / "trials.csv"
    if csv_path.exists():
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    jsonl_path = root / "trials.jsonl"
    if jsonl_path.exists():
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        except OSError:
            pass
    return rows


def migrate_campaign(
    campaign_config: dict[str, Any],
    *,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(state_path or campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    state = load_state(target)
    imported_trials = 0
    imported_stages = 0
    existing_trials = 0
    existing_stages = 0
    observations_added = 0
    relations_added = 0
    root = Path(campaign_config["artifacts"]["root"])
    campaign_state_path = root / "campaign_state.json"
    live_reconciliation = None
    if campaign_state_path.exists():
        live_reconciliation = inspect_campaign(
            campaign_config,
            campaign_state_path=campaign_state_path,
            stale_after_seconds=float(campaign_config.get("observation", {}).get("watchdog_stale_after_seconds", 180.0)),
        )
    for trial_dir in _iter_trial_dirs(root):
        manifest = _read_json(trial_dir / "trial_manifest.json")
        summary = _read_json(trial_dir / "trial_summary.json")
        trial_id = str(manifest.get("trial_id") or trial_dir.name)
        stages = list(manifest.get("stages") or campaign_config["campaign"]["stages"])
        stage_runs = manifest.get("stage_runs", {})
        trial_configs = manifest.get("trial_configs", {})
        for stage in stages:
            experiment_id = f"{trial_id}:{stage}"
            run_dir = stage_runs.get(stage)
            new_stage = experiment_id not in state["entities"]["experiments"]
            if not new_stage:
                existing_stages += 1
            if new_stage:
                record = {
                    "id": experiment_id,
                    "trial_id": trial_id,
                    "stage": stage,
                    "status": "completed" if run_dir else "unknown",
                    "evaluation_protocol": "legacy_single_seed",
                    "objective_eligibility": "historical_only",
                    "lockbox_status": "not_evaluated",
                    "checkpoint_status": "legacy_unverified",
                    "configuration": {"trial_config": trial_configs.get(stage)},
                    "execution": {
                        "run_dir": run_dir,
                        "protocol_version": "legacy_single_seed",
                        "provenance": {"created_by": "state_migration", "source": str(trial_dir / "trial_manifest.json")},
                    },
                "outcome": {"summary": summary if summary else None},
                "evaluation_protocol": "legacy_single_seed",
                "objective_eligibility": "historical_only",
                "lockbox_status": "not_evaluated",
                "checkpoint_status": "legacy_unverified",
                    "provenance": {"created_by": "state_migration", "source": str(trial_dir)},
                }
                state = record_entity(state, "experiments", experiment_id, record, actor="state_migration")
            if stages.index(stage) > 0:
                relation = {
                    "type": "reuses_checkpoint",
                    "source": experiment_id,
                    "target": f"{trial_id}:{stages[stages.index(stage) - 1]}",
                }
                if not any(
                    item.get("type") == relation["type"]
                    and item.get("source") == relation["source"]
                    and item.get("target") == relation["target"]
                    for item in state.get("relations", [])
                ):
                    state = apply_operations(state, [{"operation": "relation", "value": relation}], actor="state_migration")
                    relations_added += 1
            for observation in generate_observations(experiment_id, run_dir=run_dir):
                duplicate = any(
                    item.get("type") == observation.get("type")
                    and item.get("source_experiments") == observation.get("source_experiments")
                    and item.get("statement") == observation.get("statement")
                    for item in state["entities"]["observations"].values()
                )
                if not duplicate:
                    observation_id = observation["id"]
                    state = record_entity(state, "observations", observation_id, observation, actor="observation_engine")
                    observations_added += 1
            if new_stage:
                imported_stages += 1
        if trial_id not in state["entities"]["trials"]:
            trial_record = {
                "id": trial_id,
                "status": summary.get("status", "completed") if summary else "imported",
                "stage_experiment_ids": [f"{trial_id}:{stage}" for stage in stages],
                "configuration": trial_configs,
                "outcome": summary,
                "provenance": {"created_by": "state_migration", "source": str(trial_dir)},
            }
            state = record_entity(state, "trials", trial_id, trial_record, actor="state_migration")
            imported_trials += 1
        else:
            existing_trials += 1
    # Some historical campaigns kept only ledgers, not per-trial manifests.
    # Import those rows as immutable single-seed evidence with explicit
    # migration provenance rather than silently dropping them.
    for row in _read_ledger_rows(root):
        trial_id = str(row.get("trial_id") or row.get("id") or "")
        if not trial_id or trial_id in state["entities"]["trials"]:
            continue
        stages_for_trial = list(campaign_config["campaign"].get("stages", []))
        stage_ids = [f"{trial_id}:{stage}" for stage in stages_for_trial]
        for stage_id, stage in zip(stage_ids, stages_for_trial):
            if stage_id not in state["entities"]["experiments"]:
                state = record_entity(
                    state,
                    "experiments",
                    stage_id,
                    {
                        "id": stage_id,
                        "trial_id": trial_id,
                        "stage": stage,
                        "status": "imported",
                        "outcome": {"legacy_ledger": row},
                        "provenance": {"created_by": "state_migration", "source": str(root / "trials.csv" if (root / "trials.csv").exists() else root / "trials.jsonl")},
                    },
                    actor="state_migration",
                )
        state = record_entity(
            state,
            "trials",
            trial_id,
            {
                "id": trial_id,
                "status": row.get("status", "imported"),
                "stage_experiment_ids": stage_ids,
                "outcome": {"legacy_ledger": row},
                "evaluation_protocol": "single_seed_historical",
                "objective_eligibility": "historical_only",
                "lockbox_status": "not_evaluated",
                "checkpoint_status": "legacy_unverified",
                "provenance": {"created_by": "state_migration", "source": str(root)},
            },
            actor="state_migration",
        )
        imported_trials += 1
    state = update_controller_state(
        state,
        {
            "last_migration": {"campaign_id": campaign_config["campaign"]["id"], "imported_trials": imported_trials, "imported_stages": imported_stages},
            "protocol_reconciliation": {
                "status": "live_legacy_run_observed" if live_reconciliation and live_reconciliation.get("process_running") else "historical_import",
                "live_process": live_reconciliation,
                "protocol_version": "legacy_single_seed",
                "objective_eligibility": "historical_only",
            },
        },
        actor="state_migration",
    )
    transactional_update(target, lambda current: merge_nonconflicting_states(current, state))
    return {
        "state_path": str(target),
        "imported_trials": imported_trials,
        "imported_stages": imported_stages,
        "existing_trials": existing_trials,
        "existing_stages": existing_stages,
        "observations_added": observations_added,
        "relations_added": relations_added,
    }


def rebuild_compatibility_views(
    campaign_config: dict[str, Any],
    *,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write CSV/Markdown views without changing immutable scientific state."""

    target = Path(state_path or campaign_config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
    state = load_state(target)
    root = Path(campaign_config["artifacts"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial_id, trial in state["entities"]["trials"].items():
        outcome = trial.get("outcome", {}) if isinstance(trial.get("outcome"), dict) else {}
        rows.append(
            {
                "trial_id": trial_id,
                "status": trial.get("status", ""),
                "score": outcome.get("score", ""),
                "selected_metric": outcome.get("selected_metric", ""),
                "ranking_score": outcome.get("ranking_score", ""),
                "objective_eligible": outcome.get("objective_eligible", ""),
                "guardrail_passed": outcome.get("guardrail_passed", ""),
                "stage_experiment_ids": ";".join(trial.get("stage_experiment_ids", [])),
            }
        )
    columns = ["trial_id", "status", "score", "selected_metric", "ranking_score", "objective_eligible", "guardrail_passed", "stage_experiment_ids"]
    for filename, selected in (("trials.csv", rows), ("leaderboard.csv", [row for row in rows if str(row["objective_eligible"]).lower() == "true"])):
        path = root / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)
    markdown_path = root / "scientific_logbook.md"
    lines = [f"# Scientific campaign view: {campaign_config['campaign']['id']}", "", "| Trial | Status | Score | Eligible |", "| --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['trial_id']} | {row['status']} | {row['score']} | {row['objective_eligible']} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"root": str(root), "trials": len(rows), "trials_csv": str(root / "trials.csv"), "leaderboard_csv": str(root / "leaderboard.csv"), "logbook": str(markdown_path)}
