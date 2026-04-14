from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path


DEFAULT_CURRENT_DATASET_CONFIG_PATH = Path("artifacts/current_dataset.json")


def write_current_dataset_config(
    dataset_artifact_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CURRENT_DATASET_CONFIG_PATH,
) -> Path:
    artifact_path = Path(dataset_artifact_path).expanduser().resolve()
    target_path = Path(config_path).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_artifact_path": str(artifact_path),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target_path


def load_current_dataset_artifact_path(
    *,
    config_path: str | Path = DEFAULT_CURRENT_DATASET_CONFIG_PATH,
) -> Path:
    target_path = Path(config_path).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if "dataset_artifact_path" not in payload:
        raise KeyError(f"{target_path} does not contain 'dataset_artifact_path'")
    return Path(payload["dataset_artifact_path"])
