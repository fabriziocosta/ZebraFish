from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # Controller-only commands do not need pandas.
    pd = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{target} must contain a mapping")
    return payload


def write_yaml_mapping(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False) if yaml is not None else json.dumps(payload, indent=2)
    target.write_text(rendered, encoding="utf-8")
    return target


def to_yamlable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_yamlable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_yamlable(item) for item in value]
    return value


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def update_agent_run_status(**updates: Any) -> None:
    status_path = os.environ.get("ZF_AGENT_RUN_STATUS_PATH")
    if not status_path:
        return
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            payload = {}
    payload.update({key: to_yamlable(value) for key, value in updates.items()})
    if pd is not None:
        payload["updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    else:
        from datetime import datetime, timezone

        payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
