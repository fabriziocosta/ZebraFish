from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any

from src.models.configs import CommutativeTransformerConfig, LossWeightConfig, OptimizationConfig

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_COMMUTATIVE_TRANSFORMER_PRETRAINING_CONFIG_PATH = Path(
    "artifacts/pretrained_commutative_transformer/config.yaml"
)


@dataclass(frozen=True)
class CommutativeTransformerPretrainingConfig:
    unlabeled_dataset_path: Path
    pretrained_encoder_path: Path
    model_config: CommutativeTransformerConfig
    optimization_config: OptimizationConfig
    loss_weight_config: LossWeightConfig = field(default_factory=LossWeightConfig)


def _keep_dataclass_keys(config_class, values: dict[str, Any]) -> dict[str, Any]:
    valid_keys = {field.name for field in fields(config_class)}
    return {key: value for key, value in values.items() if key in valid_keys}


def _tupleify_config_values(values: dict[str, Any]) -> dict[str, Any]:
    coerced = _keep_dataclass_keys(CommutativeTransformerConfig, dict(values))
    for field_name in ("spatial_patch_size_st", "spatial_patch_size_ts", "probe_region_grid"):
        value = coerced.get(field_name)
        if isinstance(value, list):
            coerced[field_name] = tuple(value)
    return coerced


def _to_payload(config: CommutativeTransformerPretrainingConfig) -> dict[str, Any]:
    return {
        "unlabeled_dataset_path": str(config.unlabeled_dataset_path),
        "pretrained_encoder_path": str(config.pretrained_encoder_path),
        "model_config": asdict(config.model_config),
        "optimization_config": asdict(config.optimization_config),
        "loss_weight_config": asdict(config.loss_weight_config),
    }


def _read_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def write_commutative_transformer_pretraining_config(
    config: CommutativeTransformerPretrainingConfig,
    path: str | Path = DEFAULT_COMMUTATIVE_TRANSFORMER_PRETRAINING_CONFIG_PATH,
) -> Path:
    target_path = Path(path).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_payload(config)
    if yaml is not None:
        rendered = yaml.safe_dump(payload, sort_keys=False)
    else:
        rendered = json.dumps(payload, indent=2, sort_keys=False)
    target_path.write_text(rendered, encoding="utf-8")
    return target_path


def load_commutative_transformer_pretraining_config(
    path: str | Path = DEFAULT_COMMUTATIVE_TRANSFORMER_PRETRAINING_CONFIG_PATH,
) -> CommutativeTransformerPretrainingConfig:
    target_path = Path(path).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    payload = _read_payload(target_path)
    return CommutativeTransformerPretrainingConfig(
        unlabeled_dataset_path=Path(payload["unlabeled_dataset_path"]),
        pretrained_encoder_path=Path(payload["pretrained_encoder_path"]),
        model_config=CommutativeTransformerConfig(**_tupleify_config_values(dict(payload["model_config"]))),
        optimization_config=OptimizationConfig(**dict(payload["optimization_config"])),
        loss_weight_config=LossWeightConfig(
            **_keep_dataclass_keys(LossWeightConfig, dict(payload.get("loss_weight_config", {})))
        ),
    )
