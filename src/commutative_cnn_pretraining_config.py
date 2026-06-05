from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

from src.models.configs import CommutativeCNNConfig, LossWeightConfig, OptimizationConfig

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_COMMUTATIVE_CNN_PRETRAINING_CONFIG_PATH = Path("artifacts/pretrained_commutative_cnn/config.yaml")


@dataclass(frozen=True)
class CommutativeCNNPretrainingConfig:
    unlabeled_dataset_path: Path
    pretrained_encoder_path: Path
    validation_fraction: float
    train_num_random_rotations: int
    rotation_range_degrees: float
    model_config: CommutativeCNNConfig
    optimization_config: OptimizationConfig
    loss_weight_config: LossWeightConfig


_COMMUTATIVE_CNN_TUPLE_FIELDS = {
    "spatial_conv_channels",
    "temporal_st_channels",
    "temporal_ts_channels",
    "spatial_agg_channels",
}
def _keep_dataclass_keys(config_class, values: dict[str, Any]) -> dict[str, Any]:
    valid_keys = {field.name for field in fields(config_class)}
    return {key: value for key, value in values.items() if key in valid_keys}


def _tupleify_config_values(config_class, values: dict[str, Any]) -> dict[str, Any]:
    coerced = _keep_dataclass_keys(config_class, dict(values))
    tuple_fields = set(_COMMUTATIVE_CNN_TUPLE_FIELDS)
    if config_class is CommutativeCNNConfig:
        tuple_fields.update(
            {
                "spatial_kernel_size_z",
                "spatial_kernel_size_xy",
                "spatial_stride_z",
                "spatial_stride_xy",
                "spatial_pool_kernel_z",
                "spatial_pool_kernel_xy",
                "spatial_pool_stride_z",
                "spatial_pool_stride_xy",
                "temporal_st_kernel_sizes",
                "temporal_ts_kernel_sizes",
                "spatial_agg_kernel_size_z",
                "spatial_agg_kernel_size_xy",
                "spatial_agg_stride_z",
                "spatial_agg_stride_xy",
                "spatial_agg_pool_kernel_z",
                "spatial_agg_pool_kernel_xy",
                "spatial_agg_pool_stride_z",
                "spatial_agg_pool_stride_xy",
                "probe_region_grid",
            }
        )
    for field_name in tuple_fields:
        value = coerced.get(field_name)
        if isinstance(value, list):
            coerced[field_name] = tuple(value)
    return coerced


def _to_payload(config: CommutativeCNNPretrainingConfig) -> dict[str, Any]:
    return {
        "unlabeled_dataset_path": str(config.unlabeled_dataset_path),
        "pretrained_encoder_path": str(config.pretrained_encoder_path),
        "validation_fraction": float(config.validation_fraction),
        "train_num_random_rotations": int(config.train_num_random_rotations),
        "rotation_range_degrees": float(config.rotation_range_degrees),
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


def write_commutative_cnn_pretraining_config(
    config: CommutativeCNNPretrainingConfig,
    path: str | Path = DEFAULT_COMMUTATIVE_CNN_PRETRAINING_CONFIG_PATH,
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


def load_commutative_cnn_pretraining_config(
    path: str | Path = DEFAULT_COMMUTATIVE_CNN_PRETRAINING_CONFIG_PATH,
) -> CommutativeCNNPretrainingConfig:
    target_path = Path(path).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    payload = _read_payload(target_path)
    return CommutativeCNNPretrainingConfig(
        unlabeled_dataset_path=Path(payload["unlabeled_dataset_path"]),
        pretrained_encoder_path=Path(payload["pretrained_encoder_path"]),
        validation_fraction=float(payload["validation_fraction"]),
        train_num_random_rotations=int(payload["train_num_random_rotations"]),
        rotation_range_degrees=float(payload["rotation_range_degrees"]),
        model_config=CommutativeCNNConfig(
            **_tupleify_config_values(CommutativeCNNConfig, dict(payload["model_config"]))
        ),
        optimization_config=OptimizationConfig(**dict(payload["optimization_config"])),
        loss_weight_config=LossWeightConfig(
            **_keep_dataclass_keys(LossWeightConfig, dict(payload["loss_weight_config"]))
        ),
    )
