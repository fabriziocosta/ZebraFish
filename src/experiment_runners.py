from __future__ import annotations

from src.experiment_runner_cnn import (
    DEFAULT_10C_CONFIG_PATH,
    DEFAULT_13C_CONFIG_PATH,
    default_10c_pretraining_config,
    default_13c_config,
    ensure_default_cnn_configs,
    run_10c_pretraining,
    run_13c_finetune,
    write_default_10c_config,
    write_default_13c_config,
)
from src.experiment_runner_shared import (
    merge_dicts,
    read_yaml_mapping,
    to_yamlable,
    update_agent_run_status,
    write_yaml_mapping,
)
from src.experiment_runner_transformer import (
    DEFAULT_12T_CONFIG_PATH,
    DEFAULT_15T_CONFIG_PATH,
    default_12t_pretraining_config,
    default_15t_config,
    ensure_default_transformer_configs,
    run_12t_pretraining,
    run_15t_finetune,
    write_default_12t_config,
    write_default_15t_config,
)


def ensure_default_experiment_configs() -> None:
    ensure_default_cnn_configs()
    ensure_default_transformer_configs()


__all__ = [
    "DEFAULT_10C_CONFIG_PATH",
    "DEFAULT_12T_CONFIG_PATH",
    "DEFAULT_13C_CONFIG_PATH",
    "DEFAULT_15T_CONFIG_PATH",
    "default_10c_pretraining_config",
    "default_12t_pretraining_config",
    "default_13c_config",
    "default_15t_config",
    "ensure_default_cnn_configs",
    "ensure_default_experiment_configs",
    "ensure_default_transformer_configs",
    "merge_dicts",
    "read_yaml_mapping",
    "run_10c_pretraining",
    "run_12t_pretraining",
    "run_13c_finetune",
    "run_15t_finetune",
    "to_yamlable",
    "update_agent_run_status",
    "write_default_10c_config",
    "write_default_12t_config",
    "write_default_13c_config",
    "write_default_15t_config",
    "write_yaml_mapping",
]
