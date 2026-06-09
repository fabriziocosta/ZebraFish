from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_start_epoch: int | None = None
    early_stopping_monitor: str = "loss"
    early_stopping_smoothing: str = "none"
    early_stopping_smoothing_window: int = 1
    scheduler_patience: int | None = None
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    training_plot_dir: str | None = None
    training_plot_every_n_epochs: int = 1
    training_plot_smoothing_window: int = 5
    validation_split: float = 0.2
    random_state: int = 0
    standardize: bool = True
    device: str | None = None
    verbose: bool = True


@dataclass(frozen=True)
class LossWeightConfig:
    action_weight: float = 1.0
    compound_weight: float = 0.2
    concentration_weight: float = 0.2
    lambda_cross: float = 1.0
    lambda_align: float = 0.0
    cross_warmup_epochs: int = 5
    cross_ramp_epochs: int = 5
    teacher_student_warmup_epochs: int = 0
    probe_mask_probability: float = 0.25
    probe_alpha_local: float = 1.0
    probe_alpha_region_time: float = 1.0
    probe_alpha_derivative: float = 1.0
    probe_alpha_frequency: float = 1.0
    probe_alpha_correlation: float = 1.0


@dataclass(frozen=True)
class TimeChannel3DCNNConfig:
    conv_channels: tuple[int, ...] = (16, 32, 64)
    kernel_size_z: int | tuple[int, ...] = 1
    kernel_size_xy: int | tuple[int, ...] = 3
    stride_z: int | tuple[int, ...] = 1
    stride_xy: int | tuple[int, ...] = 1
    pool_kernel_z: int | tuple[int, ...] = 1
    pool_kernel_xy: int | tuple[int, ...] = 2
    pool_stride_z: int | tuple[int, ...] | None = None
    pool_stride_xy: int | tuple[int, ...] | None = None
    embedding_dim: int = 64
    dropout: float = 0.2


@dataclass(frozen=True)
class CommutativeCNNConfig:
    spatial_conv_channels: tuple[int, ...] = (16, 32, 64)
    spatial_kernel_size_z: int | tuple[int, ...] = 1
    spatial_kernel_size_xy: int | tuple[int, ...] = 3
    spatial_stride_z: int | tuple[int, ...] = 1
    spatial_stride_xy: int | tuple[int, ...] = 1
    spatial_pool_kernel_z: int | tuple[int, ...] = 1
    spatial_pool_kernel_xy: int | tuple[int, ...] = 2
    spatial_pool_stride_z: int | tuple[int, ...] | None = None
    spatial_pool_stride_xy: int | tuple[int, ...] | None = None
    temporal_st_channels: tuple[int, ...] = (128, 128)
    temporal_st_kernel_sizes: int | tuple[int, ...] = 3
    temporal_ts_channels: tuple[int, ...] = (64, 64)
    temporal_ts_kernel_sizes: int | tuple[int, ...] = 5
    spatial_agg_channels: tuple[int, ...] = (64, 128)
    spatial_agg_kernel_size_z: int | tuple[int, ...] = 3
    spatial_agg_kernel_size_xy: int | tuple[int, ...] = 3
    spatial_agg_stride_z: int | tuple[int, ...] = 1
    spatial_agg_stride_xy: int | tuple[int, ...] = 1
    spatial_agg_pool_kernel_z: int | tuple[int, ...] = 1
    spatial_agg_pool_kernel_xy: int | tuple[int, ...] = 2
    spatial_agg_pool_stride_z: int | tuple[int, ...] | None = None
    spatial_agg_pool_stride_xy: int | tuple[int, ...] | None = None
    patch_size_z: int = 1
    patch_size_xy: int = 16
    embedding_dim: int = 128
    probe_local_count: int = 32
    probe_region_grid: tuple[int, int, int] = (1, 2, 2)
    probe_time_bins: int = 8
    probe_frequency_bins: int = 4
    dropout: float = 0.2
    verbose: bool = True


@dataclass(frozen=True)
class CommutativeTransformerConfig:
    spatial_patch_size_st: tuple[int, int, int] = (1, 16, 16)
    spatial_patch_size_ts: tuple[int, int, int] = (1, 16, 16)
    temporal_patch_size_ts: int = 5
    embed_dim: int = 96
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.2
    attention_dropout: float = 0.0
    st_spatial_depth: int = 2
    st_temporal_depth: int = 2
    ts_temporal_depth: int = 2
    ts_spatial_depth: int = 2
    embedding_dim: int = 96
    probe_local_count: int = 32
    probe_region_grid: tuple[int, int, int] = (1, 2, 2)
    probe_time_bins: int = 8
    probe_frequency_bins: int = 4
    verbose: bool = True


def config_as_dict(config) -> dict[str, object]:
    return asdict(config)
