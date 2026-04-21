from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from torch import nn

from src.models.backbones_cnn import _PureCNNDualPathwayNetwork, _TimeChannel3DCNN
from src.models.backbones_transformer import _CommutativeTransformerNetwork
from src.models.configs import (
    CommutativeCNNConfig,
    CommutativeTransformerConfig,
    LossWeightConfig,
    OptimizationConfig,
    TimeChannel3DCNNConfig,
    config_as_dict,
)
from src.models.common import _PreparedData, _SharedMultitaskEstimatorMixin, _expand_per_block
from src.training.losses import apply_auxiliary_head_losses, commutative_consistency_loss
from src.training.loop import _collect_output_batches
from src.training.pretraining import _pretrain_commutative_estimator


_HEAD_PREFIXES = ("classifier.", "compound_classifier.", "concentration_classifier.")


def _apply_config(obj, *configs) -> None:
    for config in configs:
        if config is None:
            continue
        for key, value in config_as_dict(config).items():
            setattr(obj, key, value)


def _load_state_payload(path_or_state: str | Path | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(path_or_state, (str, Path)):
        payload = torch.load(Path(path_or_state), map_location="cpu")
    else:
        payload = path_or_state
    if "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return {str(key): value.detach().cpu() for key, value in payload.items()}


class _CommutativePretrainingMixin:
    def pretrain(
        self,
        X: torch.Tensor | np.ndarray,
        *,
        validation_data: torch.Tensor | np.ndarray | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
    ):
        return _pretrain_commutative_estimator(
            self,
            X,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    def _extract_transfer_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith(_HEAD_PREFIXES)
        }

    def load_pretrained_encoder(self, path_or_state: str | Path | Mapping[str, torch.Tensor]):
        self.pretrained_encoder_state_dict_ = {
            key: value
            for key, value in _load_state_payload(path_or_state).items()
            if not key.startswith(_HEAD_PREFIXES)
        }
        return self

    def save_pretrained_encoder(self, path: str | Path) -> Path:
        if hasattr(self, "pretrained_encoder_state_dict_"):
            state_dict = self.pretrained_encoder_state_dict_
        elif hasattr(self, "model_"):
            state_dict = self._extract_transfer_state_dict(self.model_)
        else:
            raise AttributeError("No pretrained encoder state is available to save")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
                "input_mean": getattr(self, "input_mean_", None),
                "input_std": getattr(self, "input_std_", None),
            },
            output_path,
        )
        return output_path

    def _load_pretrained_weights_into_model(self, model: nn.Module) -> None:
        state_dict = getattr(self, "pretrained_encoder_state_dict_", None)
        if state_dict is None and getattr(self, "pretrained_state_path", None):
            self.load_pretrained_encoder(self.pretrained_state_path)
            state_dict = getattr(self, "pretrained_encoder_state_dict_", None)
        if state_dict is None:
            return
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
        }
        if not compatible_state:
            raise ValueError("No compatible pretrained encoder weights matched the current model architecture")
        model.load_state_dict(compatible_state, strict=False)
        self.pretrained_loaded_keys_ = sorted(compatible_state)

    def _set_encoder_trainable(self, model: nn.Module, *, trainable: bool) -> None:
        for name, parameter in model.named_parameters():
            if name.startswith(_HEAD_PREFIXES):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = bool(trainable)


class TimeChannel3DCNNClassifier(
    _SharedMultitaskEstimatorMixin,
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
):
    def __init__(
        self,
        *,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size_z: int | tuple[int, ...] = 1,
        kernel_size_xy: int | tuple[int, ...] = 3,
        stride_z: int | tuple[int, ...] = 1,
        stride_xy: int | tuple[int, ...] = 1,
        pool_kernel_z: int | tuple[int, ...] = 1,
        pool_kernel_xy: int | tuple[int, ...] = 2,
        pool_stride_z: int | tuple[int, ...] | None = None,
        pool_stride_xy: int | tuple[int, ...] | None = None,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        batch_size: int = 16,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        action_weight: float = 1.0,
        compound_weight: float = 0.2,
        concentration_weight: float = 0.2,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        scheduler_patience: int | None = None,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        validation_split: float = 0.2,
        random_state: int = 0,
        standardize: bool = True,
        device: str | None = None,
        verbose: bool = True,
        model_config: TimeChannel3DCNNConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        loss_weight_config: LossWeightConfig | None = None,
    ) -> None:
        self.conv_channels = conv_channels
        self.kernel_size_z = kernel_size_z
        self.kernel_size_xy = kernel_size_xy
        self.stride_z = stride_z
        self.stride_xy = stride_xy
        self.pool_kernel_z = pool_kernel_z
        self.pool_kernel_xy = pool_kernel_xy
        self.pool_stride_z = pool_stride_z
        self.pool_stride_xy = pool_stride_xy
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.action_weight = action_weight
        self.compound_weight = compound_weight
        self.concentration_weight = concentration_weight
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.validation_split = validation_split
        self.random_state = random_state
        self.standardize = standardize
        self.device = device
        self.verbose = verbose
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.loss_weight_config = loss_weight_config
        _apply_config(self, model_config, optimization_config, loss_weight_config)

    def _build_model(self, in_channels: int, num_classes: int) -> _TimeChannel3DCNN:
        n_blocks = len(self.conv_channels)
        pool_stride_z = self.pool_kernel_z if self.pool_stride_z is None else self.pool_stride_z
        pool_stride_xy = self.pool_kernel_xy if self.pool_stride_xy is None else self.pool_stride_xy
        return _TimeChannel3DCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=self.conv_channels,
            kernel_size_z=_expand_per_block(self.kernel_size_z, n_blocks, "kernel_size_z"),
            kernel_size_xy=_expand_per_block(self.kernel_size_xy, n_blocks, "kernel_size_xy"),
            stride_z=_expand_per_block(self.stride_z, n_blocks, "stride_z"),
            stride_xy=_expand_per_block(self.stride_xy, n_blocks, "stride_xy"),
            pool_kernel_z=_expand_per_block(self.pool_kernel_z, n_blocks, "pool_kernel_z"),
            pool_kernel_xy=_expand_per_block(self.pool_kernel_xy, n_blocks, "pool_kernel_xy"),
            pool_stride_z=_expand_per_block(pool_stride_z, n_blocks, "pool_stride_z"),
            pool_stride_xy=_expand_per_block(pool_stride_xy, n_blocks, "pool_stride_xy"),
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
            num_compound_classes=0 if self.compound_classes_ is None else len(self.compound_classes_),
            num_concentration_classes=0 if self.concentration_classes_ is None else len(self.concentration_classes_),
        )

    def _build_model_from_prepared(self, prepared: _PreparedData) -> nn.Module:
        return self._build_model(
            in_channels=int(prepared.X_train.shape[1]),
            num_classes=len(self.classes_),
        )

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_loss = criterion(outputs["logits"], targets)
        total_loss = float(self.action_weight) * action_loss
        total_loss, compound_loss_value, concentration_loss_value = apply_auxiliary_head_losses(
            total_loss=total_loss,
            outputs=outputs,
            criterion=criterion,
            compound_targets=compound_targets,
            concentration_targets=concentration_targets,
            compound_weight=self.compound_weight,
            concentration_weight=self.concentration_weight,
        )
        return total_loss, {
            "action_loss": float(action_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }


class CommutativeCNNClassifier(
    _CommutativePretrainingMixin,
    _SharedMultitaskEstimatorMixin,
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
):
    def __init__(
        self,
        *,
        spatial_conv_channels: tuple[int, ...] = (16, 32, 64),
        spatial_kernel_size_z: int | tuple[int, ...] = 1,
        spatial_kernel_size_xy: int | tuple[int, ...] = 3,
        spatial_stride_z: int | tuple[int, ...] = 1,
        spatial_stride_xy: int | tuple[int, ...] = 1,
        spatial_pool_kernel_z: int | tuple[int, ...] = 1,
        spatial_pool_kernel_xy: int | tuple[int, ...] = 2,
        spatial_pool_stride_z: int | tuple[int, ...] | None = None,
        spatial_pool_stride_xy: int | tuple[int, ...] | None = None,
        temporal_st_channels: tuple[int, ...] = (128, 128),
        temporal_st_kernel_sizes: int | tuple[int, ...] = 3,
        temporal_ts_channels: tuple[int, ...] = (64, 64),
        temporal_ts_kernel_sizes: int | tuple[int, ...] = 5,
        spatial_agg_channels: tuple[int, ...] = (64, 128),
        spatial_agg_kernel_size_z: int | tuple[int, ...] = 3,
        spatial_agg_kernel_size_xy: int | tuple[int, ...] = 3,
        spatial_agg_stride_z: int | tuple[int, ...] = 1,
        spatial_agg_stride_xy: int | tuple[int, ...] = 1,
        spatial_agg_pool_kernel_z: int | tuple[int, ...] = 1,
        spatial_agg_pool_kernel_xy: int | tuple[int, ...] = 2,
        spatial_agg_pool_stride_z: int | tuple[int, ...] | None = None,
        spatial_agg_pool_stride_xy: int | tuple[int, ...] | None = None,
        patch_size_z: int = 1,
        patch_size_xy: int = 16,
        embedding_dim: int = 128,
        num_prototypes: int = 64,
        consistency_weight: float = 0.5,
        feature_weight: float = 0.1,
        prototype_temperature: float = 0.1,
        dropout: float = 0.2,
        batch_size: int = 16,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        action_weight: float = 1.0,
        compound_weight: float = 0.2,
        concentration_weight: float = 0.2,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        scheduler_patience: int | None = None,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        validation_split: float = 0.2,
        random_state: int = 0,
        standardize: bool = True,
        device: str | None = None,
        verbose: bool = True,
        model_config: CommutativeCNNConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        loss_weight_config: LossWeightConfig | None = None,
        pretrained_state_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        self.spatial_conv_channels = spatial_conv_channels
        self.spatial_kernel_size_z = spatial_kernel_size_z
        self.spatial_kernel_size_xy = spatial_kernel_size_xy
        self.spatial_stride_z = spatial_stride_z
        self.spatial_stride_xy = spatial_stride_xy
        self.spatial_pool_kernel_z = spatial_pool_kernel_z
        self.spatial_pool_kernel_xy = spatial_pool_kernel_xy
        self.spatial_pool_stride_z = spatial_pool_stride_z
        self.spatial_pool_stride_xy = spatial_pool_stride_xy
        self.temporal_st_channels = temporal_st_channels
        self.temporal_st_kernel_sizes = temporal_st_kernel_sizes
        self.temporal_ts_channels = temporal_ts_channels
        self.temporal_ts_kernel_sizes = temporal_ts_kernel_sizes
        self.spatial_agg_channels = spatial_agg_channels
        self.spatial_agg_kernel_size_z = spatial_agg_kernel_size_z
        self.spatial_agg_kernel_size_xy = spatial_agg_kernel_size_xy
        self.spatial_agg_stride_z = spatial_agg_stride_z
        self.spatial_agg_stride_xy = spatial_agg_stride_xy
        self.spatial_agg_pool_kernel_z = spatial_agg_pool_kernel_z
        self.spatial_agg_pool_kernel_xy = spatial_agg_pool_kernel_xy
        self.spatial_agg_pool_stride_z = spatial_agg_pool_stride_z
        self.spatial_agg_pool_stride_xy = spatial_agg_pool_stride_xy
        self.patch_size_z = patch_size_z
        self.patch_size_xy = patch_size_xy
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.consistency_weight = consistency_weight
        self.feature_weight = feature_weight
        self.prototype_temperature = prototype_temperature
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.action_weight = action_weight
        self.compound_weight = compound_weight
        self.concentration_weight = concentration_weight
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.validation_split = validation_split
        self.random_state = random_state
        self.standardize = standardize
        self.device = device
        self.verbose = verbose
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.loss_weight_config = loss_weight_config
        self.pretrained_state_path = pretrained_state_path
        self.freeze_backbone = freeze_backbone
        _apply_config(self, model_config, optimization_config, loss_weight_config)

    def _build_model(self, num_classes: int) -> _PureCNNDualPathwayNetwork:
        n_spatial_blocks = len(self.spatial_conv_channels)
        n_agg_blocks = len(self.spatial_agg_channels)
        temporal_st_kernels = _expand_per_block(
            self.temporal_st_kernel_sizes,
            len(self.temporal_st_channels),
            "temporal_st_kernel_sizes",
        )
        temporal_ts_kernels = _expand_per_block(
            self.temporal_ts_kernel_sizes,
            len(self.temporal_ts_channels),
            "temporal_ts_kernel_sizes",
        )
        spatial_pool_stride_z = self.spatial_pool_kernel_z if self.spatial_pool_stride_z is None else self.spatial_pool_stride_z
        spatial_pool_stride_xy = self.spatial_pool_kernel_xy if self.spatial_pool_stride_xy is None else self.spatial_pool_stride_xy
        spatial_agg_pool_stride_z = (
            self.spatial_agg_pool_kernel_z if self.spatial_agg_pool_stride_z is None else self.spatial_agg_pool_stride_z
        )
        spatial_agg_pool_stride_xy = (
            self.spatial_agg_pool_kernel_xy if self.spatial_agg_pool_stride_xy is None else self.spatial_agg_pool_stride_xy
        )
        return _PureCNNDualPathwayNetwork(
            num_classes=num_classes,
            spatial_conv_channels=self.spatial_conv_channels,
            spatial_kernel_size_z=_expand_per_block(self.spatial_kernel_size_z, n_spatial_blocks, "spatial_kernel_size_z"),
            spatial_kernel_size_xy=_expand_per_block(self.spatial_kernel_size_xy, n_spatial_blocks, "spatial_kernel_size_xy"),
            spatial_stride_z=_expand_per_block(self.spatial_stride_z, n_spatial_blocks, "spatial_stride_z"),
            spatial_stride_xy=_expand_per_block(self.spatial_stride_xy, n_spatial_blocks, "spatial_stride_xy"),
            spatial_pool_kernel_z=_expand_per_block(self.spatial_pool_kernel_z, n_spatial_blocks, "spatial_pool_kernel_z"),
            spatial_pool_kernel_xy=_expand_per_block(self.spatial_pool_kernel_xy, n_spatial_blocks, "spatial_pool_kernel_xy"),
            spatial_pool_stride_z=_expand_per_block(spatial_pool_stride_z, n_spatial_blocks, "spatial_pool_stride_z"),
            spatial_pool_stride_xy=_expand_per_block(spatial_pool_stride_xy, n_spatial_blocks, "spatial_pool_stride_xy"),
            temporal_st_channels=self.temporal_st_channels,
            temporal_st_kernel_sizes=temporal_st_kernels,
            temporal_ts_channels=self.temporal_ts_channels,
            temporal_ts_kernel_sizes=temporal_ts_kernels,
            spatial_agg_channels=self.spatial_agg_channels,
            spatial_agg_kernel_size_z=_expand_per_block(self.spatial_agg_kernel_size_z, n_agg_blocks, "spatial_agg_kernel_size_z"),
            spatial_agg_kernel_size_xy=_expand_per_block(self.spatial_agg_kernel_size_xy, n_agg_blocks, "spatial_agg_kernel_size_xy"),
            spatial_agg_stride_z=_expand_per_block(self.spatial_agg_stride_z, n_agg_blocks, "spatial_agg_stride_z"),
            spatial_agg_stride_xy=_expand_per_block(self.spatial_agg_stride_xy, n_agg_blocks, "spatial_agg_stride_xy"),
            spatial_agg_pool_kernel_z=_expand_per_block(self.spatial_agg_pool_kernel_z, n_agg_blocks, "spatial_agg_pool_kernel_z"),
            spatial_agg_pool_kernel_xy=_expand_per_block(self.spatial_agg_pool_kernel_xy, n_agg_blocks, "spatial_agg_pool_kernel_xy"),
            spatial_agg_pool_stride_z=_expand_per_block(spatial_agg_pool_stride_z, n_agg_blocks, "spatial_agg_pool_stride_z"),
            spatial_agg_pool_stride_xy=_expand_per_block(spatial_agg_pool_stride_xy, n_agg_blocks, "spatial_agg_pool_stride_xy"),
            patch_size_z=self.patch_size_z,
            patch_size_xy=self.patch_size_xy,
            embedding_dim=self.embedding_dim,
            num_prototypes=self.num_prototypes,
            dropout=self.dropout,
            num_compound_classes=0 if self.compound_classes_ is None else len(self.compound_classes_),
            num_concentration_classes=0 if self.concentration_classes_ is None else len(self.concentration_classes_),
        )

    def _build_model_from_prepared(self, prepared: _PreparedData) -> nn.Module:
        return self._build_model(num_classes=len(self.classes_))

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_loss = criterion(outputs["logits"], targets)
        consistency = commutative_consistency_loss(
            outputs["st_prototypes"],
            outputs["ts_prototypes"],
            temperature=float(self.prototype_temperature),
        )
        feature_alignment_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
        total_loss = (
            float(self.action_weight) * action_loss
            + float(self.consistency_weight) * consistency
            + float(self.feature_weight) * feature_alignment_loss
        )
        total_loss, compound_loss_value, concentration_loss_value = apply_auxiliary_head_losses(
            total_loss=total_loss,
            outputs=outputs,
            criterion=criterion,
            compound_targets=compound_targets,
            concentration_targets=concentration_targets,
            compound_weight=self.compound_weight,
            concentration_weight=self.concentration_weight,
        )
        return total_loss, {
            "action_loss": float(action_loss.item()),
            "commutative_consistency_loss": float(consistency.item()),
            "feature_alignment_loss": float(feature_alignment_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }

    def transform_branches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = _collect_output_batches(self, X)
        return {
            "st_embedding": outputs["st_embedding"],
            "ts_embedding": outputs["ts_embedding"],
            "embedding": outputs["embedding"],
        }


class CommutativeTransformerClassifier(
    _CommutativePretrainingMixin,
    _SharedMultitaskEstimatorMixin,
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
):
    def __init__(
        self,
        *,
        spatial_patch_size_st: tuple[int, int, int] = (1, 16, 16),
        spatial_patch_size_ts: tuple[int, int, int] = (1, 16, 16),
        temporal_patch_size_ts: int = 5,
        embed_dim: int = 96,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        attention_dropout: float = 0.0,
        st_spatial_depth: int = 2,
        st_temporal_depth: int = 2,
        ts_temporal_depth: int = 2,
        ts_spatial_depth: int = 2,
        embedding_dim: int = 96,
        num_prototypes: int = 32,
        consistency_weight: float = 0.5,
        feature_weight: float = 0.1,
        prototype_temperature: float = 0.1,
        batch_size: int = 8,
        epochs: int = 20,
        learning_rate: float = 2e-4,
        weight_decay: float = 3e-3,
        action_weight: float = 1.0,
        compound_weight: float = 0.2,
        concentration_weight: float = 0.2,
        early_stopping_patience: int | None = 4,
        early_stopping_min_delta: float = 0.0,
        scheduler_patience: int | None = 1,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        validation_split: float = 0.2,
        random_state: int = 0,
        standardize: bool = True,
        device: str | None = None,
        verbose: bool = True,
        model_config: CommutativeTransformerConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        loss_weight_config: LossWeightConfig | None = None,
        pretrained_state_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        self.spatial_patch_size_st = spatial_patch_size_st
        self.spatial_patch_size_ts = spatial_patch_size_ts
        self.temporal_patch_size_ts = temporal_patch_size_ts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.st_spatial_depth = st_spatial_depth
        self.st_temporal_depth = st_temporal_depth
        self.ts_temporal_depth = ts_temporal_depth
        self.ts_spatial_depth = ts_spatial_depth
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.consistency_weight = consistency_weight
        self.feature_weight = feature_weight
        self.prototype_temperature = prototype_temperature
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.action_weight = action_weight
        self.compound_weight = compound_weight
        self.concentration_weight = concentration_weight
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.validation_split = validation_split
        self.random_state = random_state
        self.standardize = standardize
        self.device = device
        self.verbose = verbose
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.loss_weight_config = loss_weight_config
        self.pretrained_state_path = pretrained_state_path
        self.freeze_backbone = freeze_backbone
        _apply_config(self, model_config, optimization_config, loss_weight_config)

    def _build_model(self, num_classes: int) -> _CommutativeTransformerNetwork:
        return _CommutativeTransformerNetwork(
            num_classes=num_classes,
            spatial_patch_size_st=self.spatial_patch_size_st,
            spatial_patch_size_ts=self.spatial_patch_size_ts,
            temporal_patch_size_ts=self.temporal_patch_size_ts,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            st_spatial_depth=self.st_spatial_depth,
            st_temporal_depth=self.st_temporal_depth,
            ts_temporal_depth=self.ts_temporal_depth,
            ts_spatial_depth=self.ts_spatial_depth,
            embedding_dim=self.embedding_dim,
            num_prototypes=self.num_prototypes,
            num_compound_classes=0 if self.compound_classes_ is None else len(self.compound_classes_),
            num_concentration_classes=0 if self.concentration_classes_ is None else len(self.concentration_classes_),
        )

    def _build_model_from_prepared(self, prepared: _PreparedData) -> nn.Module:
        return self._build_model(num_classes=len(self.classes_))

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_loss = criterion(outputs["logits"], targets)
        consistency = commutative_consistency_loss(
            outputs["st_prototypes"],
            outputs["ts_prototypes"],
            temperature=float(self.prototype_temperature),
        )
        feature_alignment_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
        total_loss = (
            float(self.action_weight) * action_loss
            + float(self.consistency_weight) * consistency
            + float(self.feature_weight) * feature_alignment_loss
        )
        total_loss, compound_loss_value, concentration_loss_value = apply_auxiliary_head_losses(
            total_loss=total_loss,
            outputs=outputs,
            criterion=criterion,
            compound_targets=compound_targets,
            concentration_targets=concentration_targets,
            compound_weight=self.compound_weight,
            concentration_weight=self.concentration_weight,
        )
        return total_loss, {
            "action_loss": float(action_loss.item()),
            "commutative_consistency_loss": float(consistency.item()),
            "feature_alignment_loss": float(feature_alignment_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }

    def transform_branches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = _collect_output_batches(self, X)
        return {
            "st_embedding": outputs["st_embedding"],
            "ts_embedding": outputs["ts_embedding"],
            "embedding": outputs["embedding"],
        }
