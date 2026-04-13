from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.tensor_utils import rotate_tensor_xy


def _as_tuple(value: int, xy_value: int) -> tuple[int, int, int]:
    return int(value), int(xy_value), int(xy_value)


def _expand_per_block(
    value: int | Sequence[int],
    n_blocks: int,
    name: str,
) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
        if len(values) != n_blocks:
            raise ValueError(f"{name} must have length {n_blocks}, got {len(values)}")
        return values
    return [int(value)] * n_blocks


def _format_eta(seconds: float) -> str:
    remaining = max(int(round(seconds)), 0)
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _ensure_tensor_5d(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(X, np.ndarray):
        tensor = torch.from_numpy(X)
    elif isinstance(X, torch.Tensor):
        tensor = X.detach().cpu()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(X)!r}")

    if tensor.ndim != 5:
        raise ValueError(f"Expected input with shape (N, T, Z, Y, X), got {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def _ensure_labels_1d(y: torch.Tensor | np.ndarray | Sequence[int]) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        values = y.detach().cpu().numpy()
    else:
        values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {values.shape}")
    return values


def _fit_target_encoder(
    values: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[object, int] | None]:
    if values is None:
        return None, None
    classes = np.array(sorted(np.unique(values)))
    return classes, {label: index for index, label in enumerate(classes)}


def _encode_with_mapping(
    values: np.ndarray | None,
    mapping: dict[object, int] | None,
) -> torch.Tensor | None:
    if values is None or mapping is None:
        return None
    encoded = np.array([mapping[value] for value in values], dtype=np.int64)
    return torch.from_numpy(encoded)


def _slice_optional(values: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return values[indices]


@dataclass
class _PreparedData:
    X_train: torch.Tensor
    y_train: torch.Tensor
    compound_train: torch.Tensor | None
    concentration_train: torch.Tensor | None
    X_val: torch.Tensor | None
    y_val: torch.Tensor | None
    compound_val: torch.Tensor | None
    concentration_val: torch.Tensor | None


@dataclass
class TensorDatasetSplits:
    metadata_all: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    holdout_indices: np.ndarray
    train_instance_ids: np.ndarray
    val_instance_ids: np.ndarray
    holdout_instance_ids: np.ndarray
    X_train_base: torch.Tensor
    y_train_base: torch.Tensor
    compound_train_base: torch.Tensor | None
    concentration_train_base: torch.Tensor | None
    metadata_train_base: pd.DataFrame
    X_val: torch.Tensor
    y_val: np.ndarray
    compound_val: np.ndarray | None
    concentration_val: np.ndarray | None
    metadata_val: pd.DataFrame
    X_holdout: torch.Tensor
    y_holdout: np.ndarray
    compound_holdout: np.ndarray | None
    concentration_holdout: np.ndarray | None
    metadata_holdout: pd.DataFrame


class _TimeChannel3DCNN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_size_z: Sequence[int],
        kernel_size_xy: Sequence[int],
        stride_z: Sequence[int],
        stride_xy: Sequence[int],
        pool_kernel_z: Sequence[int],
        pool_kernel_xy: Sequence[int],
        pool_stride_z: Sequence[int],
        pool_stride_xy: Sequence[int],
        embedding_dim: int,
        dropout: float,
        num_compound_classes: int = 0,
        num_concentration_classes: int = 0,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = in_channels
        for block_index, out_channels in enumerate(conv_channels):
            blocks.append(
                nn.Conv3d(
                    in_channels=current_channels,
                    out_channels=int(out_channels),
                    kernel_size=_as_tuple(kernel_size_z[block_index], kernel_size_xy[block_index]),
                    stride=_as_tuple(stride_z[block_index], stride_xy[block_index]),
                    padding=_as_tuple(kernel_size_z[block_index] // 2, kernel_size_xy[block_index] // 2),
                    bias=False,
                )
            )
            blocks.append(nn.BatchNorm3d(int(out_channels)))
            blocks.append(nn.ReLU(inplace=True))

            pool_kernel = _as_tuple(pool_kernel_z[block_index], pool_kernel_xy[block_index])
            pool_stride = _as_tuple(pool_stride_z[block_index], pool_stride_xy[block_index])
            if pool_kernel != (1, 1, 1):
                blocks.append(nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride))
            current_channels = int(out_channels)

        self.backbone = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.compound_classifier = nn.Linear(embedding_dim, num_compound_classes) if num_compound_classes > 0 else None
        self.concentration_classifier = (
            nn.Linear(embedding_dim, num_concentration_classes) if num_concentration_classes > 0 else None
        )

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        X = self.backbone(X)
        X = self.global_pool(X)
        return self.embedding(X)

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.forward_features(X)
        outputs = {
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }
        if self.compound_classifier is not None:
            outputs["compound_logits"] = self.compound_classifier(embedding)
        if self.concentration_classifier is not None:
            outputs["concentration_logits"] = self.concentration_classifier(embedding)
        return outputs


class _Conv1DStack(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for index, out_channels in enumerate(channels):
            kernel_size = int(kernel_sizes[index])
            blocks.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=int(out_channels),
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            blocks.append(nn.BatchNorm1d(int(out_channels)))
            blocks.append(nn.ReLU(inplace=True))
            if dropout > 0:
                blocks.append(nn.Dropout(p=dropout))
            current_channels = int(out_channels)
        self.network = nn.Sequential(*blocks)
        self.out_channels = current_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


class _Conv3DBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        conv_channels: Sequence[int],
        kernel_size_z: Sequence[int],
        kernel_size_xy: Sequence[int],
        stride_z: Sequence[int],
        stride_xy: Sequence[int],
        pool_kernel_z: Sequence[int],
        pool_kernel_xy: Sequence[int],
        pool_stride_z: Sequence[int],
        pool_stride_xy: Sequence[int],
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = in_channels
        for block_index, out_channels in enumerate(conv_channels):
            blocks.append(
                nn.Conv3d(
                    in_channels=current_channels,
                    out_channels=int(out_channels),
                    kernel_size=_as_tuple(kernel_size_z[block_index], kernel_size_xy[block_index]),
                    stride=_as_tuple(stride_z[block_index], stride_xy[block_index]),
                    padding=_as_tuple(kernel_size_z[block_index] // 2, kernel_size_xy[block_index] // 2),
                    bias=False,
                )
            )
            blocks.append(nn.BatchNorm3d(int(out_channels)))
            blocks.append(nn.ReLU(inplace=True))

            pool_kernel = _as_tuple(pool_kernel_z[block_index], pool_kernel_xy[block_index])
            pool_stride = _as_tuple(pool_stride_z[block_index], pool_stride_xy[block_index])
            if pool_kernel != (1, 1, 1):
                blocks.append(nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride))
            current_channels = int(out_channels)

        self.network = nn.Sequential(*blocks)
        self.out_channels = current_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def _sinusoidal_positional_encoding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0 or dim <= 0:
        raise ValueError("length and dim must be positive")
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-np.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class _PatchEmbed3D(nn.Module):
    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: Sequence[int]) -> None:
        super().__init__()
        patch = tuple(int(size) for size in patch_size)
        if len(patch) != 3 or any(size <= 0 for size in patch):
            raise ValueError("patch_size must contain three positive integers")
        self.patch_size = patch
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch,
            stride=patch,
            bias=True,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.proj(X)
        return X.flatten(2).transpose(1, 2)


class _PatchEmbed1D(nn.Module):
    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        if int(patch_size) <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.proj(X).transpose(1, 2)


class _TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        depth: int,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth == 0:
            self.encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=max(int(round(embed_dim * mlp_ratio)), embed_dim),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            layer.dropout1.p = attention_dropout
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        return self.norm(X)


class _PureCNNDualPathwayNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        spatial_conv_channels: Sequence[int],
        spatial_kernel_size_z: Sequence[int],
        spatial_kernel_size_xy: Sequence[int],
        spatial_stride_z: Sequence[int],
        spatial_stride_xy: Sequence[int],
        spatial_pool_kernel_z: Sequence[int],
        spatial_pool_kernel_xy: Sequence[int],
        spatial_pool_stride_z: Sequence[int],
        spatial_pool_stride_xy: Sequence[int],
        temporal_st_channels: Sequence[int],
        temporal_st_kernel_sizes: Sequence[int],
        temporal_ts_channels: Sequence[int],
        temporal_ts_kernel_sizes: Sequence[int],
        spatial_agg_channels: Sequence[int],
        spatial_agg_kernel_size_z: Sequence[int],
        spatial_agg_kernel_size_xy: Sequence[int],
        spatial_agg_stride_z: Sequence[int],
        spatial_agg_stride_xy: Sequence[int],
        spatial_agg_pool_kernel_z: Sequence[int],
        spatial_agg_pool_kernel_xy: Sequence[int],
        spatial_agg_pool_stride_z: Sequence[int],
        spatial_agg_pool_stride_xy: Sequence[int],
        patch_size_z: int,
        patch_size_xy: int,
        embedding_dim: int,
        num_prototypes: int,
        dropout: float,
        num_compound_classes: int = 0,
        num_concentration_classes: int = 0,
    ) -> None:
        super().__init__()
        self.patch_size_z = int(patch_size_z)
        self.patch_size_xy = int(patch_size_xy)
        if self.patch_size_z <= 0 or self.patch_size_xy <= 0:
            raise ValueError("patch_size_z and patch_size_xy must be positive")

        self.frame_encoder = _Conv3DBackbone(
            in_channels=1,
            conv_channels=spatial_conv_channels,
            kernel_size_z=spatial_kernel_size_z,
            kernel_size_xy=spatial_kernel_size_xy,
            stride_z=spatial_stride_z,
            stride_xy=spatial_stride_xy,
            pool_kernel_z=spatial_pool_kernel_z,
            pool_kernel_xy=spatial_pool_kernel_xy,
            pool_stride_z=spatial_pool_stride_z,
            pool_stride_xy=spatial_pool_stride_xy,
        )
        self.frame_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.temporal_st = _Conv1DStack(
            in_channels=self.frame_encoder.out_channels,
            channels=temporal_st_channels,
            kernel_sizes=temporal_st_kernel_sizes,
            dropout=dropout,
        )
        self.st_projection = nn.Linear(self.temporal_st.out_channels, embedding_dim)

        self.temporal_ts = _Conv1DStack(
            in_channels=1,
            channels=temporal_ts_channels,
            kernel_sizes=temporal_ts_kernel_sizes,
            dropout=dropout,
        )
        self.spatial_aggregator = _Conv3DBackbone(
            in_channels=self.temporal_ts.out_channels,
            conv_channels=spatial_agg_channels,
            kernel_size_z=spatial_agg_kernel_size_z,
            kernel_size_xy=spatial_agg_kernel_size_xy,
            stride_z=spatial_agg_stride_z,
            stride_xy=spatial_agg_stride_xy,
            pool_kernel_z=spatial_agg_pool_kernel_z,
            pool_kernel_xy=spatial_agg_pool_kernel_xy,
            pool_stride_z=spatial_agg_pool_stride_z,
            pool_stride_xy=spatial_agg_pool_stride_xy,
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.ts_projection = nn.Linear(self.spatial_aggregator.out_channels, embedding_dim)

        self.prototype_layer = nn.Linear(embedding_dim, num_prototypes, bias=False)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.compound_classifier = nn.Linear(embedding_dim, num_compound_classes) if num_compound_classes > 0 else None
        self.concentration_classifier = (
            nn.Linear(embedding_dim, num_concentration_classes) if num_concentration_classes > 0 else None
        )
        self.dropout = nn.Dropout(p=dropout)

    def _validate_patch_shape(self, X: torch.Tensor) -> tuple[int, int, int]:
        _, _, z_size, y_size, x_size = X.shape
        if z_size % self.patch_size_z != 0 or y_size % self.patch_size_xy != 0 or x_size % self.patch_size_xy != 0:
            raise ValueError(
                "Input spatial dimensions must be divisible by patch sizes: "
                f"got {(z_size, y_size, x_size)} with patch sizes "
                f"{(self.patch_size_z, self.patch_size_xy, self.patch_size_xy)}"
            )
        return z_size // self.patch_size_z, y_size // self.patch_size_xy, x_size // self.patch_size_xy

    def _forward_st(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_timepoints, _, _, _ = X.shape
        frames = X.reshape(n_samples * n_timepoints, 1, *X.shape[2:])
        frame_features = self.frame_encoder(frames)
        frame_features = self.frame_pool(frame_features).flatten(1)
        frame_features = frame_features.reshape(n_samples, n_timepoints, -1).transpose(1, 2)
        temporal_features = self.temporal_st(frame_features)
        temporal_features = temporal_features.mean(dim=-1)
        return self.st_projection(self.dropout(temporal_features))

    def _forward_ts(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_timepoints, _, _, _ = X.shape
        patch_grid_z, patch_grid_y, patch_grid_x = self._validate_patch_shape(X)

        pooled = F.avg_pool3d(
            X.reshape(n_samples * n_timepoints, 1, *X.shape[2:]),
            kernel_size=(self.patch_size_z, self.patch_size_xy, self.patch_size_xy),
            stride=(self.patch_size_z, self.patch_size_xy, self.patch_size_xy),
        )
        pooled = pooled.reshape(n_samples, n_timepoints, patch_grid_z, patch_grid_y, patch_grid_x)
        patch_sequences = pooled.permute(0, 2, 3, 4, 1).reshape(n_samples * patch_grid_z * patch_grid_y * patch_grid_x, 1, n_timepoints)

        patch_embeddings = self.temporal_ts(patch_sequences).mean(dim=-1)
        patch_embeddings = patch_embeddings.reshape(
            n_samples,
            patch_grid_z,
            patch_grid_y,
            patch_grid_x,
            self.temporal_ts.out_channels,
        ).permute(0, 4, 1, 2, 3)
        spatial_features = self.spatial_aggregator(patch_embeddings)
        spatial_features = self.spatial_pool(spatial_features).flatten(1)
        return self.ts_projection(self.dropout(spatial_features))

    def forward_features(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        st_embedding = self._forward_st(X)
        ts_embedding = self._forward_ts(X)
        fused_embedding = 0.5 * (st_embedding + ts_embedding)
        return {
            "st_embedding": st_embedding,
            "ts_embedding": ts_embedding,
            "embedding": fused_embedding,
            "st_prototypes": self.prototype_layer(st_embedding),
            "ts_prototypes": self.prototype_layer(ts_embedding),
        }

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward_features(X)
        outputs["logits"] = self.classifier(outputs["embedding"])
        if self.compound_classifier is not None:
            outputs["compound_logits"] = self.compound_classifier(outputs["embedding"])
        if self.concentration_classifier is not None:
            outputs["concentration_logits"] = self.concentration_classifier(outputs["embedding"])
        return outputs


class _CommutativeTransformerNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        spatial_patch_size_st: Sequence[int],
        spatial_patch_size_ts: Sequence[int],
        temporal_patch_size_ts: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        st_spatial_depth: int,
        st_temporal_depth: int,
        ts_temporal_depth: int,
        ts_spatial_depth: int,
        embedding_dim: int,
        num_prototypes: int,
        num_compound_classes: int = 0,
        num_concentration_classes: int = 0,
    ) -> None:
        super().__init__()
        self.spatial_patch_size_st = tuple(int(size) for size in spatial_patch_size_st)
        self.spatial_patch_size_ts = tuple(int(size) for size in spatial_patch_size_ts)
        self.temporal_patch_size_ts = int(temporal_patch_size_ts)
        if len(self.spatial_patch_size_st) != 3 or len(self.spatial_patch_size_ts) != 3:
            raise ValueError("spatial patch sizes must contain three integers")
        if any(size <= 0 for size in self.spatial_patch_size_st):
            raise ValueError("spatial_patch_size_st must be positive")
        if any(size <= 0 for size in self.spatial_patch_size_ts):
            raise ValueError("spatial_patch_size_ts must be positive")
        if self.temporal_patch_size_ts <= 0:
            raise ValueError("temporal_patch_size_ts must be positive")

        self.st_patch_embed = _PatchEmbed3D(in_channels=1, embed_dim=embed_dim, patch_size=self.spatial_patch_size_st)
        self.st_spatial_transformer = _TransformerEncoderStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            depth=st_spatial_depth,
        )
        self.st_temporal_transformer = _TransformerEncoderStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            depth=st_temporal_depth,
        )
        self.st_projection = nn.Sequential(
            nn.Linear(embed_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.ts_patch_embed = _PatchEmbed1D(in_channels=1, embed_dim=embed_dim, patch_size=self.temporal_patch_size_ts)
        self.ts_temporal_transformer = _TransformerEncoderStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            depth=ts_temporal_depth,
        )
        self.ts_spatial_transformer = _TransformerEncoderStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            depth=ts_spatial_depth,
        )
        self.ts_projection = nn.Sequential(
            nn.Linear(embed_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.prototype_layer = nn.Linear(embedding_dim, num_prototypes, bias=False)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.compound_classifier = nn.Linear(embedding_dim, num_compound_classes) if num_compound_classes > 0 else None
        self.concentration_classifier = (
            nn.Linear(embedding_dim, num_concentration_classes) if num_concentration_classes > 0 else None
        )

    def _validate_spatial_shapes(self, X: torch.Tensor) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        _, _, z_size, y_size, x_size = X.shape
        st_patch = self.spatial_patch_size_st
        ts_patch = self.spatial_patch_size_ts
        if z_size % st_patch[0] != 0 or y_size % st_patch[1] != 0 or x_size % st_patch[2] != 0:
            raise ValueError(
                "Input spatial dimensions must be divisible by spatial_patch_size_st: "
                f"got {(z_size, y_size, x_size)} and patch size {st_patch}"
            )
        if z_size % ts_patch[0] != 0 or y_size % ts_patch[1] != 0 or x_size % ts_patch[2] != 0:
            raise ValueError(
                "Input spatial dimensions must be divisible by spatial_patch_size_ts: "
                f"got {(z_size, y_size, x_size)} and patch size {ts_patch}"
            )
        return (
            (z_size // st_patch[0], y_size // st_patch[1], x_size // st_patch[2]),
            (z_size // ts_patch[0], y_size // ts_patch[1], x_size // ts_patch[2]),
        )

    def _forward_st(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_timepoints, _, _, _ = X.shape
        frames = X.reshape(n_samples * n_timepoints, 1, *X.shape[2:])
        spatial_tokens = self.st_patch_embed(frames)
        spatial_tokens = spatial_tokens + _sinusoidal_positional_encoding(
            spatial_tokens.shape[1],
            spatial_tokens.shape[2],
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        spatial_tokens = self.st_spatial_transformer(spatial_tokens)
        frame_embeddings = spatial_tokens.mean(dim=1).reshape(n_samples, n_timepoints, -1)
        frame_embeddings = frame_embeddings + _sinusoidal_positional_encoding(
            frame_embeddings.shape[1],
            frame_embeddings.shape[2],
            device=frame_embeddings.device,
            dtype=frame_embeddings.dtype,
        )
        temporal_tokens = self.st_temporal_transformer(frame_embeddings)
        return self.st_projection(temporal_tokens.mean(dim=1))

    def _forward_ts(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_timepoints, _, _, _ = X.shape
        if n_timepoints % self.temporal_patch_size_ts != 0:
            raise ValueError(
                "Input time dimension must be divisible by temporal_patch_size_ts: "
                f"got {n_timepoints} and patch size {self.temporal_patch_size_ts}"
            )

        _, patch_grid = self._validate_spatial_shapes(X)
        pooled = F.avg_pool3d(
            X.reshape(n_samples * n_timepoints, 1, *X.shape[2:]),
            kernel_size=self.spatial_patch_size_ts,
            stride=self.spatial_patch_size_ts,
        )
        pooled = pooled.reshape(n_samples, n_timepoints, *patch_grid)
        patch_sequences = pooled.permute(0, 2, 3, 4, 1).reshape(-1, 1, n_timepoints)
        temporal_tokens = self.ts_patch_embed(patch_sequences)
        temporal_tokens = temporal_tokens + _sinusoidal_positional_encoding(
            temporal_tokens.shape[1],
            temporal_tokens.shape[2],
            device=temporal_tokens.device,
            dtype=temporal_tokens.dtype,
        )
        temporal_tokens = self.ts_temporal_transformer(temporal_tokens)
        patch_embeddings = temporal_tokens.mean(dim=1).reshape(n_samples, -1, temporal_tokens.shape[-1])
        patch_embeddings = patch_embeddings + _sinusoidal_positional_encoding(
            patch_embeddings.shape[1],
            patch_embeddings.shape[2],
            device=patch_embeddings.device,
            dtype=patch_embeddings.dtype,
        )
        spatial_tokens = self.ts_spatial_transformer(patch_embeddings)
        return self.ts_projection(spatial_tokens.mean(dim=1))

    def forward_features(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_spatial_shapes(X)
        st_embedding = self._forward_st(X)
        ts_embedding = self._forward_ts(X)
        fused_embedding = 0.5 * (st_embedding + ts_embedding)
        return {
            "st_embedding": st_embedding,
            "ts_embedding": ts_embedding,
            "embedding": fused_embedding,
            "st_prototypes": self.prototype_layer(st_embedding),
            "ts_prototypes": self.prototype_layer(ts_embedding),
        }

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward_features(X)
        outputs["logits"] = self.classifier(outputs["embedding"])
        if self.compound_classifier is not None:
            outputs["compound_logits"] = self.compound_classifier(outputs["embedding"])
        if self.concentration_classifier is not None:
            outputs["concentration_logits"] = self.concentration_classifier(outputs["embedding"])
        return outputs


class TimeChannel3DCNNClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
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

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _standardize_fit(self, X: torch.Tensor) -> None:
        if self.standardize:
            self.input_mean_ = float(X.mean().item())
            self.input_std_ = float(X.std().item())
            if self.input_std_ == 0:
                self.input_std_ = 1.0
        else:
            self.input_mean_ = 0.0
            self.input_std_ = 1.0

    def _standardize_apply(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.input_mean_) / self.input_std_

    def _encode_labels(self, y: np.ndarray) -> torch.Tensor:
        encoded = np.array([self.class_to_index_[value] for value in y], dtype=np.int64)
        return torch.from_numpy(encoded)

    def _encode_compound_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.compound_class_to_index_)

    def _encode_concentration_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.concentration_class_to_index_)

    def _prepare_training_data(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> _PreparedData:
        X_tensor = _ensure_tensor_5d(X)
        y_values = _ensure_labels_1d(y)
        compound_values = _ensure_labels_1d(compound_y) if compound_y is not None else None
        concentration_values = _ensure_labels_1d(concentration_y) if concentration_y is not None else None
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")
        if compound_values is not None and len(compound_values) != len(y_values):
            raise ValueError("compound_y must have the same number of samples as y")
        if concentration_values is not None and len(concentration_values) != len(y_values):
            raise ValueError("concentration_y must have the same number of samples as y")

        self.classes_ = np.array(sorted(np.unique(y_values)))
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        self.compound_classes_, self.compound_class_to_index_ = _fit_target_encoder(compound_values)
        self.concentration_classes_, self.concentration_class_to_index_ = _fit_target_encoder(concentration_values)

        if validation_data is not None:
            X_val = _ensure_tensor_5d(validation_data[0])
            y_val_values = _ensure_labels_1d(validation_data[1])
            if len(X_val) != len(y_val_values):
                raise ValueError("validation_data tensors and labels must have the same number of samples")
            X_train = X_tensor
            y_train_values = y_values
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = _ensure_labels_1d(validation_compound_y) if validation_compound_y is not None else None
            concentration_val_values = (
                _ensure_labels_1d(validation_concentration_y) if validation_concentration_y is not None else None
            )
        elif self.validation_split and 0 < self.validation_split < 1:
            indices = np.arange(len(y_values))
            stratify = y_values if len(np.unique(y_values)) > 1 else None
            try:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=None,
                )
            X_train = X_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_train_values = y_values[train_indices]
            y_val_values = y_values[val_indices]
            compound_train_values = _slice_optional(compound_values, train_indices)
            concentration_train_values = _slice_optional(concentration_values, train_indices)
            compound_val_values = _slice_optional(compound_values, val_indices)
            concentration_val_values = _slice_optional(concentration_values, val_indices)
        else:
            X_train = X_tensor
            X_val = None
            y_train_values = y_values
            y_val_values = None
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = None
            concentration_val_values = None

        self._standardize_fit(X_train)
        X_train = self._standardize_apply(X_train)
        y_train = self._encode_labels(y_train_values)
        compound_train = self._encode_compound_labels(compound_train_values)
        concentration_train = self._encode_concentration_labels(concentration_train_values)

        if X_val is not None and y_val_values is not None:
            X_val = self._standardize_apply(X_val)
            y_val = self._encode_labels(y_val_values)
            compound_val = self._encode_compound_labels(compound_val_values)
            concentration_val = self._encode_concentration_labels(concentration_val_values)
        else:
            X_val = None
            y_val = None
            compound_val = None
            concentration_val = None

        return _PreparedData(
            X_train=X_train,
            y_train=y_train,
            compound_train=compound_train,
            concentration_train=concentration_train,
            X_val=X_val,
            y_val=y_val,
            compound_val=compound_val,
            concentration_val=concentration_val,
        )

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

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        classification_loss = criterion(outputs["logits"], targets)
        total_loss = classification_loss
        compound_loss_value = 0.0
        concentration_loss_value = 0.0
        if compound_targets is not None and "compound_logits" in outputs:
            compound_loss = criterion(outputs["compound_logits"], compound_targets)
            total_loss = total_loss + float(self.compound_weight) * compound_loss
            compound_loss_value = float(compound_loss.item())
        if concentration_targets is not None and "concentration_logits" in outputs:
            concentration_loss = criterion(outputs["concentration_logits"], concentration_targets)
            total_loss = total_loss + float(self.concentration_weight) * concentration_loss
            concentration_loss_value = float(concentration_loss.item())
        return total_loss, {
            "classification_loss": float(classification_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        *,
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None = None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> "TimeChannel3DCNNClassifier":
        prepared = self._prepare_training_data(
            X,
            y,
            validation_data,
            compound_y=compound_y,
            concentration_y=concentration_y,
            validation_compound_y=validation_compound_y,
            validation_concentration_y=validation_concentration_y,
        )
        self.model_ = self._build_model(
            in_channels=int(prepared.X_train.shape[1]),
            num_classes=len(self.classes_),
        )
        self.device_ = self._device()
        self.model_.to(self.device_)
        self.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])

        train_tensors: list[torch.Tensor] = [prepared.X_train, prepared.y_train]
        if prepared.compound_train is not None:
            train_tensors.append(prepared.compound_train)
        if prepared.concentration_train is not None:
            train_tensors.append(prepared.concentration_train)
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(
                TensorDataset(
                    *(
                        [prepared.X_val, prepared.y_val]
                        + ([prepared.compound_val] if prepared.compound_val is not None else [])
                        + ([prepared.concentration_val] if prepared.concentration_val is not None else [])
                    )
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )
            if prepared.X_val is not None and prepared.y_val is not None
            else None
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            if self.scheduler_patience is not None and self.scheduler_patience >= 0
            else None
        )

        history_rows: list[dict[str, float | int]] = []
        best_state = deepcopy(self.model_.state_dict())
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss_sum = 0.0
            train_classification_loss_sum = 0.0
            train_compound_loss_sum = 0.0
            train_concentration_loss_sum = 0.0
            train_count = 0
            for batch in train_loader:
                X_batch = batch[0].to(self.device_, non_blocking=True)
                y_batch = batch[1].to(self.device_, non_blocking=True)
                compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_train is not None else None
                concentration_batch = (
                    batch[3 if prepared.compound_train is not None else 2].to(self.device_, non_blocking=True)
                    if prepared.concentration_train is not None
                    else None
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model_(X_batch)
                loss, loss_components = self._compute_losses(
                    outputs,
                    y_batch,
                    criterion,
                    compound_targets=compound_batch,
                    concentration_targets=concentration_batch,
                )
                loss.backward()
                optimizer.step()

                batch_size = int(X_batch.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_classification_loss_sum += loss_components["classification_loss"] * batch_size
                train_compound_loss_sum += loss_components["compound_loss"] * batch_size
                train_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                train_count += batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "train_classification_loss": train_classification_loss_sum / max(train_count, 1),
                "train_compound_loss": train_compound_loss_sum / max(train_count, 1),
                "train_concentration_loss": train_concentration_loss_sum / max(train_count, 1),
            }

            if val_loader is not None:
                self.model_.eval()
                val_loss_sum = 0.0
                val_classification_loss_sum = 0.0
                val_compound_loss_sum = 0.0
                val_concentration_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0].to(self.device_, non_blocking=True)
                        y_batch = batch[1].to(self.device_, non_blocking=True)
                        compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_val is not None else None
                        concentration_batch = (
                            batch[3 if prepared.compound_val is not None else 2].to(self.device_, non_blocking=True)
                            if prepared.concentration_val is not None
                            else None
                        )
                        outputs = self.model_(X_batch)
                        loss, loss_components = self._compute_losses(
                            outputs,
                            y_batch,
                            criterion,
                            compound_targets=compound_batch,
                            concentration_targets=concentration_batch,
                        )
                        batch_size = int(X_batch.shape[0])
                        val_loss_sum += float(loss.item()) * batch_size
                        val_classification_loss_sum += loss_components["classification_loss"] * batch_size
                        val_compound_loss_sum += loss_components["compound_loss"] * batch_size
                        val_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                        val_count += batch_size
                row["val_loss"] = val_loss_sum / max(val_count, 1)
                row["val_classification_loss"] = val_classification_loss_sum / max(val_count, 1)
                row["val_compound_loss"] = val_compound_loss_sum / max(val_count, 1)
                row["val_concentration_loss"] = val_concentration_loss_sum / max(val_count, 1)
                metric = float(row["val_loss"])
            else:
                metric = float(row["train_loss"])

            improved = metric < (best_metric - float(self.early_stopping_min_delta))
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = deepcopy(self.model_.state_dict())
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                scheduler.step(metric)

            history_rows.append(row)
            if self.verbose:
                elapsed = time.perf_counter() - training_start
                avg_epoch_seconds = elapsed / epoch
                eta = _format_eta(avg_epoch_seconds * (self.epochs - epoch))
                current_lr = optimizer.param_groups[0]["lr"]
                if "val_loss" in row:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"val_loss={row['val_loss']:.4f} lr={current_lr:.2e} eta={eta}"
                    )
                else:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"lr={current_lr:.2e} eta={eta}"
                    )

            if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(
                        f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                        f"best_metric={best_metric:.4f}"
                    )
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = pd.DataFrame(history_rows)
        self.best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
        self.best_metric_ = float(best_metric)
        return self

    def _forward_batches(self, X: torch.Tensor | np.ndarray, *, features: bool) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                batch_out = (
                    self.model_.forward_features(X_batch)
                    if features
                    else self.model_(X_batch)["logits"]
                )
                outputs.append(batch_out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _forward_output_batches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        collected: dict[str, list[np.ndarray]] = {}
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                outputs = self.model_(X_batch)
                for key, value in outputs.items():
                    if key == "embedding":
                        continue
                    collected.setdefault(key, []).append(value.detach().cpu().numpy())
        return {key: np.concatenate(values, axis=0) for key, values in collected.items()}

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._forward_output_batches(X)
        result = {"action": torch.softmax(torch.from_numpy(outputs["logits"]), dim=1).numpy()}
        if "compound_logits" in outputs:
            result["compound"] = torch.softmax(torch.from_numpy(outputs["compound_logits"]), dim=1).numpy()
        if "concentration_logits" in outputs:
            result["concentration"] = torch.softmax(torch.from_numpy(outputs["concentration_logits"]), dim=1).numpy()
        return result

    def predict(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        probabilities = self.predict_proba(X)
        predictions = {"action": self.classes_[probabilities["action"].argmax(axis=1)]}
        if "compound" in probabilities:
            predictions["compound"] = self.compound_classes_[probabilities["compound"].argmax(axis=1)]
        if "concentration" in probabilities:
            predictions["concentration"] = self.concentration_classes_[probabilities["concentration"].argmax(axis=1)]
        return predictions

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return self._forward_batches(X, features=True)

    def score(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | Sequence[int]) -> float:
        y_true = _ensure_labels_1d(y)
        y_pred = self.predict(X)["action"]
        return float(accuracy_score(y_true, y_pred))


class CommutativeCNNClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
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

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _standardize_fit(self, X: torch.Tensor) -> None:
        if self.standardize:
            self.input_mean_ = float(X.mean().item())
            self.input_std_ = float(X.std().item())
            if self.input_std_ == 0:
                self.input_std_ = 1.0
        else:
            self.input_mean_ = 0.0
            self.input_std_ = 1.0

    def _standardize_apply(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.input_mean_) / self.input_std_

    def _encode_labels(self, y: np.ndarray) -> torch.Tensor:
        encoded = np.array([self.class_to_index_[value] for value in y], dtype=np.int64)
        return torch.from_numpy(encoded)

    def _encode_compound_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.compound_class_to_index_)

    def _encode_concentration_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.concentration_class_to_index_)

    def _prepare_training_data(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> _PreparedData:
        X_tensor = _ensure_tensor_5d(X)
        y_values = _ensure_labels_1d(y)
        compound_values = _ensure_labels_1d(compound_y) if compound_y is not None else None
        concentration_values = _ensure_labels_1d(concentration_y) if concentration_y is not None else None
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")
        if compound_values is not None and len(compound_values) != len(y_values):
            raise ValueError("compound_y must have the same number of samples as y")
        if concentration_values is not None and len(concentration_values) != len(y_values):
            raise ValueError("concentration_y must have the same number of samples as y")

        self.classes_ = np.array(sorted(np.unique(y_values)))
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        self.compound_classes_, self.compound_class_to_index_ = _fit_target_encoder(compound_values)
        self.concentration_classes_, self.concentration_class_to_index_ = _fit_target_encoder(concentration_values)

        if validation_data is not None:
            X_val = _ensure_tensor_5d(validation_data[0])
            y_val_values = _ensure_labels_1d(validation_data[1])
            if len(X_val) != len(y_val_values):
                raise ValueError("validation_data tensors and labels must have the same number of samples")
            X_train = X_tensor
            y_train_values = y_values
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = _ensure_labels_1d(validation_compound_y) if validation_compound_y is not None else None
            concentration_val_values = (
                _ensure_labels_1d(validation_concentration_y) if validation_concentration_y is not None else None
            )
        elif self.validation_split and 0 < self.validation_split < 1:
            indices = np.arange(len(y_values))
            stratify = y_values if len(np.unique(y_values)) > 1 else None
            try:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=None,
                )
            X_train = X_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_train_values = y_values[train_indices]
            y_val_values = y_values[val_indices]
            compound_train_values = _slice_optional(compound_values, train_indices)
            concentration_train_values = _slice_optional(concentration_values, train_indices)
            compound_val_values = _slice_optional(compound_values, val_indices)
            concentration_val_values = _slice_optional(concentration_values, val_indices)
        else:
            X_train = X_tensor
            X_val = None
            y_train_values = y_values
            y_val_values = None
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = None
            concentration_val_values = None

        self._standardize_fit(X_train)
        X_train = self._standardize_apply(X_train)
        y_train = self._encode_labels(y_train_values)
        compound_train = self._encode_compound_labels(compound_train_values)
        concentration_train = self._encode_concentration_labels(concentration_train_values)

        if X_val is not None and y_val_values is not None:
            X_val = self._standardize_apply(X_val)
            y_val = self._encode_labels(y_val_values)
            compound_val = self._encode_compound_labels(compound_val_values)
            concentration_val = self._encode_concentration_labels(concentration_val_values)
        else:
            X_val = None
            y_val = None
            compound_val = None
            concentration_val = None

        return _PreparedData(
            X_train=X_train,
            y_train=y_train,
            compound_train=compound_train,
            concentration_train=concentration_train,
            X_val=X_val,
            y_val=y_val,
            compound_val=compound_val,
            concentration_val=concentration_val,
        )

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

    def _consistency_loss(self, st_logits: torch.Tensor, ts_logits: torch.Tensor) -> torch.Tensor:
        temperature = float(self.prototype_temperature)
        st_targets = torch.softmax(st_logits.detach() / temperature, dim=1)
        ts_targets = torch.softmax(ts_logits.detach() / temperature, dim=1)
        st_log_probs = torch.log_softmax(st_logits / temperature, dim=1)
        ts_log_probs = torch.log_softmax(ts_logits / temperature, dim=1)
        loss_st = -(st_targets * ts_log_probs).sum(dim=1).mean()
        loss_ts = -(ts_targets * st_log_probs).sum(dim=1).mean()
        return 0.5 * (loss_st + loss_ts)

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        classification_loss = criterion(outputs["logits"], targets)
        consistency_loss = self._consistency_loss(outputs["st_prototypes"], outputs["ts_prototypes"])
        feature_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
        total_loss = (
            classification_loss
            + float(self.consistency_weight) * consistency_loss
            + float(self.feature_weight) * feature_loss
        )
        compound_loss_value = 0.0
        concentration_loss_value = 0.0
        if compound_targets is not None and "compound_logits" in outputs:
            compound_loss = criterion(outputs["compound_logits"], compound_targets)
            total_loss = total_loss + float(self.compound_weight) * compound_loss
            compound_loss_value = float(compound_loss.item())
        if concentration_targets is not None and "concentration_logits" in outputs:
            concentration_loss = criterion(outputs["concentration_logits"], concentration_targets)
            total_loss = total_loss + float(self.concentration_weight) * concentration_loss
            concentration_loss_value = float(concentration_loss.item())
        return total_loss, {
            "classification_loss": float(classification_loss.item()),
            "consistency_loss": float(consistency_loss.item()),
            "feature_loss": float(feature_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        *,
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None = None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> "CommutativeCNNClassifier":
        prepared = self._prepare_training_data(
            X,
            y,
            validation_data,
            compound_y=compound_y,
            concentration_y=concentration_y,
            validation_compound_y=validation_compound_y,
            validation_concentration_y=validation_concentration_y,
        )
        self.model_ = self._build_model(num_classes=len(self.classes_))
        self.device_ = self._device()
        self.model_.to(self.device_)
        self.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])

        train_tensors: list[torch.Tensor] = [prepared.X_train, prepared.y_train]
        if prepared.compound_train is not None:
            train_tensors.append(prepared.compound_train)
        if prepared.concentration_train is not None:
            train_tensors.append(prepared.concentration_train)
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(
                TensorDataset(
                    *(
                        [prepared.X_val, prepared.y_val]
                        + ([prepared.compound_val] if prepared.compound_val is not None else [])
                        + ([prepared.concentration_val] if prepared.concentration_val is not None else [])
                    )
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )
            if prepared.X_val is not None and prepared.y_val is not None
            else None
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            if self.scheduler_patience is not None and self.scheduler_patience >= 0
            else None
        )

        history_rows: list[dict[str, float | int]] = []
        best_state = deepcopy(self.model_.state_dict())
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_total_loss_sum = 0.0
            train_classification_loss_sum = 0.0
            train_consistency_loss_sum = 0.0
            train_feature_loss_sum = 0.0
            train_compound_loss_sum = 0.0
            train_concentration_loss_sum = 0.0
            train_count = 0

            for batch in train_loader:
                X_batch = batch[0].to(self.device_, non_blocking=True)
                y_batch = batch[1].to(self.device_, non_blocking=True)
                compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_train is not None else None
                concentration_batch = (
                    batch[3 if prepared.compound_train is not None else 2].to(self.device_, non_blocking=True)
                    if prepared.concentration_train is not None
                    else None
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model_(X_batch)
                loss, loss_components = self._compute_losses(
                    outputs,
                    y_batch,
                    criterion,
                    compound_targets=compound_batch,
                    concentration_targets=concentration_batch,
                )
                loss.backward()
                optimizer.step()

                batch_size = int(X_batch.shape[0])
                train_total_loss_sum += float(loss.item()) * batch_size
                train_classification_loss_sum += loss_components["classification_loss"] * batch_size
                train_consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                train_feature_loss_sum += loss_components["feature_loss"] * batch_size
                train_compound_loss_sum += loss_components["compound_loss"] * batch_size
                train_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                train_count += batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_total_loss_sum / max(train_count, 1),
                "train_classification_loss": train_classification_loss_sum / max(train_count, 1),
                "train_consistency_loss": train_consistency_loss_sum / max(train_count, 1),
                "train_feature_loss": train_feature_loss_sum / max(train_count, 1),
                "train_compound_loss": train_compound_loss_sum / max(train_count, 1),
                "train_concentration_loss": train_concentration_loss_sum / max(train_count, 1),
            }

            if val_loader is not None:
                self.model_.eval()
                val_total_loss_sum = 0.0
                val_classification_loss_sum = 0.0
                val_consistency_loss_sum = 0.0
                val_feature_loss_sum = 0.0
                val_compound_loss_sum = 0.0
                val_concentration_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0].to(self.device_, non_blocking=True)
                        y_batch = batch[1].to(self.device_, non_blocking=True)
                        compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_val is not None else None
                        concentration_batch = (
                            batch[3 if prepared.compound_val is not None else 2].to(self.device_, non_blocking=True)
                            if prepared.concentration_val is not None
                            else None
                        )
                        outputs = self.model_(X_batch)
                        loss, loss_components = self._compute_losses(
                            outputs,
                            y_batch,
                            criterion,
                            compound_targets=compound_batch,
                            concentration_targets=concentration_batch,
                        )
                        batch_size = int(X_batch.shape[0])
                        val_total_loss_sum += float(loss.item()) * batch_size
                        val_classification_loss_sum += loss_components["classification_loss"] * batch_size
                        val_consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                        val_feature_loss_sum += loss_components["feature_loss"] * batch_size
                        val_compound_loss_sum += loss_components["compound_loss"] * batch_size
                        val_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                        val_count += batch_size
                row["val_loss"] = val_total_loss_sum / max(val_count, 1)
                row["val_classification_loss"] = val_classification_loss_sum / max(val_count, 1)
                row["val_consistency_loss"] = val_consistency_loss_sum / max(val_count, 1)
                row["val_feature_loss"] = val_feature_loss_sum / max(val_count, 1)
                row["val_compound_loss"] = val_compound_loss_sum / max(val_count, 1)
                row["val_concentration_loss"] = val_concentration_loss_sum / max(val_count, 1)
                metric = float(row["val_loss"])
            else:
                metric = float(row["train_loss"])

            improved = metric < (best_metric - float(self.early_stopping_min_delta))
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = deepcopy(self.model_.state_dict())
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                scheduler.step(metric)

            history_rows.append(row)
            if self.verbose:
                elapsed = time.perf_counter() - training_start
                avg_epoch_seconds = elapsed / epoch
                eta = _format_eta(avg_epoch_seconds * (self.epochs - epoch))
                current_lr = optimizer.param_groups[0]["lr"]
                if "val_loss" in row:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"val_loss={row['val_loss']:.4f} lr={current_lr:.2e} eta={eta}"
                    )
                else:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"lr={current_lr:.2e} eta={eta}"
                    )

            if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(
                        f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                        f"best_metric={best_metric:.4f}"
                    )
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = pd.DataFrame(history_rows)
        self.best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
        self.best_metric_ = float(best_metric)
        return self

    def _forward_batches(self, X: torch.Tensor | np.ndarray, *, output_key: str) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                batch_out = self.model_(X_batch)[output_key]
                outputs.append(batch_out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _forward_feature_batches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        collected = {
            "st_embedding": [],
            "ts_embedding": [],
            "embedding": [],
            "st_prototypes": [],
            "ts_prototypes": [],
            "logits": [],
            "compound_logits": [],
            "concentration_logits": [],
        }
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                outputs = self.model_(X_batch)
                for key in list(collected):
                    if key in outputs:
                        collected[key].append(outputs[key].detach().cpu().numpy())
                    else:
                        collected.pop(key, None)
        return {key: np.concatenate(values, axis=0) for key, values in collected.items()}

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._forward_feature_batches(X)
        result = {"action": torch.softmax(torch.from_numpy(outputs["logits"]), dim=1).numpy()}
        if "compound_logits" in outputs:
            result["compound"] = torch.softmax(torch.from_numpy(outputs["compound_logits"]), dim=1).numpy()
        if "concentration_logits" in outputs:
            result["concentration"] = torch.softmax(torch.from_numpy(outputs["concentration_logits"]), dim=1).numpy()
        return result

    def predict(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        probabilities = self.predict_proba(X)
        predictions = {"action": self.classes_[probabilities["action"].argmax(axis=1)]}
        if "compound" in probabilities:
            predictions["compound"] = self.compound_classes_[probabilities["compound"].argmax(axis=1)]
        if "concentration" in probabilities:
            predictions["concentration"] = self.concentration_classes_[probabilities["concentration"].argmax(axis=1)]
        return predictions

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return self._forward_batches(X, output_key="embedding")

    def transform_branches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._forward_feature_batches(X)
        return {
            "st_embedding": outputs["st_embedding"],
            "ts_embedding": outputs["ts_embedding"],
            "embedding": outputs["embedding"],
        }

    def evaluate_loss_components(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
    ) -> dict[str, float]:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        y_values = _ensure_labels_1d(y)
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")
        y_tensor = self._encode_labels(y_values)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        total_loss_sum = 0.0
        classification_loss_sum = 0.0
        consistency_loss_sum = 0.0
        feature_loss_sum = 0.0
        count = 0
        self.model_.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                y_batch = y_batch.to(self.device_, non_blocking=True)
                outputs = self.model_(X_batch)
                loss, loss_components = self._compute_losses(outputs, y_batch, criterion)
                batch_size = int(X_batch.shape[0])
                total_loss_sum += float(loss.item()) * batch_size
                classification_loss_sum += loss_components["classification_loss"] * batch_size
                consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                feature_loss_sum += loss_components["feature_loss"] * batch_size
                count += batch_size
        return {
            "loss": total_loss_sum / max(count, 1),
            "classification_loss": classification_loss_sum / max(count, 1),
            "consistency_loss": consistency_loss_sum / max(count, 1),
            "feature_loss": feature_loss_sum / max(count, 1),
        }

    def score(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | Sequence[int]) -> float:
        y_true = _ensure_labels_1d(y)
        y_pred = self.predict(X)["action"]
        return float(accuracy_score(y_true, y_pred))


class CommutativeTransformerClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
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

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _standardize_fit(self, X: torch.Tensor) -> None:
        if self.standardize:
            self.input_mean_ = float(X.mean().item())
            self.input_std_ = float(X.std().item())
            if self.input_std_ == 0:
                self.input_std_ = 1.0
        else:
            self.input_mean_ = 0.0
            self.input_std_ = 1.0

    def _standardize_apply(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.input_mean_) / self.input_std_

    def _encode_labels(self, y: np.ndarray) -> torch.Tensor:
        encoded = np.array([self.class_to_index_[value] for value in y], dtype=np.int64)
        return torch.from_numpy(encoded)

    def _encode_compound_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.compound_class_to_index_)

    def _encode_concentration_labels(self, y: np.ndarray | None) -> torch.Tensor | None:
        return _encode_with_mapping(y, self.concentration_class_to_index_)

    def _prepare_training_data(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> _PreparedData:
        X_tensor = _ensure_tensor_5d(X)
        y_values = _ensure_labels_1d(y)
        compound_values = _ensure_labels_1d(compound_y) if compound_y is not None else None
        concentration_values = _ensure_labels_1d(concentration_y) if concentration_y is not None else None
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")
        if compound_values is not None and len(compound_values) != len(y_values):
            raise ValueError("compound_y must have the same number of samples as y")
        if concentration_values is not None and len(concentration_values) != len(y_values):
            raise ValueError("concentration_y must have the same number of samples as y")

        self.classes_ = np.array(sorted(np.unique(y_values)))
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        self.compound_classes_, self.compound_class_to_index_ = _fit_target_encoder(compound_values)
        self.concentration_classes_, self.concentration_class_to_index_ = _fit_target_encoder(concentration_values)

        if validation_data is not None:
            X_val = _ensure_tensor_5d(validation_data[0])
            y_val_values = _ensure_labels_1d(validation_data[1])
            if len(X_val) != len(y_val_values):
                raise ValueError("validation_data tensors and labels must have the same number of samples")
            X_train = X_tensor
            y_train_values = y_values
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = _ensure_labels_1d(validation_compound_y) if validation_compound_y is not None else None
            concentration_val_values = (
                _ensure_labels_1d(validation_concentration_y) if validation_concentration_y is not None else None
            )
        elif self.validation_split and 0 < self.validation_split < 1:
            indices = np.arange(len(y_values))
            stratify = y_values if len(np.unique(y_values)) > 1 else None
            try:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.validation_split,
                    random_state=self.random_state,
                    stratify=None,
                )
            X_train = X_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_train_values = y_values[train_indices]
            y_val_values = y_values[val_indices]
            compound_train_values = _slice_optional(compound_values, train_indices)
            concentration_train_values = _slice_optional(concentration_values, train_indices)
            compound_val_values = _slice_optional(compound_values, val_indices)
            concentration_val_values = _slice_optional(concentration_values, val_indices)
        else:
            X_train = X_tensor
            X_val = None
            y_train_values = y_values
            y_val_values = None
            compound_train_values = compound_values
            concentration_train_values = concentration_values
            compound_val_values = None
            concentration_val_values = None

        self._standardize_fit(X_train)
        X_train = self._standardize_apply(X_train)
        y_train = self._encode_labels(y_train_values)
        compound_train = self._encode_compound_labels(compound_train_values)
        concentration_train = self._encode_concentration_labels(concentration_train_values)

        if X_val is not None and y_val_values is not None:
            X_val = self._standardize_apply(X_val)
            y_val = self._encode_labels(y_val_values)
            compound_val = self._encode_compound_labels(compound_val_values)
            concentration_val = self._encode_concentration_labels(concentration_val_values)
        else:
            X_val = None
            y_val = None
            compound_val = None
            concentration_val = None

        return _PreparedData(
            X_train=X_train,
            y_train=y_train,
            compound_train=compound_train,
            concentration_train=concentration_train,
            X_val=X_val,
            y_val=y_val,
            compound_val=compound_val,
            concentration_val=concentration_val,
        )

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

    def _consistency_loss(self, st_logits: torch.Tensor, ts_logits: torch.Tensor) -> torch.Tensor:
        temperature = float(self.prototype_temperature)
        st_targets = torch.softmax(st_logits.detach() / temperature, dim=1)
        ts_targets = torch.softmax(ts_logits.detach() / temperature, dim=1)
        st_log_probs = torch.log_softmax(st_logits / temperature, dim=1)
        ts_log_probs = torch.log_softmax(ts_logits / temperature, dim=1)
        loss_st = -(st_targets * ts_log_probs).sum(dim=1).mean()
        loss_ts = -(ts_targets * st_log_probs).sum(dim=1).mean()
        return 0.5 * (loss_st + loss_ts)

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        criterion: nn.Module,
        compound_targets: torch.Tensor | None = None,
        concentration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        classification_loss = criterion(outputs["logits"], targets)
        consistency_loss = self._consistency_loss(outputs["st_prototypes"], outputs["ts_prototypes"])
        feature_loss = F.mse_loss(outputs["st_embedding"], outputs["ts_embedding"])
        total_loss = (
            classification_loss
            + float(self.consistency_weight) * consistency_loss
            + float(self.feature_weight) * feature_loss
        )
        compound_loss_value = 0.0
        concentration_loss_value = 0.0
        if compound_targets is not None and "compound_logits" in outputs:
            compound_loss = criterion(outputs["compound_logits"], compound_targets)
            total_loss = total_loss + float(self.compound_weight) * compound_loss
            compound_loss_value = float(compound_loss.item())
        if concentration_targets is not None and "concentration_logits" in outputs:
            concentration_loss = criterion(outputs["concentration_logits"], concentration_targets)
            total_loss = total_loss + float(self.concentration_weight) * concentration_loss
            concentration_loss_value = float(concentration_loss.item())
        return total_loss, {
            "classification_loss": float(classification_loss.item()),
            "consistency_loss": float(consistency_loss.item()),
            "feature_loss": float(feature_loss.item()),
            "compound_loss": compound_loss_value,
            "concentration_loss": concentration_loss_value,
        }

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        *,
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None = None,
        compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_compound_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
        validation_concentration_y: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    ) -> "CommutativeTransformerClassifier":
        prepared = self._prepare_training_data(
            X,
            y,
            validation_data,
            compound_y=compound_y,
            concentration_y=concentration_y,
            validation_compound_y=validation_compound_y,
            validation_concentration_y=validation_concentration_y,
        )
        self.model_ = self._build_model(num_classes=len(self.classes_))
        self.device_ = self._device()
        self.model_.to(self.device_)
        self.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])

        train_tensors: list[torch.Tensor] = [prepared.X_train, prepared.y_train]
        if prepared.compound_train is not None:
            train_tensors.append(prepared.compound_train)
        if prepared.concentration_train is not None:
            train_tensors.append(prepared.concentration_train)
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(
                TensorDataset(
                    *(
                        [prepared.X_val, prepared.y_val]
                        + ([prepared.compound_val] if prepared.compound_val is not None else [])
                        + ([prepared.concentration_val] if prepared.concentration_val is not None else [])
                    )
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )
            if prepared.X_val is not None and prepared.y_val is not None
            else None
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            if self.scheduler_patience is not None and self.scheduler_patience >= 0
            else None
        )

        history_rows: list[dict[str, float | int]] = []
        best_state = deepcopy(self.model_.state_dict())
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_total_loss_sum = 0.0
            train_classification_loss_sum = 0.0
            train_consistency_loss_sum = 0.0
            train_feature_loss_sum = 0.0
            train_compound_loss_sum = 0.0
            train_concentration_loss_sum = 0.0
            train_count = 0

            for batch in train_loader:
                X_batch = batch[0].to(self.device_, non_blocking=True)
                y_batch = batch[1].to(self.device_, non_blocking=True)
                compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_train is not None else None
                concentration_batch = (
                    batch[3 if prepared.compound_train is not None else 2].to(self.device_, non_blocking=True)
                    if prepared.concentration_train is not None
                    else None
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model_(X_batch)
                loss, loss_components = self._compute_losses(
                    outputs,
                    y_batch,
                    criterion,
                    compound_targets=compound_batch,
                    concentration_targets=concentration_batch,
                )
                loss.backward()
                optimizer.step()

                batch_size = int(X_batch.shape[0])
                train_total_loss_sum += float(loss.item()) * batch_size
                train_classification_loss_sum += loss_components["classification_loss"] * batch_size
                train_consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                train_feature_loss_sum += loss_components["feature_loss"] * batch_size
                train_compound_loss_sum += loss_components["compound_loss"] * batch_size
                train_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                train_count += batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_total_loss_sum / max(train_count, 1),
                "train_classification_loss": train_classification_loss_sum / max(train_count, 1),
                "train_consistency_loss": train_consistency_loss_sum / max(train_count, 1),
                "train_feature_loss": train_feature_loss_sum / max(train_count, 1),
                "train_compound_loss": train_compound_loss_sum / max(train_count, 1),
                "train_concentration_loss": train_concentration_loss_sum / max(train_count, 1),
            }

            if val_loader is not None:
                self.model_.eval()
                val_total_loss_sum = 0.0
                val_classification_loss_sum = 0.0
                val_consistency_loss_sum = 0.0
                val_feature_loss_sum = 0.0
                val_compound_loss_sum = 0.0
                val_concentration_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0].to(self.device_, non_blocking=True)
                        y_batch = batch[1].to(self.device_, non_blocking=True)
                        compound_batch = batch[2].to(self.device_, non_blocking=True) if prepared.compound_val is not None else None
                        concentration_batch = (
                            batch[3 if prepared.compound_val is not None else 2].to(self.device_, non_blocking=True)
                            if prepared.concentration_val is not None
                            else None
                        )
                        outputs = self.model_(X_batch)
                        loss, loss_components = self._compute_losses(
                            outputs,
                            y_batch,
                            criterion,
                            compound_targets=compound_batch,
                            concentration_targets=concentration_batch,
                        )
                        batch_size = int(X_batch.shape[0])
                        val_total_loss_sum += float(loss.item()) * batch_size
                        val_classification_loss_sum += loss_components["classification_loss"] * batch_size
                        val_consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                        val_feature_loss_sum += loss_components["feature_loss"] * batch_size
                        val_compound_loss_sum += loss_components["compound_loss"] * batch_size
                        val_concentration_loss_sum += loss_components["concentration_loss"] * batch_size
                        val_count += batch_size
                row["val_loss"] = val_total_loss_sum / max(val_count, 1)
                row["val_classification_loss"] = val_classification_loss_sum / max(val_count, 1)
                row["val_consistency_loss"] = val_consistency_loss_sum / max(val_count, 1)
                row["val_feature_loss"] = val_feature_loss_sum / max(val_count, 1)
                row["val_compound_loss"] = val_compound_loss_sum / max(val_count, 1)
                row["val_concentration_loss"] = val_concentration_loss_sum / max(val_count, 1)
                metric = float(row["val_loss"])
            else:
                metric = float(row["train_loss"])

            improved = metric < (best_metric - float(self.early_stopping_min_delta))
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = deepcopy(self.model_.state_dict())
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                scheduler.step(metric)

            history_rows.append(row)
            if self.verbose:
                elapsed = time.perf_counter() - training_start
                avg_epoch_seconds = elapsed / epoch
                eta = _format_eta(avg_epoch_seconds * (self.epochs - epoch))
                current_lr = optimizer.param_groups[0]["lr"]
                if "val_loss" in row:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"val_loss={row['val_loss']:.4f} lr={current_lr:.2e} eta={eta}"
                    )
                else:
                    print(
                        f"epoch {epoch:03d}/{self.epochs:03d} train_loss={row['train_loss']:.4f} "
                        f"lr={current_lr:.2e} eta={eta}"
                    )

            if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(
                        f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                        f"best_metric={best_metric:.4f}"
                    )
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = pd.DataFrame(history_rows)
        self.best_epoch_ = int(best_epoch) if best_epoch else int(len(history_rows))
        self.best_metric_ = float(best_metric)
        return self

    def _forward_batches(self, X: torch.Tensor | np.ndarray, *, output_key: str) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                batch_out = self.model_(X_batch)[output_key]
                outputs.append(batch_out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _forward_feature_batches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        collected = {
            "st_embedding": [],
            "ts_embedding": [],
            "embedding": [],
            "st_prototypes": [],
            "ts_prototypes": [],
            "logits": [],
            "compound_logits": [],
            "concentration_logits": [],
        }
        self.model_.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                outputs = self.model_(X_batch)
                for key in list(collected):
                    if key in outputs:
                        collected[key].append(outputs[key].detach().cpu().numpy())
                    else:
                        collected.pop(key, None)
        return {key: np.concatenate(values, axis=0) for key, values in collected.items()}

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._forward_feature_batches(X)
        result = {"action": torch.softmax(torch.from_numpy(outputs["logits"]), dim=1).numpy()}
        if "compound_logits" in outputs:
            result["compound"] = torch.softmax(torch.from_numpy(outputs["compound_logits"]), dim=1).numpy()
        if "concentration_logits" in outputs:
            result["concentration"] = torch.softmax(torch.from_numpy(outputs["concentration_logits"]), dim=1).numpy()
        return result

    def predict(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        probabilities = self.predict_proba(X)
        predictions = {"action": self.classes_[probabilities["action"].argmax(axis=1)]}
        if "compound" in probabilities:
            predictions["compound"] = self.compound_classes_[probabilities["compound"].argmax(axis=1)]
        if "concentration" in probabilities:
            predictions["concentration"] = self.concentration_classes_[probabilities["concentration"].argmax(axis=1)]
        return predictions

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return self._forward_batches(X, output_key="embedding")

    def transform_branches(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._forward_feature_batches(X)
        return {
            "st_embedding": outputs["st_embedding"],
            "ts_embedding": outputs["ts_embedding"],
            "embedding": outputs["embedding"],
        }

    def evaluate_loss_components(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
    ) -> dict[str, float]:
        check_is_fitted(self, ["model_", "classes_", "input_mean_", "input_std_"])
        X_tensor = self._standardize_apply(_ensure_tensor_5d(X))
        y_values = _ensure_labels_1d(y)
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")
        y_tensor = self._encode_labels(y_values)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        total_loss_sum = 0.0
        classification_loss_sum = 0.0
        consistency_loss_sum = 0.0
        feature_loss_sum = 0.0
        count = 0
        self.model_.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                y_batch = y_batch.to(self.device_, non_blocking=True)
                outputs = self.model_(X_batch)
                loss, loss_components = self._compute_losses(outputs, y_batch, criterion)
                batch_size = int(X_batch.shape[0])
                total_loss_sum += float(loss.item()) * batch_size
                classification_loss_sum += loss_components["classification_loss"] * batch_size
                consistency_loss_sum += loss_components["consistency_loss"] * batch_size
                feature_loss_sum += loss_components["feature_loss"] * batch_size
                count += batch_size
        return {
            "loss": total_loss_sum / max(count, 1),
            "classification_loss": classification_loss_sum / max(count, 1),
            "consistency_loss": consistency_loss_sum / max(count, 1),
            "feature_loss": feature_loss_sum / max(count, 1),
        }

    def score(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | Sequence[int]) -> float:
        y_true = _ensure_labels_1d(y)
        y_pred = self.predict(X)["action"]
        return float(accuracy_score(y_true, y_pred))


def augment_training_tensors_with_rotations(
    tensors: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray | Sequence[int],
    *,
    metadata: pd.DataFrame | None = None,
    num_random_rotations: int,
    rotation_range_degrees: float = 5.0,
    random_state: int = 0,
) -> tuple[torch.Tensor, np.ndarray, pd.DataFrame | None]:
    if num_random_rotations < 0:
        raise ValueError("num_random_rotations must be non-negative")

    tensor_values = _ensure_tensor_5d(tensors)
    label_values = _ensure_labels_1d(labels)
    if len(tensor_values) != len(label_values):
        raise ValueError("tensors and labels must have the same number of samples")

    metadata_df = metadata.reset_index(drop=True).copy() if metadata is not None else None
    if metadata_df is not None and len(metadata_df) != len(tensor_values):
        raise ValueError("metadata length must match number of tensors")

    rng = np.random.default_rng(random_state)
    augmented_tensors: list[torch.Tensor] = []
    augmented_labels: list[int] = []
    augmented_rows: list[dict[str, object]] | None = [] if metadata_df is not None else None

    for index, base_tensor in enumerate(tensor_values):
        augmented_tensors.append(base_tensor)
        augmented_labels.append(int(label_values[index]))
        if metadata_df is not None and augmented_rows is not None:
            base_row = metadata_df.iloc[index].to_dict()
            if "augmentation_index" in base_row:
                base_row["augmentation_index"] = 0
            if "rotation_degrees" in base_row:
                base_row["rotation_degrees"] = 0.0
            augmented_rows.append(base_row)

        for augmentation_index in range(1, num_random_rotations + 1):
            angle = float(rng.uniform(-rotation_range_degrees, rotation_range_degrees))
            augmented_tensors.append(rotate_tensor_xy(base_tensor, angle))
            augmented_labels.append(int(label_values[index]))
            if metadata_df is not None and augmented_rows is not None:
                row = metadata_df.iloc[index].to_dict()
                row["augmentation_index"] = augmentation_index
                row["rotation_degrees"] = angle
                augmented_rows.append(row)

    augmented_metadata = pd.DataFrame(augmented_rows) if augmented_rows is not None else None
    return torch.stack(augmented_tensors, dim=0), np.asarray(augmented_labels, dtype=int), augmented_metadata


def split_labeled_tensor_dataset_by_instance(
    dataset: dict[str, object],
    *,
    holdout_fraction: float,
    validation_fraction_within_train: float,
    random_state: int = 0,
) -> TensorDatasetSplits:
    """Split a persisted labeled tensor dataset by source instance id.

    The split groups examples by ``original_instance_id`` so any derived views
    of the same source tensor can be kept in the same partition and avoid leakage.
    """
    metadata_all = dataset["metadata"].reset_index(drop=True).copy()
    if "original_instance_id" not in metadata_all.columns:
        # Backward-compatible fallback for older dataset artifacts saved before original_instance_id existed.
        metadata_all["original_instance_id"] = pd.factorize(
            metadata_all[["label", "image_condition_dir", "is_control"]].astype(str).agg("|".join, axis=1)
        )[0]

    instance_df = (
        metadata_all[["original_instance_id", "label"]]
        .drop_duplicates(subset=["original_instance_id"])
        .reset_index(drop=True)
    )
    instance_ids = instance_df["original_instance_id"].to_numpy()
    instance_labels = instance_df["label"].to_numpy()

    try:
        train_val_instance_ids, holdout_instance_ids = train_test_split(
            instance_ids,
            test_size=holdout_fraction,
            random_state=random_state,
            stratify=instance_labels,
        )
    except ValueError:
        train_val_instance_ids, holdout_instance_ids = train_test_split(
            instance_ids,
            test_size=holdout_fraction,
            random_state=random_state,
            stratify=None,
        )

    train_val_instance_df = instance_df[instance_df["original_instance_id"].isin(train_val_instance_ids)].reset_index(drop=True)
    try:
        train_instance_ids, val_instance_ids = train_test_split(
            train_val_instance_df["original_instance_id"].to_numpy(),
            test_size=validation_fraction_within_train,
            random_state=random_state,
            stratify=train_val_instance_df["label"].to_numpy(),
        )
    except ValueError:
        train_instance_ids, val_instance_ids = train_test_split(
            train_val_instance_df["original_instance_id"].to_numpy(),
            test_size=validation_fraction_within_train,
            random_state=random_state,
            stratify=None,
        )

    train_indices = metadata_all.index[metadata_all["original_instance_id"].isin(train_instance_ids)].to_numpy(dtype=np.int64, copy=True)
    val_indices = metadata_all.index[metadata_all["original_instance_id"].isin(val_instance_ids)].to_numpy(dtype=np.int64, copy=True)
    holdout_indices = metadata_all.index[metadata_all["original_instance_id"].isin(holdout_instance_ids)].to_numpy(dtype=np.int64, copy=True)

    tensors = dataset["tensors"]
    labels = dataset["labels"]
    compound_labels = dataset.get("compound_labels")
    concentration_labels = dataset.get("concentration_labels")
    if not isinstance(tensors, torch.Tensor):
        raise TypeError("dataset['tensors'] must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("dataset['labels'] must be a torch.Tensor")
    if compound_labels is not None and not isinstance(compound_labels, torch.Tensor):
        raise TypeError("dataset['compound_labels'] must be a torch.Tensor")
    if concentration_labels is not None and not isinstance(concentration_labels, torch.Tensor):
        raise TypeError("dataset['concentration_labels'] must be a torch.Tensor")

    return TensorDatasetSplits(
        metadata_all=metadata_all,
        train_indices=train_indices,
        val_indices=val_indices,
        holdout_indices=holdout_indices,
        train_instance_ids=np.asarray(train_instance_ids),
        val_instance_ids=np.asarray(val_instance_ids),
        holdout_instance_ids=np.asarray(holdout_instance_ids),
        X_train_base=tensors[train_indices],
        y_train_base=labels[train_indices],
        compound_train_base=compound_labels[train_indices] if compound_labels is not None else None,
        concentration_train_base=concentration_labels[train_indices] if concentration_labels is not None else None,
        metadata_train_base=metadata_all.iloc[train_indices].reset_index(drop=True),
        X_val=tensors[val_indices],
        y_val=labels[val_indices].detach().cpu().numpy(),
        compound_val=compound_labels[val_indices].detach().cpu().numpy() if compound_labels is not None else None,
        concentration_val=concentration_labels[val_indices].detach().cpu().numpy()
        if concentration_labels is not None
        else None,
        metadata_val=metadata_all.iloc[val_indices].reset_index(drop=True),
        X_holdout=tensors[holdout_indices],
        y_holdout=labels[holdout_indices].detach().cpu().numpy(),
        compound_holdout=compound_labels[holdout_indices].detach().cpu().numpy() if compound_labels is not None else None,
        concentration_holdout=concentration_labels[holdout_indices].detach().cpu().numpy()
        if concentration_labels is not None
        else None,
        metadata_holdout=metadata_all.iloc[holdout_indices].reset_index(drop=True),
    )


def plot_training_history(
    history: pd.DataFrame | TimeChannel3DCNNClassifier | CommutativeCNNClassifier | CommutativeTransformerClassifier,
    *,
    ax=None,
    title: str = "Training history",
):
    if isinstance(history, (TimeChannel3DCNNClassifier, CommutativeCNNClassifier, CommutativeTransformerClassifier)):
        check_is_fitted(history, ["history_"])
        history_df = history.history_
    else:
        history_df = history

    if history_df.empty:
        raise ValueError("history is empty")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def build_classification_reports(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    y_proba: np.ndarray | None = None,
    class_labels: Sequence[int] | None = None,
    label_map: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length")

    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    class_labels = list(class_labels)
    class_names = [label_map.get(int(label), str(label)) for label in class_labels] if label_map else [str(label) for label in class_labels]

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    per_class_df = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()
    per_class_df.index.name = "class"

    summary_metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "n_samples": int(len(y_true_arr)),
    }

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim != 2:
            raise ValueError("y_proba must have shape (n_samples, n_classes)")
        if y_proba_arr.shape[0] != len(y_true_arr):
            raise ValueError("y_proba must have the same number of rows as y_true")
        if y_proba_arr.shape[1] != len(class_labels):
            raise ValueError("y_proba column count must match the number of class labels")

        y_true_bin = label_binarize(y_true_arr, classes=class_labels)
        if len(class_labels) == 2:
            positive_scores = y_proba_arr[:, 1]
            if y_true_bin.shape[1] == 1:
                positive_targets = y_true_bin[:, 0]
            else:
                positive_targets = y_true_bin[:, 1]
            summary_metrics["roc_auc"] = float(roc_auc_score(positive_targets, positive_scores))
            summary_metrics["average_precision"] = float(
                average_precision_score(positive_targets, positive_scores)
            )
        else:
            summary_metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, y_proba_arr, multi_class="ovr", average="macro")
            )
            summary_metrics["average_precision_macro"] = float(
                average_precision_score(y_true_bin, y_proba_arr, average="macro")
            )

    summary_df = pd.DataFrame([summary_metrics]).T.rename(columns={0: "value"})
    return per_class_df, summary_df


def build_multitask_classification_reports(
    y_true: dict[str, Iterable[int]],
    y_pred: dict[str, Iterable[int]],
    *,
    y_proba: dict[str, np.ndarray] | None = None,
    class_labels: dict[str, Sequence[int]] | None = None,
    label_maps: dict[str, dict[int, str]] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target, target_y_true in y_true.items():
        if target not in y_pred:
            raise KeyError(f"Missing predictions for target {target!r}")
        target_labels = class_labels.get(target) if class_labels is not None else None
        target_label_map = label_maps.get(target) if label_maps is not None else None
        target_proba = y_proba.get(target) if y_proba is not None else None
        results[target] = build_classification_reports(
            target_y_true,
            y_pred[target],
            y_proba=target_proba,
            class_labels=target_labels,
            label_map=target_label_map,
        )
    return results


def plot_confusion_matrices(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    class_labels: Sequence[int] | None = None,
    label_map: dict[int, str] | None = None,
    axes=None,
    cmap: str = "Blues",
):
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if class_labels is None:
        class_labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    class_labels = list(class_labels)
    tick_labels = [label_map.get(int(label), str(label)) for label in class_labels] if label_map else [str(label) for label in class_labels]

    cm_abs = confusion_matrix(y_true_arr, y_pred_arr, labels=class_labels)
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_frac = np.divide(cm_abs, row_sums, out=np.zeros_like(cm_abs, dtype=float), where=row_sums > 0)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes[0].figure

    panels = [
        (axes[0], cm_abs, "Confusion matrix (counts)", "d"),
        (axes[1], cm_frac, "Confusion matrix (row fractions)", ".2f"),
    ]
    for ax, matrix, title, fmt in panels:
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = format(matrix[row_index, column_index], fmt)
                ax.text(column_index, row_index, value, ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, axes, cm_abs, cm_frac
