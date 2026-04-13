from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.common import _as_tuple


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
        outputs = {"embedding": embedding, "logits": self.classifier(embedding)}
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
        patch_sequences = pooled.permute(0, 2, 3, 4, 1).reshape(
            n_samples * patch_grid_z * patch_grid_y * patch_grid_x,
            1,
            n_timepoints,
        )

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
