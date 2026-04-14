from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        return self.norm(X)


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
