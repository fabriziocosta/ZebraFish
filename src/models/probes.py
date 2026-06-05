from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn


PROBE_TYPES: tuple[str, ...] = ("local", "region_time", "derivative", "frequency", "correlation")


@dataclass(frozen=True)
class ProbeSpec:
    local_count: int = 32
    region_grid: tuple[int, int, int] = (1, 2, 2)
    time_bins: int = 8
    frequency_bins: int = 4

    def __post_init__(self) -> None:
        if self.local_count <= 0:
            raise ValueError("local_count must be positive")
        if len(self.region_grid) != 3 or any(size <= 0 for size in self.region_grid):
            raise ValueError("region_grid must contain three positive integers")
        if self.time_bins <= 0:
            raise ValueError("time_bins must be positive")
        if self.frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")

    @property
    def num_regions(self) -> int:
        return int(self.region_grid[0] * self.region_grid[1] * self.region_grid[2])

    @property
    def num_region_pairs(self) -> int:
        return max(self.num_regions * (self.num_regions - 1) // 2, 1)

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "local": (self.local_count,),
            "region_time": (self.num_regions, self.time_bins),
            "derivative": (self.num_regions, self.time_bins),
            "frequency": (self.num_regions, self.frequency_bins),
            "correlation": (self.num_region_pairs,),
        }


class ProbeDecoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        probe_spec: ProbeSpec,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or embedding_dim)
        self.probe_spec = probe_spec
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.heads = nn.ModuleDict(
            {
                probe_type: nn.Linear(hidden, _numel(shape))
                for probe_type, shape in self.probe_spec.shapes.items()
            }
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(embedding)
        return {
            probe_type: self.heads[probe_type](features).reshape(embedding.shape[0], *shape)
            for probe_type, shape in self.probe_spec.shapes.items()
        }


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _sample_axis_indices(size: int, count: int, *, device: torch.device) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    if size <= 1:
        return torch.zeros(count, dtype=torch.long, device=device)
    return torch.linspace(0, size - 1, steps=count, device=device).round().long()


def _region_traces(X: torch.Tensor, probe_spec: ProbeSpec) -> torch.Tensor:
    n_samples, n_timepoints, z_size, y_size, x_size = X.shape
    pooled = F.adaptive_avg_pool3d(
        X.reshape(n_samples * n_timepoints, 1, z_size, y_size, x_size),
        output_size=probe_spec.region_grid,
    )
    return pooled.reshape(n_samples, n_timepoints, probe_spec.num_regions)


def build_probe_targets(X: torch.Tensor, probe_spec: ProbeSpec) -> dict[str, torch.Tensor]:
    if X.ndim != 5:
        raise ValueError(f"Expected X with shape (N, T, Z, Y, X), got {tuple(X.shape)}")
    n_samples, n_timepoints, z_size, y_size, x_size = X.shape
    time_idx = _sample_axis_indices(n_timepoints, probe_spec.local_count, device=X.device)
    z_idx = _sample_axis_indices(z_size, probe_spec.local_count, device=X.device)
    y_idx = _sample_axis_indices(y_size, probe_spec.local_count, device=X.device)
    x_idx = _sample_axis_indices(x_size, probe_spec.local_count, device=X.device)
    local = X[:, time_idx, z_idx, y_idx.roll(1), x_idx.roll(2)]

    traces = _region_traces(X, probe_spec)
    region_time = F.adaptive_avg_pool1d(
        traces.transpose(1, 2),
        output_size=probe_spec.time_bins,
    )
    derivative = F.pad(region_time.diff(dim=-1), (1, 0))

    frequency = torch.fft.rfft(traces - traces.mean(dim=1, keepdim=True), dim=1).abs().transpose(1, 2)
    if frequency.shape[-1] >= probe_spec.frequency_bins:
        frequency = frequency[..., : probe_spec.frequency_bins]
    else:
        frequency = F.pad(frequency, (0, probe_spec.frequency_bins - frequency.shape[-1]))

    centered = traces - traces.mean(dim=1, keepdim=True)
    normalized = centered / (centered.square().mean(dim=1, keepdim=True).sqrt() + 1e-6)
    corr = torch.einsum("btr,bts->brs", normalized, normalized) / max(n_timepoints, 1)
    pair_indices = list(combinations(range(probe_spec.num_regions), 2))
    if pair_indices:
        first = torch.tensor([pair[0] for pair in pair_indices], device=X.device)
        second = torch.tensor([pair[1] for pair in pair_indices], device=X.device)
        correlation = corr[:, first, second]
    else:
        correlation = torch.zeros(n_samples, 1, device=X.device, dtype=X.dtype)

    return {
        "local": local,
        "region_time": region_time,
        "derivative": derivative,
        "frequency": frequency,
        "correlation": correlation,
    }


def build_probe_masks(
    targets: Mapping[str, torch.Tensor],
    *,
    observe_probability: float,
) -> dict[str, torch.Tensor]:
    probability = float(observe_probability)
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("observe_probability must be in (0, 1]")
    return {
        probe_type: (torch.rand(target.shape, device=target.device, dtype=target.dtype) < probability).to(dtype=target.dtype)
        for probe_type, target in targets.items()
    }


def masked_probe_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    alpha_weights: Mapping[str, float],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_probe: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for probe_type in PROBE_TYPES:
        pred = predictions[probe_type]
        target = targets[probe_type]
        mask = masks[probe_type].to(dtype=pred.dtype)
        loss = (mask * (pred - target).square()).sum() / (mask.sum() + eps)
        weighted = float(alpha_weights.get(probe_type, 1.0)) * loss
        total = weighted if total is None else total + weighted
        per_probe[probe_type] = loss
    if total is None:
        raise ValueError("No probe losses were computed")
    return total, per_probe
