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
    region_grid: tuple[int, int, int] = (1, 2, 2)
    time_bins: int = 8
    frequency_bins: int = 4
    local_stats: tuple[str, ...] = ("mean", "std")

    def __post_init__(self) -> None:
        if len(self.region_grid) != 3 or any(size <= 0 for size in self.region_grid):
            raise ValueError("region_grid must contain three positive integers")
        if self.time_bins <= 0:
            raise ValueError("time_bins must be positive")
        if self.frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")
        if tuple(self.local_stats) != ("mean", "std"):
            raise ValueError("local_stats must be ('mean', 'std')")

    @property
    def num_regions(self) -> int:
        return int(self.region_grid[0] * self.region_grid[1] * self.region_grid[2])

    @property
    def num_region_pairs(self) -> int:
        return max(self.num_regions * (self.num_regions - 1) // 2, 1)

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "local": (self.num_regions, self.time_bins, len(self.local_stats)),
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


def _validate_coarse_bins(X: torch.Tensor, probe_spec: ProbeSpec) -> None:
    _, n_timepoints, z_size, y_size, x_size = X.shape
    grid_z, grid_y, grid_x = probe_spec.region_grid
    if probe_spec.time_bins > n_timepoints:
        raise ValueError("time_bins cannot exceed the number of timepoints")
    if grid_z > z_size or grid_y > y_size or grid_x > x_size:
        raise ValueError("region_grid cannot exceed the spatial tensor shape")


def _coarse_spatiotemporal_stats(X: torch.Tensor, probe_spec: ProbeSpec) -> torch.Tensor:
    _validate_coarse_bins(X, probe_spec)
    grid_z, grid_y, grid_x = probe_spec.region_grid
    region_stats: list[torch.Tensor] = []
    for z_chunk in torch.tensor_split(X, grid_z, dim=2):
        for y_chunk in torch.tensor_split(z_chunk, grid_y, dim=3):
            for x_chunk in torch.tensor_split(y_chunk, grid_x, dim=4):
                stats_by_time: list[torch.Tensor] = []
                for time_chunk in torch.tensor_split(x_chunk, probe_spec.time_bins, dim=1):
                    flattened = time_chunk.flatten(start_dim=1)
                    mean = flattened.mean(dim=1)
                    std = flattened.std(dim=1, unbiased=False)
                    stats_by_time.append(torch.stack((mean, std), dim=-1))
                region_stats.append(torch.stack(stats_by_time, dim=1))
    return torch.stack(region_stats, dim=1)


def build_probe_targets(X: torch.Tensor, probe_spec: ProbeSpec) -> dict[str, torch.Tensor]:
    if X.ndim != 5:
        raise ValueError(f"Expected X with shape (N, T, Z, Y, X), got {tuple(X.shape)}")
    n_samples = X.shape[0]
    local = _coarse_spatiotemporal_stats(X, probe_spec)
    region_time = local[..., 0]
    derivative = F.pad(region_time.diff(dim=-1), (1, 0))

    frequency = torch.fft.rfft(region_time - region_time.mean(dim=-1, keepdim=True), dim=-1).abs()
    if frequency.shape[-1] >= probe_spec.frequency_bins:
        frequency = frequency[..., : probe_spec.frequency_bins]
    else:
        frequency = F.pad(frequency, (0, probe_spec.frequency_bins - frequency.shape[-1]))

    centered = region_time - region_time.mean(dim=-1, keepdim=True)
    normalized = centered / (centered.square().mean(dim=-1, keepdim=True).sqrt() + 1e-6)
    corr = torch.einsum("brt,bst->brs", normalized, normalized) / max(probe_spec.time_bins, 1)
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
    full: bool = False,
) -> dict[str, torch.Tensor]:
    probability = float(observe_probability)
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("observe_probability must be in (0, 1]")
    masks: dict[str, torch.Tensor] = {}
    for probe_type, target in targets.items():
        if full:
            masks[probe_type] = torch.ones_like(target)
            continue
        mask = (torch.rand(target.shape, device=target.device, dtype=target.dtype) < probability).to(dtype=target.dtype)
        if not bool(mask.any().item()):
            flat = mask.reshape(-1)
            flat[torch.randint(flat.numel(), (1,), device=target.device)] = 1
        masks[probe_type] = mask
    return masks


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
