from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch
from tifffile import TiffFile


def _squeeze_array_and_axes(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = axes or ""
    while len(axes) < arr.ndim:
        axes += "?"
    while len(axes) > arr.ndim:
        axes = axes[:arr.ndim]

    for axis_index in range(arr.ndim - 1, -1, -1):
        if arr.shape[axis_index] == 1:
            arr = np.squeeze(arr, axis=axis_index)
            if axis_index < len(axes):
                axes = axes[:axis_index] + axes[axis_index + 1 :]
    return arr, axes


def load_tiff_as_tzyx(path: str | Path) -> torch.Tensor:
    path = Path(path)
    with TiffFile(path) as tif:
        series = tif.series[0]
        arr = np.asarray(series.asarray())
        axes = getattr(series, "axes", "")

    arr, axes = _squeeze_array_and_axes(arr, axes)
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis, :, :]
        axes = "ZYX"

    unsupported_axes = [axis for axis in axes if axis not in {"T", "Z", "Y", "X"}]
    if unsupported_axes:
        raise ValueError(f"Unsupported non-singleton axes {unsupported_axes} in {path} with axes={axes!r}")

    source_axes = list(axes)
    target_axes = [axis for axis in "TZYX" if axis in source_axes]
    transpose_order = [source_axes.index(axis) for axis in target_axes]
    arr = np.transpose(arr, axes=transpose_order)

    for axis in "TZYX":
        if axis not in target_axes:
            insert_at = "TZYX".index(axis)
            arr = np.expand_dims(arr, axis=insert_at)

    return torch.from_numpy(np.ascontiguousarray(arr))


def timepoint_sort_key(path: Path):
    match = re.search(r"TL(\d+)", path.name)
    if match:
        return int(match.group(1))
    return path.name


def list_timepoint_files(condition_dir: str | Path) -> list[Path]:
    condition_dir = Path(condition_dir)
    direct_files = sorted(condition_dir.glob("*.tif*"), key=timepoint_sort_key)
    if direct_files:
        return direct_files
    return sorted(condition_dir.rglob("*.tif*"), key=timepoint_sort_key)


def select_evenly_spaced_indices(n_total: int, n_keep: int) -> list[int]:
    if n_total <= 0:
        return []
    if n_keep <= 0:
        raise ValueError(f"Requested n_keep must be positive, got {n_keep}")
    if n_keep > n_total:
        raise ValueError(f"Cannot downsample to {n_keep} items from source size {n_total}")
    if n_keep == n_total:
        return list(range(n_total))
    if n_keep == 1:
        return [n_total // 2]

    indices = np.rint(np.linspace(0, n_total - 1, n_keep)).astype(int).tolist()
    indices[0] = 0
    indices[-1] = n_total - 1

    if n_keep % 2 == 1:
        indices[n_keep // 2] = n_total // 2

    deduped = []
    for index in indices:
        if not deduped or deduped[-1] != index:
            deduped.append(index)

    while len(deduped) < n_keep:
        for candidate in range(n_total):
            if candidate not in deduped:
                deduped.append(candidate)
            if len(deduped) == n_keep:
                break

    return sorted(deduped)


def downsample_tzyx(
    tensor: torch.Tensor,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
) -> torch.Tensor:
    if output_size is None:
        return tensor
    if len(output_size) != 4:
        raise ValueError(f"output_size must have 4 elements (T, Z, Y, X), got {output_size}")

    result = tensor
    for dim, n_keep in enumerate(output_size):
        if n_keep is None:
            continue
        indices = select_evenly_spaced_indices(int(result.shape[dim]), int(n_keep))
        index_tensor = torch.tensor(indices, dtype=torch.long, device=result.device)
        result = torch.index_select(result, dim=dim, index=index_tensor)
    return result


def load_image_condition_tensor(
    condition_dir: str | Path,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
) -> torch.Tensor:
    condition_dir = Path(condition_dir)
    timepoint_files = list_timepoint_files(condition_dir)
    if not timepoint_files:
        raise FileNotFoundError(f"No TIFF files found under {condition_dir}")

    if output_size is not None:
        time_size = output_size[0]
        if time_size is not None:
            time_indices = select_evenly_spaced_indices(len(timepoint_files), int(time_size))
            timepoint_files = [timepoint_files[index] for index in time_indices]

    tensors = [load_tiff_as_tzyx(path) for path in timepoint_files]
    reference_shape = tensors[0].shape[1:]
    mismatched = [str(path) for path, tensor in zip(timepoint_files, tensors) if tensor.shape[1:] != reference_shape]
    if mismatched:
        raise ValueError(
            f"Inconsistent Z/Y/X shapes in {condition_dir}; expected {reference_shape}, mismatched files: {mismatched}"
        )

    tensor = torch.cat(tensors, dim=0)
    if output_size is None:
        return tensor
    return downsample_tzyx(tensor, output_size=(None, output_size[1], output_size[2], output_size[3]))
