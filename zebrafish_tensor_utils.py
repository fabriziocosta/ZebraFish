from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from tifffile import TiffFile, memmap as tiff_memmap


CACHE_VERSION = 1
TENSOR_CACHE_DIR = Path(__file__).resolve().parent / ".tensor_cache"


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


def load_tiff_as_tzyx(
    path: str | Path,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
) -> torch.Tensor:
    path = Path(path)
    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = getattr(series, "axes", "")
        try:
            arr = np.asarray(tiff_memmap(path, series=0, mode="r"))
        except Exception:
            arr = np.asarray(series.asarray())

    if output_size is not None:
        z_keep = output_size[1]
        if z_keep is not None:
            if "Z" in axes:
                z_axis = axes.index("Z")
                z_indices = select_evenly_spaced_indices(int(arr.shape[z_axis]), int(z_keep))
                arr = np.take(arr, z_indices, axis=z_axis)
            elif int(z_keep) != 1:
                raise ValueError(f"Requested Z={z_keep} from {path}, but source axes are {axes!r}")

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


def build_tensor_cache_key(
    condition_dir: str | Path,
    timepoint_files: list[Path],
    output_size: tuple[int | None, int | None, int | None, int | None] | None,
    normalize_global_drift: bool,
    loess_frac: float,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "condition_dir": str(Path(condition_dir).resolve()),
        "output_size": output_size,
        "normalize_global_drift": normalize_global_drift,
        "loess_frac": loess_frac,
        "files": [
            {
                "path": str(path.resolve()),
                "mtime_ns": path.stat().st_mtime_ns,
                "size": path.stat().st_size,
            }
            for path in timepoint_files
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cached_tensor(cache_key: str) -> torch.Tensor | None:
    cache_path = TENSOR_CACHE_DIR / f"{cache_key}.pt"
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu")


def save_cached_tensor(cache_key: str, tensor: torch.Tensor) -> None:
    TENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TENSOR_CACHE_DIR / f"{cache_key}.pt"
    torch.save(tensor.cpu(), cache_path)


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


def loess_smooth_1d(values: np.ndarray, frac: float = 0.25) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return values.copy()
    if n == 1:
        return values.copy()
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    x = np.arange(n, dtype=float)
    span = max(3, int(np.ceil(frac * n)))
    span = min(span, n)
    smoothed = np.empty(n, dtype=float)

    for i in range(n):
        distances = np.abs(x - x[i])
        bandwidth = np.partition(distances, span - 1)[span - 1]
        if bandwidth == 0:
            smoothed[i] = values[i]
            continue

        scaled = distances / bandwidth
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)
        if not np.any(weights):
            smoothed[i] = values[i]
            continue

        x_centered = x - x[i]
        design = np.column_stack([np.ones(n, dtype=float), x_centered])
        xtwx = design.T @ (weights[:, None] * design)
        xtwy = design.T @ (weights * values)
        beta = np.linalg.pinv(xtwx) @ xtwy
        smoothed[i] = beta[0]

    return smoothed


def normalize_global_intensity_drift(
    tensor: torch.Tensor,
    loess_frac: float = 0.25,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor with shape T x Z x Y x X, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] < 2:
        return tensor

    reference_dtype = tensor.dtype
    work_tensor = tensor.to(torch.float32)
    global_mean = work_tensor.mean(dim=(1, 2, 3)).detach().cpu().numpy()
    smooth_mean = loess_smooth_1d(global_mean, frac=loess_frac)
    drift = smooth_mean - smooth_mean.mean()
    drift_tensor = torch.tensor(drift, dtype=work_tensor.dtype, device=work_tensor.device).view(-1, 1, 1, 1)
    normalized = work_tensor - drift_tensor
    return normalized.to(reference_dtype) if torch.is_floating_point(tensor) else normalized


def load_image_condition_tensor(
    condition_dir: str | Path,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
    normalize_global_drift: bool = True,
    loess_frac: float = 0.25,
    use_cache: bool = True,
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

    cache_key = build_tensor_cache_key(
        condition_dir=condition_dir,
        timepoint_files=timepoint_files,
        output_size=output_size,
        normalize_global_drift=normalize_global_drift,
        loess_frac=loess_frac,
    )
    if use_cache:
        cached_tensor = load_cached_tensor(cache_key)
        if cached_tensor is not None:
            return cached_tensor

    tensors = [load_tiff_as_tzyx(path, output_size=output_size) for path in timepoint_files]
    reference_shape = tensors[0].shape[1:]
    mismatched = [str(path) for path, tensor in zip(timepoint_files, tensors) if tensor.shape[1:] != reference_shape]
    if mismatched:
        raise ValueError(
            f"Inconsistent Z/Y/X shapes in {condition_dir}; expected {reference_shape}, mismatched files: {mismatched}"
        )

    tensor = torch.cat(tensors, dim=0)
    if output_size is not None:
        tensor = downsample_tzyx(tensor, output_size=(None, None, output_size[2], output_size[3]))
    if normalize_global_drift:
        tensor = normalize_global_intensity_drift(tensor, loess_frac=loess_frac)
    if use_cache:
        save_cached_tensor(cache_key, tensor)
    return tensor
