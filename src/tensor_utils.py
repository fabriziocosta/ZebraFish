from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tifffile import TiffFile, memmap as tiff_memmap
from sklearn.decomposition import PCA
from sklearn.svm import SVC


CACHE_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_CACHE_DIR = PROJECT_ROOT / ".tensor_cache"
TIFF_CACHE_DIR = PROJECT_ROOT / ".tiff_cache"
DATASET_CACHE_DIR = PROJECT_ROOT / ".dataset_cache"


def _format_eta(seconds: float) -> str:
    remaining = max(int(round(seconds)), 0)
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
    use_local_pages = False
    try:
        with TiffFile(path) as tif:
            use_local_pages = len(tif.pages) > 1
            if use_local_pages:
                z_keep = output_size[1] if output_size is not None else None
                z_indices = (
                    select_evenly_spaced_indices(len(tif.pages), int(z_keep))
                    if z_keep is not None
                    else list(range(len(tif.pages)))
                )
                page_arrays = []
                for page_index in z_indices:
                    memmap_exc: Exception | None = None
                    page_exc: Exception | None = None
                    try:
                        page_arr = np.asarray(tiff_memmap(path, page=page_index, mode="r"))
                    except Exception as exc:
                        memmap_exc = exc
                        try:
                            page_arr = np.asarray(tif.pages[page_index].asarray())
                        except Exception as exc2:
                            page_exc = exc2
                            try:
                                page_arr = np.asarray(tif.asarray(key=page_index))
                            except Exception as exc3:
                                raise RuntimeError(
                                    f"Failed to read TIFF page {page_index} from {path}. "
                                    f"memmap error: {memmap_exc!r}; "
                                    f"page.asarray error: {page_exc!r}; "
                                    f"tif.asarray(key=...) error: {exc3!r}"
                                ) from exc3
                    page_arrays.append(page_arr)
                arr = np.stack(page_arrays, axis=0)
                axes = "ZYX"
            else:
                series = tif.series[0]
                axes = getattr(series, "axes", "")
                memmap_exc: Exception | None = None
                try:
                    arr = np.asarray(tiff_memmap(path, series=0, mode="r"))
                except Exception as exc:
                    memmap_exc = exc
                    try:
                        arr = np.asarray(series.asarray())
                    except Exception as exc2:
                        raise RuntimeError(
                            f"Failed to read TIFF series 0 from {path}. "
                            f"memmap error: {memmap_exc!r}; "
                            f"series.asarray error: {exc2!r}"
                        ) from exc2
    except Exception as exc:
        raise RuntimeError(f"Failed to open TIFF tensor from {path}") from exc

    if output_size is not None and not use_local_pages:
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
        axes = "TZYX"

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


def build_tiff_cache_path(path: str | Path) -> Path:
    source_path = Path(path).resolve()
    relative_parts = source_path.parts[1:] if source_path.is_absolute() else source_path.parts
    return TIFF_CACHE_DIR.joinpath(*relative_parts)


def ensure_cached_tiff(path: str | Path) -> Path:
    source_path = Path(path).resolve()
    cache_path = build_tiff_cache_path(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        source_stat = source_path.stat()
        cache_stat = cache_path.stat()
        if (
            cache_stat.st_size == source_stat.st_size
            and cache_stat.st_mtime_ns >= source_stat.st_mtime_ns
        ):
            return cache_path

    shutil.copy2(source_path, cache_path)
    return cache_path


def is_tiff_cached(path: str | Path) -> bool:
    source_path = Path(path).resolve()
    cache_path = build_tiff_cache_path(source_path)
    if not cache_path.exists():
        return False

    source_stat = source_path.stat()
    cache_stat = cache_path.stat()
    return (
        cache_stat.st_size == source_stat.st_size
        and cache_stat.st_mtime_ns >= source_stat.st_mtime_ns
    )


def ensure_cached_tiffs(paths: list[Path]) -> list[Path]:
    return [ensure_cached_tiff(path) for path in paths]


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


def has_cached_tensor(cache_key: str) -> bool:
    cache_path = TENSOR_CACHE_DIR / f"{cache_key}.pt"
    return cache_path.exists()


def save_cached_tensor(cache_key: str, tensor: torch.Tensor) -> None:
    TENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TENSOR_CACHE_DIR / f"{cache_key}.pt"
    torch.save(tensor.cpu(), cache_path)


def save_labeled_tensor_dataset(
    dataset: dict[str, object],
    path: str | Path,
) -> Path:
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = DATASET_CACHE_DIR / dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = dataset["metadata"]
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("dataset['metadata'] must be a pandas DataFrame")

    payload: dict[str, object] = {
        "tensors": dataset["tensors"].detach().cpu(),
        "labels": dataset["labels"].detach().cpu(),
        "metadata_records": metadata.to_dict(orient="records"),
        "label_map": {int(key): str(value) for key, value in dict(dataset["label_map"]).items()},
    }
    for tensor_key in ("compound_labels", "concentration_labels", "is_control"):
        if tensor_key in dataset:
            value = dataset[tensor_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"dataset[{tensor_key!r}] must be a torch.Tensor")
            payload[tensor_key] = value.detach().cpu()
    for map_key in ("compound_label_map", "concentration_label_map"):
        if map_key in dataset:
            payload[map_key] = {int(key): str(value) for key, value in dict(dataset[map_key]).items()}

    torch.save(payload, dataset_path)
    return dataset_path


def load_labeled_tensor_dataset(path: str | Path) -> dict[str, object]:
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = DATASET_CACHE_DIR / dataset_path
    payload = torch.load(dataset_path, map_location="cpu")
    dataset = {
        "tensors": payload["tensors"],
        "labels": payload["labels"],
        "metadata": pd.DataFrame(payload["metadata_records"]),
        "label_map": {int(key): value for key, value in payload["label_map"].items()},
    }
    for tensor_key in ("compound_labels", "concentration_labels", "is_control"):
        if tensor_key in payload:
            dataset[tensor_key] = payload[tensor_key]
    for map_key in ("compound_label_map", "concentration_label_map"):
        if map_key in payload:
            dataset[map_key] = {int(key): value for key, value in payload[map_key].items()}
    return dataset


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
    use_tiff_cache: bool = True,
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

    load_paths = ensure_cached_tiffs(timepoint_files) if use_tiff_cache else timepoint_files

    tensors = []
    for path in load_paths:
        try:
            tensors.append(load_tiff_as_tzyx(path, output_size=output_size))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TIFF timepoint {path} for condition directory {condition_dir}"
            ) from exc
    reference_shape = tensors[0].shape[1:]
    mismatched = [str(path) for path, tensor in zip(load_paths, tensors) if tensor.shape[1:] != reference_shape]
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


def describe_condition_tensor_source(
    condition_dir: str | Path,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
    *,
    normalize_global_drift: bool = True,
    loess_frac: float = 0.25,
    use_cache: bool = True,
    use_tiff_cache: bool = True,
) -> str:
    condition_dir = Path(condition_dir)
    timepoint_files = list_timepoint_files(condition_dir)
    if not timepoint_files:
        return "missing"

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
    if use_cache and has_cached_tensor(cache_key):
        return "tensor_cache"
    if use_tiff_cache and all(is_tiff_cached(path) for path in timepoint_files):
        return "tiff_cache"
    return "source_fs"


def rotate_tensor_xy(tensor: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor with shape T x Z x Y x X, got shape {tuple(tensor.shape)}")

    work_tensor = tensor.to(torch.float32)
    t, z, y, x = work_tensor.shape
    batched = work_tensor.reshape(t * z, 1, y, x)

    theta = math.radians(float(angle_degrees))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    affine = torch.tensor(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0]],
        dtype=batched.dtype,
        device=batched.device,
    ).unsqueeze(0).repeat(t * z, 1, 1)

    grid = F.affine_grid(affine, batched.size(), align_corners=False)
    rotated = F.grid_sample(
        batched,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    rotated = rotated.reshape(t, z, y, x)
    return rotated.to(tensor.dtype) if torch.is_floating_point(tensor) else rotated


def build_moa_labeled_tensor_dataset(
    condition_df: pd.DataFrame,
    selected_mechanisms: list[str],
    selected_concentrations: list[str],
    max_compounds_per_action: int,
    max_tensors_per_compound: int,
    output_size: tuple[int | None, int | None, int | None, int | None],
    *,
    only_active: bool = True,
    normalize_global_drift: bool = True,
    loess_frac: float = 0.25,
    use_cache: bool = True,
    use_tiff_cache: bool = True,
    skip_failed_conditions: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    if not selected_mechanisms:
        raise ValueError("selected_mechanisms must not be empty")
    if not selected_concentrations:
        raise ValueError("selected_concentrations must not be empty")
    if max_compounds_per_action <= 0:
        raise ValueError("max_compounds_per_action must be positive")
    if max_tensors_per_compound <= 0:
        raise ValueError("max_tensors_per_compound must be positive")
    working_df = condition_df.copy()
    if only_active:
        working_df = working_df[working_df["condition_folder_status"] == "active"].copy()

    label_map = {0: "Water"}
    for index, mechanism in enumerate(selected_mechanisms, start=1):
        label_map[index] = mechanism
    compound_label_map = {0: "Control"}
    concentration_label_map = {0: "control"}
    compound_name_to_label: dict[str, int] = {}
    concentration_value_to_label: dict[str, int] = {}

    tensors: list[torch.Tensor] = []
    labels: list[int] = []
    compound_labels: list[int] = []
    concentration_labels: list[int] = []
    is_control_values: list[int] = []
    rows: list[dict[str, object]] = []
    total_conditions = 0

    for mechanism in selected_mechanisms:
        mechanism_df = working_df[
            (working_df["mechanism_of_action"] == mechanism)
            & (working_df["condition_kind"] == "treatment")
            & (working_df["concentration_band"].isin(selected_concentrations))
        ].copy()
        compounds = mechanism_df["compound"].drop_duplicates().tolist()[:max_compounds_per_action]
        for compound in compounds:
            treatment_rows = (
                mechanism_df[mechanism_df["compound"] == compound]
                .drop_duplicates(subset=["image_condition_dir"])
                .sort_values(["concentration_band", "image_condition_dir"])
                .head(max_tensors_per_compound)
            )
            control_rows = (
                working_df[
                    (working_df["compound"] == compound)
                    & (working_df["condition_kind"] == "control")
                ]
                .drop_duplicates(subset=["image_condition_dir"])
                .sort_values("image_condition_dir")
                .head(max_tensors_per_compound)
            )
            total_conditions += int(len(treatment_rows) + len(control_rows))

    attempted_conditions = 0
    build_start = time.perf_counter()

    def log_progress(*, row: pd.Series, mechanism: str, kind: str, source: str) -> None:
        if not verbose or total_conditions <= 0:
            return
        current_index = attempted_conditions + 1
        elapsed = time.perf_counter() - build_start
        avg_seconds = elapsed / max(attempted_conditions, 1)
        eta = _format_eta(avg_seconds * (total_conditions - attempted_conditions))
        elapsed_text = _format_eta(elapsed)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compound = str(row["compound"])
        concentration = str(row["concentration_band"])
        print(
            f"[{timestamp}] "
            f"[{current_index:03d}/{total_conditions:03d}] "
            f"kind={kind:<9} "
            f"source={source:<12} "
            f"conc={concentration:<8} "
            f"elapsed={elapsed_text} "
            f"eta={eta} "
            f"mechanism={mechanism} "
            f"compound={compound}"
        )

    def add_example(
        *,
        base_tensor: torch.Tensor,
        label: int,
        label_name: str,
        row: pd.Series,
        is_control: bool,
        original_instance_id: int,
    ) -> None:
        compound_name = str(row["compound"])
        concentration_value = str(row["concentration_band"])
        if is_control:
            compound_label = 0
            concentration_label = 0
        else:
            if compound_name not in compound_name_to_label:
                compound_label = len(compound_label_map)
                compound_name_to_label[compound_name] = compound_label
                compound_label_map[compound_label] = compound_name
            else:
                compound_label = compound_name_to_label[compound_name]

            if concentration_value not in concentration_value_to_label:
                concentration_label = len(concentration_label_map)
                concentration_value_to_label[concentration_value] = concentration_label
                concentration_label_map[concentration_label] = concentration_value
            else:
                concentration_label = concentration_value_to_label[concentration_value]

        tensors.append(base_tensor)
        labels.append(label)
        compound_labels.append(int(compound_label))
        concentration_labels.append(int(concentration_label))
        is_control_values.append(int(is_control))
        rows.append(
            {
                "original_instance_id": original_instance_id,
                "label": label,
                "compound_label": int(compound_label),
                "compound_label_name": compound_label_map[int(compound_label)],
                "concentration_label_id": int(concentration_label),
                "concentration_label_name": concentration_label_map[int(concentration_label)],
                "label_name": label_name,
                "mechanism_of_action": row["mechanism_of_action"],
                "compound": row["compound"],
                "concentration_band": row["concentration_band"],
                "concentration_label": row["concentration_label"],
                "image_condition_dir": row["image_condition_dir"],
                "is_control": is_control,
            }
        )

    original_instance_id = 0
    for mechanism_index, mechanism in enumerate(selected_mechanisms, start=1):
        mechanism_df = working_df[
            (working_df["mechanism_of_action"] == mechanism)
            & (working_df["condition_kind"] == "treatment")
            & (working_df["concentration_band"].isin(selected_concentrations))
        ].copy()

        compounds = mechanism_df["compound"].drop_duplicates().tolist()[:max_compounds_per_action]
        for compound in compounds:
            treatment_rows = (
                mechanism_df[mechanism_df["compound"] == compound]
                .drop_duplicates(subset=["image_condition_dir"])
                .sort_values(["concentration_band", "image_condition_dir"])
                .head(max_tensors_per_compound)
            )
            control_rows = (
                working_df[
                    (working_df["compound"] == compound)
                    & (working_df["condition_kind"] == "control")
                ]
                .drop_duplicates(subset=["image_condition_dir"])
                .sort_values("image_condition_dir")
                .head(max_tensors_per_compound)
            )

            for _, row in treatment_rows.iterrows():
                source = describe_condition_tensor_source(
                    condition_dir=row["image_condition_dir"],
                    output_size=output_size,
                    normalize_global_drift=normalize_global_drift,
                    loess_frac=loess_frac,
                    use_cache=use_cache,
                    use_tiff_cache=use_tiff_cache,
                )
                log_progress(row=row, mechanism=mechanism, kind="treatment", source=source)
                attempted_conditions += 1
                try:
                    tensor = load_image_condition_tensor(
                        condition_dir=row["image_condition_dir"],
                        output_size=output_size,
                        normalize_global_drift=normalize_global_drift,
                        loess_frac=loess_frac,
                        use_cache=use_cache,
                        use_tiff_cache=use_tiff_cache,
                    )
                except Exception as exc:
                    message = (
                        "Failed to build treatment tensor dataset example "
                        f"for mechanism={mechanism!r}, compound={row['compound']!r}, "
                        f"concentration_band={row['concentration_band']!r}, "
                        f"image_condition_dir={row['image_condition_dir']!r}"
                    )
                    if skip_failed_conditions:
                        warnings.warn(f"{message}. Skipping this example. Root cause: {exc!r}")
                        continue
                    raise RuntimeError(message) from exc
                add_example(
                    base_tensor=tensor,
                    label=mechanism_index,
                    label_name=mechanism,
                    row=row,
                    is_control=False,
                    original_instance_id=original_instance_id,
                )
                original_instance_id += 1

            for _, row in control_rows.iterrows():
                source = describe_condition_tensor_source(
                    condition_dir=row["image_condition_dir"],
                    output_size=output_size,
                    normalize_global_drift=normalize_global_drift,
                    loess_frac=loess_frac,
                    use_cache=use_cache,
                    use_tiff_cache=use_tiff_cache,
                )
                log_progress(row=row, mechanism=mechanism, kind="control", source=source)
                attempted_conditions += 1
                try:
                    tensor = load_image_condition_tensor(
                        condition_dir=row["image_condition_dir"],
                        output_size=output_size,
                        normalize_global_drift=normalize_global_drift,
                        loess_frac=loess_frac,
                        use_cache=use_cache,
                        use_tiff_cache=use_tiff_cache,
                    )
                except Exception as exc:
                    message = (
                        "Failed to build control tensor dataset example "
                        f"for compound={row['compound']!r}, "
                        f"image_condition_dir={row['image_condition_dir']!r}"
                    )
                    if skip_failed_conditions:
                        warnings.warn(f"{message}. Skipping this example. Root cause: {exc!r}")
                        continue
                    raise RuntimeError(message) from exc
                add_example(
                    base_tensor=tensor,
                    label=0,
                    label_name="Water",
                    row=row,
                    is_control=True,
                    original_instance_id=original_instance_id,
                )
                original_instance_id += 1

    if not tensors:
        raise ValueError("No dataset examples were created with the provided filters")

    return {
        "tensors": torch.stack(tensors, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "compound_labels": torch.tensor(compound_labels, dtype=torch.long),
        "concentration_labels": torch.tensor(concentration_labels, dtype=torch.long),
        "is_control": torch.tensor(is_control_values, dtype=torch.bool),
        "metadata": pd.DataFrame(rows),
        "label_map": label_map,
        "compound_label_map": compound_label_map,
        "concentration_label_map": concentration_label_map,
    }


def build_tensor_embedding_2d(
    tensors: torch.Tensor,
    labels: torch.Tensor | np.ndarray | list[int],
    *,
    label_map: dict[int, str] | None = None,
    metadata: pd.DataFrame | None = None,
    method: str = "pca",
    random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> pd.DataFrame:
    if tensors.ndim < 2:
        raise ValueError(f"Expected tensors with leading sample dimension, got shape {tuple(tensors.shape)}")

    features = tensors.detach().cpu().reshape(tensors.shape[0], -1).numpy()
    labels_np = np.asarray(labels, dtype=int)
    if labels_np.shape[0] != features.shape[0]:
        raise ValueError("labels length must match number of tensors")

    method_lower = method.lower()
    if method_lower == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features)
    elif method_lower == "umap":
        try:
            import umap
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "UMAP requires the 'umap-learn' package in the active environment"
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "n_jobs value 1 overridden to 1 by setting random_state. "
                    "Use no seed for parallelism."
                ),
                category=UserWarning,
            )
            embedding = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unsupported embedding method {method!r}; use 'pca' or 'umap'")

    embedding_df = pd.DataFrame(
        {
            "embed_x": embedding[:, 0],
            "embed_y": embedding[:, 1],
            "label": labels_np,
            "label_name": [label_map.get(int(label), str(int(label))) for label in labels_np]
            if label_map is not None
            else labels_np.astype(str),
        }
    )
    embedding_df["method"] = method_lower

    if metadata is not None:
        metadata_reset = metadata.reset_index(drop=True).copy()
        if len(metadata_reset) != len(embedding_df):
            raise ValueError("metadata length must match number of tensors")
        duplicate_columns = [column for column in metadata_reset.columns if column in embedding_df.columns]
        if duplicate_columns:
            metadata_reset = metadata_reset.drop(columns=duplicate_columns)
        embedding_df = pd.concat([embedding_df, metadata_reset], axis=1)

    return embedding_df


def build_dataset_tensor_embedding_2d(
    dataset: dict[str, object],
    *,
    target: str = "mechanism",
    method: str = "pca",
    random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> pd.DataFrame:
    target_lower = target.lower()
    target_config = {
        "mechanism": ("labels", "label_map"),
        "compound": ("compound_labels", "compound_label_map"),
        "concentration": ("concentration_labels", "concentration_label_map"),
        "control": ("is_control", {0: "Treatment", 1: "Control"}),
    }
    if target_lower not in target_config:
        raise ValueError(
            f"Unsupported target {target!r}; use one of {sorted(target_config)}"
        )

    tensor_key, label_map = target_config[target_lower]
    if tensor_key not in dataset:
        raise KeyError(f"dataset does not contain {tensor_key!r}")
    labels = dataset[tensor_key]
    if isinstance(label_map, str):
        if label_map not in dataset:
            raise KeyError(f"dataset does not contain {label_map!r}")
        label_map = dataset[label_map]

    if isinstance(labels, torch.Tensor):
        labels_array = labels.detach().cpu().numpy().astype(int)
    else:
        labels_array = np.asarray(labels, dtype=int)

    embedding_df = build_tensor_embedding_2d(
        dataset["tensors"],
        labels_array,
        label_map=label_map,
        metadata=dataset.get("metadata"),
        method=method,
        random_state=random_state,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )
    embedding_df["target"] = target_lower
    return embedding_df


def plot_tensor_embedding_2d(
    embedding_df: pd.DataFrame,
    *,
    title: str | None = None,
    ax=None,
    marker_column: str | None = "compound",
    show_svm_background: bool = False,
    svm_background_alpha: float = 0.14,
    svm_background_resolution: int = 300,
    svm_c: float = 1.0,
    svm_gamma: str | float = "scale",
):
    class_palette = [
        "#A0C2E7",
        "#F58518",
        "#E45756",
        "#36852D",
        "#66EE8F",
        "#B279A2",
        "#E1B0FE",
    ]
    class_color_overrides = {
        "Water": "#A0C2E7",
        "GABAAR_Antagonist": "#F58518",
        "GABAAR_NegativeAllostericModulator": "#E45756",
        "NMDAR_Activation": "#36852D",
        "NMDAR_Antagonist": "#66EE8F",
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(15.5, 10.5))
    else:
        fig = ax.figure

    unique_labels = embedding_df[["label", "label_name"]].drop_duplicates().sort_values("label")
    color_map = {
        int(row.label): class_color_overrides.get(
            str(row.label_name), class_palette[color_index % len(class_palette)]
        )
        for color_index, row in enumerate(unique_labels.itertuples(index=False))
    }
    ordered_labels = [int(row.label) for row in unique_labels.itertuples(index=False)]

    if show_svm_background and len(ordered_labels) >= 2:
        X = embedding_df[["embed_x", "embed_y"]].to_numpy(dtype=float)
        y = embedding_df["label"].to_numpy(dtype=int)
        label_to_index = {label: index for index, label in enumerate(ordered_labels)}
        y_index = np.array([label_to_index[int(label)] for label in y], dtype=int)

        classifier = SVC(kernel="rbf", C=svm_c, gamma=svm_gamma)
        classifier.fit(X, y_index)

        x_margin = 0.05 * max(float(np.ptp(X[:, 0])), 1e-6)
        y_margin = 0.05 * max(float(np.ptp(X[:, 1])), 1e-6)
        x_min, x_max = float(X[:, 0].min() - x_margin), float(X[:, 0].max() + x_margin)
        y_min, y_max = float(X[:, 1].min() - y_margin), float(X[:, 1].max() + y_margin)

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, int(svm_background_resolution)),
            np.linspace(y_min, y_max, int(svm_background_resolution)),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        zz = classifier.predict(grid).reshape(xx.shape)

        background_colors = [
            to_rgba(color_map[label], alpha=svm_background_alpha) for label in ordered_labels
        ]
        ax.contourf(
            xx,
            yy,
            zz,
            levels=np.arange(len(ordered_labels) + 1) - 0.5,
            cmap=ListedColormap(background_colors),
            antialiased=True,
            zorder=0,
        )

    marker_map = None
    if marker_column is not None and marker_column in embedding_df.columns:
        unique_markers = (
            embedding_df[[marker_column]]
            .drop_duplicates()
            .sort_values(marker_column)
            .reset_index(drop=True)[marker_column]
            .tolist()
        )
        marker_map = {
            marker_value: marker_cycle[marker_index % len(marker_cycle)]
            for marker_index, marker_value in enumerate(unique_markers)
        }

    if marker_map is None:
        for row in unique_labels.itertuples(index=False):
            class_df = embedding_df[embedding_df["label"] == row.label]
            ax.scatter(
                class_df["embed_x"],
                class_df["embed_y"],
                s=64,
                alpha=0.9,
                color=color_map[int(row.label)],
                marker="o",
                edgecolors="white",
                linewidths=0.7,
                zorder=2,
            )
    else:
        for row in embedding_df.itertuples(index=False):
            ax.scatter(
                row.embed_x,
                row.embed_y,
                s=96,
                alpha=0.92,
                color=color_map[int(row.label)],
                marker=marker_map[getattr(row, marker_column)],
                edgecolors="white",
                linewidths=0.7,
                zorder=2,
            )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"{embedding_df['method'].iloc[0].upper()} tensor embedding")
    ax.tick_params(labelsize=13)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(21)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.set_box_aspect(1)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_map[int(row.label)],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=11,
            label=f"{row.label}: {row.label_name}",
        )
        for row in unique_labels.itertuples(index=False)
    ]
    legend_x = 1.01
    legend_width = 0.33

    class_legend = ax.legend(
        handles=class_handles,
        title="Class",
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.0, legend_width, 0.0),
        borderaxespad=0.0,
        frameon=True,
        mode="expand",
    )
    ax.add_artist(class_legend)

    if marker_map is not None:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_shape,
                color="#4A4A4A",
                markerfacecolor="#4A4A4A",
                linestyle="None",
                markersize=10,
                label=str(marker_value),
            )
            for marker_value, marker_shape in marker_map.items()
        ]
        ax.legend(
            handles=marker_handles,
            title=marker_column.replace("_", " ").title(),
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.46, legend_width, 0.0),
            borderaxespad=0.0,
            frameon=True,
            ncol=1,
            mode="expand",
        )

    fig.subplots_adjust(right=0.68)
    fig.tight_layout(rect=(0, 0, 0.68, 1))
    return fig, ax
