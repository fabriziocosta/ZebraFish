from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.models.common import _ensure_labels_1d, _ensure_tensor_5d


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
            augmented_tensors.append(_rotate_tensor_xy(base_tensor, angle))
            augmented_labels.append(int(label_values[index]))
            if metadata_df is not None and augmented_rows is not None:
                row = metadata_df.iloc[index].to_dict()
                row["augmentation_index"] = augmentation_index
                row["rotation_degrees"] = angle
                augmented_rows.append(row)

    augmented_metadata = pd.DataFrame(augmented_rows) if augmented_rows is not None else None
    return torch.stack(augmented_tensors, dim=0), np.asarray(augmented_labels, dtype=int), augmented_metadata


def _rotate_tensor_xy(tensor: torch.Tensor, angle_degrees: float) -> torch.Tensor:
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
