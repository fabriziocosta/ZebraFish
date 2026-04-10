from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from zebrafish_tensor_utils import rotate_tensor_xy


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


@dataclass
class _PreparedData:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor | None
    y_val: torch.Tensor | None


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

    def forward_features(self, X: torch.Tensor) -> torch.Tensor:
        X = self.backbone(X)
        X = self.global_pool(X)
        return self.embedding(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(X))


class Zebrafish3DCNNClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
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

    def _prepare_training_data(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None,
    ) -> _PreparedData:
        X_tensor = _ensure_tensor_5d(X)
        y_values = _ensure_labels_1d(y)
        if len(X_tensor) != len(y_values):
            raise ValueError("X and y must have the same number of samples")

        self.classes_ = np.array(sorted(np.unique(y_values)))
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}

        if validation_data is not None:
            X_val = _ensure_tensor_5d(validation_data[0])
            y_val_values = _ensure_labels_1d(validation_data[1])
            if len(X_val) != len(y_val_values):
                raise ValueError("validation_data tensors and labels must have the same number of samples")
            X_train = X_tensor
            y_train_values = y_values
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
        else:
            X_train = X_tensor
            X_val = None
            y_train_values = y_values
            y_val_values = None

        self._standardize_fit(X_train)
        X_train = self._standardize_apply(X_train)
        y_train = self._encode_labels(y_train_values)

        if X_val is not None and y_val_values is not None:
            X_val = self._standardize_apply(X_val)
            y_val = self._encode_labels(y_val_values)
        else:
            X_val = None
            y_val = None

        return _PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
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
        )

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
        *,
        validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None = None,
    ) -> "Zebrafish3DCNNClassifier":
        prepared = self._prepare_training_data(X, y, validation_data)
        self.model_ = self._build_model(
            in_channels=int(prepared.X_train.shape[1]),
            num_classes=len(self.classes_),
        )
        self.device_ = self._device()
        self.model_.to(self.device_)
        self.input_shape_ = tuple(int(size) for size in prepared.X_train.shape[1:])

        train_loader = DataLoader(
            TensorDataset(prepared.X_train, prepared.y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = (
            DataLoader(
                TensorDataset(prepared.X_val, prepared.y_val),
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

        history_rows: list[dict[str, float | int]] = []
        best_state = deepcopy(self.model_.state_dict())
        best_metric = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss_sum = 0.0
            train_count = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                y_batch = y_batch.to(self.device_, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = int(X_batch.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
            }

            if val_loader is not None:
                self.model_.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device_, non_blocking=True)
                        y_batch = y_batch.to(self.device_, non_blocking=True)
                        logits = self.model_(X_batch)
                        loss = criterion(logits, y_batch)
                        batch_size = int(X_batch.shape[0])
                        val_loss_sum += float(loss.item()) * batch_size
                        val_count += batch_size
                row["val_loss"] = val_loss_sum / max(val_count, 1)
                metric = float(row["val_loss"])
            else:
                metric = float(row["train_loss"])

            if metric <= best_metric:
                best_metric = metric
                best_state = deepcopy(self.model_.state_dict())

            history_rows.append(row)
            if self.verbose:
                if "val_loss" in row:
                    print(
                        f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
                        f"val_loss={row['val_loss']:.4f}"
                    )
                else:
                    print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f}")

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = pd.DataFrame(history_rows)
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
                    else self.model_(X_batch)
                )
                outputs.append(batch_out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        logits = self._forward_batches(X, features=False)
        logits_tensor = torch.from_numpy(logits)
        return torch.softmax(logits_tensor, dim=1).numpy()

    def predict(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices = probabilities.argmax(axis=1)
        return self.classes_[indices]

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return self._forward_batches(X, features=True)

    def score(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | Sequence[int]) -> float:
        y_true = _ensure_labels_1d(y)
        y_pred = self.predict(X)
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


def plot_training_history(
    history: pd.DataFrame | Zebrafish3DCNNClassifier,
    *,
    ax=None,
    title: str = "Training history",
):
    if isinstance(history, Zebrafish3DCNNClassifier):
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
