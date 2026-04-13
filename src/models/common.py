from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split


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


def _prepare_multitask_training_data(
    estimator,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray | Sequence[int],
    validation_data: tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | Sequence[int]] | None,
    *,
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

    estimator.classes_ = np.array(sorted(np.unique(y_values)))
    estimator.class_to_index_ = {label: index for index, label in enumerate(estimator.classes_)}
    estimator.compound_classes_, estimator.compound_class_to_index_ = _fit_target_encoder(compound_values)
    estimator.concentration_classes_, estimator.concentration_class_to_index_ = _fit_target_encoder(concentration_values)

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
    elif estimator.validation_split and 0 < estimator.validation_split < 1:
        indices = np.arange(len(y_values))
        stratify = y_values if len(np.unique(y_values)) > 1 else None
        try:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=estimator.validation_split,
                random_state=estimator.random_state,
                stratify=stratify,
            )
        except ValueError:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=estimator.validation_split,
                random_state=estimator.random_state,
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

    estimator._standardize_fit(X_train)
    X_train = estimator._standardize_apply(X_train)
    y_train = estimator._encode_labels(y_train_values)
    compound_train = estimator._encode_compound_labels(compound_train_values)
    concentration_train = estimator._encode_concentration_labels(concentration_train_values)

    if X_val is not None and y_val_values is not None:
        X_val = estimator._standardize_apply(X_val)
        y_val = estimator._encode_labels(y_val_values)
        compound_val = estimator._encode_compound_labels(compound_val_values)
        concentration_val = estimator._encode_concentration_labels(concentration_val_values)
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


class _SharedMultitaskEstimatorMixin:
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
        return _prepare_multitask_training_data(
            self,
            X,
            y,
            validation_data,
            compound_y=compound_y,
            concentration_y=concentration_y,
            validation_compound_y=validation_compound_y,
            validation_concentration_y=validation_concentration_y,
        )

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
    ):
        from src.training.loop import _fit_multitask_estimator

        prepared = self._prepare_training_data(
            X,
            y,
            validation_data,
            compound_y=compound_y,
            concentration_y=concentration_y,
            validation_compound_y=validation_compound_y,
            validation_concentration_y=validation_concentration_y,
        )
        return _fit_multitask_estimator(self, prepared)

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        from src.training.loop import _predict_proba_from_estimator

        return _predict_proba_from_estimator(self, X)

    def predict(self, X: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        from src.training.loop import _predict_from_estimator

        return _predict_from_estimator(self, X)

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        from src.training.loop import _transform_from_estimator

        return _transform_from_estimator(self, X)

    def evaluate_loss_components(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray | Sequence[int],
    ) -> dict[str, float]:
        from src.training.loop import _evaluate_loss_components_from_estimator

        return _evaluate_loss_components_from_estimator(self, X, y)

    def score(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | Sequence[int]) -> float:
        from src.training.loop import _score_from_estimator

        return _score_from_estimator(self, X, y)
