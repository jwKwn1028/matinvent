"""Small, inspectable linear surrogate for calibrated conductivity predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from rewards.calculators.ionic.descriptors import SURROGATE_FEATURE_NAMES


MODEL_SCHEMA_VERSION = 1
DEFAULT_TARGET = "log10_ionic_conductivity_s_cm"


def _feature_matrix(
    records: Sequence[Mapping[str, float]], feature_names: Sequence[str]
) -> np.ndarray:
    try:
        matrix = np.asarray(
            [[float(record[name]) for name in feature_names] for record in records],
            dtype=float,
        )
    except KeyError as exc:
        raise ValueError(f"Missing surrogate feature: {exc.args[0]}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("Invalid surrogate feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Surrogate features must all be finite")
    return matrix


@dataclass(frozen=True)
class LinearSurrogate:
    """Standardized ridge-regression model with applicability metadata."""

    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray
    target: str = DEFAULT_TARGET
    validation_rmse: float | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        arrays = (
            self.coefficients,
            self.feature_mean,
            self.feature_scale,
            self.feature_min,
            self.feature_max,
        )
        if not self.feature_names or any(
            np.asarray(value).shape != (size,) for value in arrays
        ):
            raise ValueError("Every surrogate parameter must match feature_names")
        if len(set(self.feature_names)) != size:
            raise ValueError("Surrogate feature_names must be unique")
        if not self.target.strip():
            raise ValueError("Surrogate target must not be empty")
        if not all(
            np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays
        ):
            raise ValueError("Surrogate parameters must all be finite")
        if not np.isfinite(self.intercept):
            raise ValueError("Surrogate intercept must be finite")
        if np.any(np.asarray(self.feature_scale) <= 0):
            raise ValueError("feature_scale values must be positive")
        if np.any(np.asarray(self.feature_min) > np.asarray(self.feature_max)):
            raise ValueError("feature_min must not exceed feature_max")
        if self.validation_rmse is not None and (
            not np.isfinite(self.validation_rmse) or self.validation_rmse < 0
        ):
            raise ValueError("validation_rmse must be finite and non-negative")

    def predict(self, records: Sequence[Mapping[str, float]]) -> np.ndarray:
        matrix = _feature_matrix(records, self.feature_names)
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    def applicability(self, records: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Return a [0, 1] score based on distance beyond the training range."""

        matrix = _feature_matrix(records, self.feature_names)
        below = np.maximum(self.feature_min - matrix, 0.0) / self.feature_scale
        above = np.maximum(matrix - self.feature_max, 0.0) / self.feature_scale
        extrapolation = np.maximum(below, above).max(axis=1)
        return np.exp(-extrapolation)

    def uncertainty(self, records: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Inflate cross-validation RMSE when a sample is out of domain."""

        base = (
            float(self.validation_rmse) if self.validation_rmse is not None else np.nan
        )
        applicability = self.applicability(records)
        return base * (1.0 + 2.0 * (1.0 - applicability))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_type": "standardized_ridge_regression",
            "target": self.target,
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "validation_rmse": self.validation_rmse,
            "metadata": self.metadata or {},
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LinearSurrogate":
        version = payload.get("schema_version")
        if version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported surrogate schema {version!r}; expected {MODEL_SCHEMA_VERSION}"
            )
        if payload.get("model_type") != "standardized_ridge_regression":
            raise ValueError("Only standardized_ridge_regression models are supported")

        feature_names = tuple(str(name) for name in payload["feature_names"])
        return cls(
            feature_names=feature_names,
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=float(payload["intercept"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            feature_min=np.asarray(payload["feature_min"], dtype=float),
            feature_max=np.asarray(payload["feature_max"], dtype=float),
            target=str(payload.get("target", DEFAULT_TARGET)),
            validation_rmse=(
                None
                if payload.get("validation_rmse") is None
                else float(payload["validation_rmse"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearSurrogate":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def fit_linear_surrogate(
    records: Sequence[Mapping[str, float]],
    targets: Iterable[float],
    feature_names: Sequence[str] = SURROGATE_FEATURE_NAMES,
    ridge_alpha: float = 1.0,
    target_name: str = DEFAULT_TARGET,
    validation_rmse: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> LinearSurrogate:
    """Fit a deterministic standardized ridge-regression model."""

    matrix = _feature_matrix(records, feature_names)
    target = np.asarray(list(targets), dtype=float)
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("records and targets must have the same length")
    if matrix.shape[0] < 2:
        raise ValueError("At least two training samples are required")
    if not np.isfinite(target).all():
        raise ValueError("Training targets must all be finite")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    intercept = float(target.mean())
    centered_target = target - intercept
    regularizer = ridge_alpha * np.eye(standardized.shape[1])
    coefficients = np.linalg.lstsq(
        standardized.T @ standardized + regularizer,
        standardized.T @ centered_target,
        rcond=None,
    )[0]

    return LinearSurrogate(
        feature_names=tuple(feature_names),
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=mean,
        feature_scale=scale,
        feature_min=matrix.min(axis=0),
        feature_max=matrix.max(axis=0),
        target=target_name,
        validation_rmse=validation_rmse,
        metadata=dict(metadata or {}),
    )


def cross_validated_rmse(
    records: Sequence[Mapping[str, float]],
    targets: Iterable[float],
    feature_names: Sequence[str] = SURROGATE_FEATURE_NAMES,
    ridge_alpha: float = 1.0,
    folds: int = 5,
    seed: int = 7,
) -> float:
    """Estimate generalization error with deterministic shuffled K-fold CV."""

    target = np.asarray(list(targets), dtype=float)
    if len(records) != target.size:
        raise ValueError("records and targets must have the same length")
    if target.size < 3:
        raise ValueError("At least three samples are required for cross-validation")

    fold_count = min(max(2, folds), target.size)
    indices = np.arange(target.size)
    np.random.default_rng(seed).shuffle(indices)
    predictions = np.empty_like(target)

    for validation_indices in np.array_split(indices, fold_count):
        training_mask = np.ones(target.size, dtype=bool)
        training_mask[validation_indices] = False
        training_records = [records[index] for index in np.flatnonzero(training_mask)]
        validation_records = [records[index] for index in validation_indices]
        model = fit_linear_surrogate(
            training_records,
            target[training_mask],
            feature_names=feature_names,
            ridge_alpha=ridge_alpha,
        )
        predictions[validation_indices] = model.predict(validation_records)

    return float(np.sqrt(np.mean((predictions - target) ** 2)))
