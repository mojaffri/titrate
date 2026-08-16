"""Bounded, thread-safe inference diagnostics for lightweight model monitoring."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Serializable view of the current rolling inference window."""

    total_predictions: int
    window_size: int
    mean_prediction: float | None
    mean_predictive_uncertainty: float | None
    feature_mean_shift: list[float]
    max_feature_mean_shift: float
    drift_detected: bool
    drift_threshold: float
    minimum_drift_samples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class InferenceMonitor:
    """Track recent inputs and predictions without retaining an unbounded log.

    Drift is measured as the largest absolute shift in a scaled feature mean,
    divided by that feature's training standard deviation. A minimum sample
    count prevents a few interactive requests from triggering an alert.
    """

    def __init__(
        self,
        *,
        max_window: int = 2048,
        minimum_drift_samples: int = 25,
        drift_threshold: float = 0.75,
    ) -> None:
        if max_window < 1 or minimum_drift_samples < 1:
            raise ValueError("max_window and minimum_drift_samples must be positive.")
        if drift_threshold <= 0:
            raise ValueError("drift_threshold must be positive.")
        self.max_window = int(max_window)
        self.minimum_drift_samples = int(minimum_drift_samples)
        self.drift_threshold = float(drift_threshold)
        self._inputs: deque[np.ndarray] = deque(maxlen=self.max_window)
        self._predictions: deque[float] = deque(maxlen=self.max_window)
        self._uncertainties: deque[float] = deque(maxlen=self.max_window)
        self._total_predictions = 0
        self._lock = Lock()

    def record(
        self,
        scaled_inputs: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> None:
        scaled_inputs = np.atleast_2d(np.asarray(scaled_inputs, dtype=float))
        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        uncertainties = np.asarray(uncertainties, dtype=float).reshape(-1)
        if len(scaled_inputs) != len(predictions) or len(predictions) != len(uncertainties):
            raise ValueError("Inputs, predictions, and uncertainties must have matching rows.")
        if not (
            np.isfinite(scaled_inputs).all()
            and np.isfinite(predictions).all()
            and np.isfinite(uncertainties).all()
        ):
            raise ValueError("Monitoring values must be finite.")

        with self._lock:
            for point, prediction, uncertainty in zip(
                scaled_inputs,
                predictions,
                uncertainties,
            ):
                self._inputs.append(point.copy())
                self._predictions.append(float(prediction))
                self._uncertainties.append(float(uncertainty))
            self._total_predictions += len(predictions)

    def snapshot(
        self,
        reference_mean: np.ndarray,
        reference_std: np.ndarray,
    ) -> MonitoringSnapshot:
        reference_mean = np.asarray(reference_mean, dtype=float).reshape(-1)
        reference_std = np.maximum(np.asarray(reference_std, dtype=float).reshape(-1), 1e-8)
        if reference_mean.shape != reference_std.shape:
            raise ValueError("Reference mean and standard deviation must have matching shapes.")

        with self._lock:
            total_predictions = self._total_predictions
            inputs = list(self._inputs)
            predictions = list(self._predictions)
            uncertainties = list(self._uncertainties)

        if not inputs:
            shifts = np.zeros_like(reference_mean)
            mean_prediction = None
            mean_uncertainty = None
        else:
            stacked = np.stack(inputs)
            if stacked.shape[1] != len(reference_mean):
                raise ValueError("Reference distribution does not match monitored feature count.")
            shifts = np.abs(stacked.mean(axis=0) - reference_mean) / reference_std
            mean_prediction = float(np.mean(predictions))
            mean_uncertainty = float(np.mean(uncertainties))

        max_shift = float(np.max(shifts, initial=0.0))
        window_size = len(inputs)
        return MonitoringSnapshot(
            total_predictions=total_predictions,
            window_size=window_size,
            mean_prediction=mean_prediction,
            mean_predictive_uncertainty=mean_uncertainty,
            feature_mean_shift=shifts.astype(float).tolist(),
            max_feature_mean_shift=max_shift,
            drift_detected=(
                window_size >= self.minimum_drift_samples
                and max_shift >= self.drift_threshold
            ),
            drift_threshold=self.drift_threshold,
            minimum_drift_samples=self.minimum_drift_samples,
        )

    def reset(self) -> None:
        """Clear process-local state, primarily for tests and controlled rollovers."""

        with self._lock:
            self._inputs.clear()
            self._predictions.clear()
            self._uncertainties.clear()
            self._total_predictions = 0
