import numpy as np

from titrate.serving.monitoring import InferenceMonitor


def test_monitor_window_is_bounded_but_total_is_cumulative() -> None:
    monitor = InferenceMonitor(max_window=3, minimum_drift_samples=2, drift_threshold=1.0)
    inputs = np.linspace(0.1, 0.9, 5).reshape(-1, 1)
    monitor.record(inputs, np.arange(5, dtype=float), np.full(5, 0.1))

    snapshot = monitor.snapshot(np.array([0.5]), np.array([0.25]))

    assert snapshot.total_predictions == 5
    assert snapshot.window_size == 3
    assert snapshot.mean_prediction == 3.0


def test_drift_requires_enough_samples_then_detects_a_shift() -> None:
    monitor = InferenceMonitor(max_window=50, minimum_drift_samples=5, drift_threshold=1.0)
    shifted = np.full((5, 2), 0.9)
    predictions = np.full(5, 0.8)
    uncertainties = np.full(5, 0.03)

    monitor.record(shifted[:4], predictions[:4], uncertainties[:4])
    early = monitor.snapshot(np.array([0.5, 0.5]), np.array([0.2, 0.2]))
    monitor.record(shifted[4:], predictions[4:], uncertainties[4:])
    ready = monitor.snapshot(np.array([0.5, 0.5]), np.array([0.2, 0.2]))

    assert np.isclose(early.max_feature_mean_shift, 2.0)
    assert early.drift_detected is False
    assert ready.drift_detected is True


def test_reset_clears_process_local_monitoring_state() -> None:
    monitor = InferenceMonitor()
    monitor.record(np.array([[0.5]]), np.array([0.7]), np.array([0.02]))
    monitor.reset()

    snapshot = monitor.snapshot(np.array([0.5]), np.array([0.2]))
    assert snapshot.total_predictions == 0
    assert snapshot.window_size == 0
    assert snapshot.mean_prediction is None
    assert snapshot.mean_predictive_uncertainty is None
