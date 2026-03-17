import numpy as np


def uniform_rate_schedule(
    duration: float,
    speed: float,
    resolution: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = max(1, int(duration / resolution))
    times = np.linspace(0, duration, n_frames)
    rates = np.full(n_frames, speed)
    return rates, times


def importance_to_rate_schedule(
    importance: np.ndarray,
    importance_times: np.ndarray,
    target_speed: float,
    min_rate: float = 1.0,
    max_rate: float = 10.0,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    imp = importance.copy()
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        imp = np.convolve(imp, kernel, mode="same")

    imp = np.clip(imp, 0, 1)

    inv_importance = 1.0 - imp

    raw_rates = min_rate + inv_importance * (max_rate - min_rate)

    dt = np.diff(importance_times, prepend=0)
    dt[0] = importance_times[0] if len(importance_times) > 0 else 0
    if dt.sum() == 0:
        return np.full_like(importance, target_speed), importance_times

    total_input_duration = importance_times[-1] if len(importance_times) > 0 else 1.0
    target_output_duration = total_input_duration / target_speed

    output_durations = dt / raw_rates
    current_output = output_durations.sum()

    if current_output > 0:
        scale = current_output / target_output_duration
        raw_rates *= scale

    rates = np.clip(raw_rates, min_rate, max_rate)

    for _ in range(20):
        output_durations = dt / rates
        current_output = output_durations.sum()
        if abs(current_output - target_output_duration) / target_output_duration < 0.001:
            break
        adjustment = current_output / target_output_duration
        rates *= adjustment
        rates = np.clip(rates, min_rate, max_rate)

    return rates, importance_times
