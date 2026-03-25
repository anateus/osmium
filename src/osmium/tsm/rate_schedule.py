import numpy as np
from scipy.ndimage import gaussian_filter1d


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
    smoothing_sigma: float = 15.0,
    max_rate_change: float = 0.3,
    gamma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    imp = importance.copy()
    imp = np.clip(imp, 0, 1)

    if smoothing_sigma > 0 and len(imp) > 1:
        imp = gaussian_filter1d(imp, sigma=smoothing_sigma)
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

    if gamma != 1.0:
        normalized = (rates - min_rate) / (max_rate - min_rate)
        rates = min_rate + (normalized ** gamma) * (max_rate - min_rate)
        rates = np.clip(rates, min_rate, max_rate)
        for _ in range(10):
            output_durations = dt / rates
            current_output = output_durations.sum()
            if abs(current_output - target_output_duration) / target_output_duration < 0.001:
                break
            rates *= current_output / target_output_duration
            rates = np.clip(rates, min_rate, max_rate)

    for i in range(1, len(rates)):
        delta = rates[i] - rates[i - 1]
        if delta > max_rate_change:
            rates[i] = rates[i - 1] + max_rate_change

    if len(rates) > 4:
        rates = gaussian_filter1d(rates, sigma=2.0)
        rates = np.clip(rates, min_rate, max_rate)

    for _ in range(10):
        output_durations = dt / rates
        current_output = output_durations.sum()
        if abs(current_output - target_output_duration) / target_output_duration < 0.002:
            break
        adjustment = current_output / target_output_duration
        rates *= adjustment
        rates = np.clip(rates, min_rate, max_rate)

    return rates, importance_times
