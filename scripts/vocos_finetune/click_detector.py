import numpy as np


def count_clicks(audio: np.ndarray, sample_rate: int = 24000, window_samples: int = 48,
                 median_window_ms: float = 50.0, threshold: float = 3.0) -> int:
    n_frames = len(audio) // window_samples
    if n_frames < 3:
        return 0
    trimmed = audio[: n_frames * window_samples]
    frames = trimmed.reshape(n_frames, window_samples)
    energy = np.mean(frames ** 2, axis=1)
    median_frames = max(3, int(median_window_ms / (window_samples / sample_rate * 1000)))
    if median_frames % 2 == 0:
        median_frames += 1
    half = median_frames // 2
    clicks = 0
    for i in range(half, n_frames - half):
        local_med = np.median(energy[i - half : i + half + 1])
        if local_med > 0 and energy[i] > threshold * local_med:
            clicks += 1
    return clicks


def clicks_per_second(audio: np.ndarray, sample_rate: int = 24000, **kwargs) -> float:
    duration = len(audio) / sample_rate
    if duration < 0.01:
        return 0.0
    return count_clicks(audio, sample_rate=sample_rate, **kwargs) / duration
