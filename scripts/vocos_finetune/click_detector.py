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


def spectral_transient_clicks(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: float = 2.0,
    median_window_ms: float = 50.0,
    threshold: float = 4.0,
    lowcut_hz: float = 300.0,
) -> int:
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(audio) // frame_samples
    if n_frames < 5:
        return 0

    trimmed = audio[: n_frames * frame_samples]
    frames = trimmed.reshape(n_frames, frame_samples)
    spec = np.abs(np.fft.rfft(frames, axis=1))

    freq_bins = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate)
    low_mask = freq_bins <= lowcut_hz

    broadband_energy = np.mean(spec ** 2, axis=1)
    low_energy = np.mean(spec[:, low_mask] ** 2, axis=1) if low_mask.any() else broadband_energy

    median_frames = max(3, int(median_window_ms / frame_ms))
    if median_frames % 2 == 0:
        median_frames += 1
    half = median_frames // 2

    eps = 1e-10
    clicks = 0
    for i in range(half, n_frames - half):
        bb_med = max(float(np.median(broadband_energy[i - half : i + half + 1])), eps)
        low_med = max(float(np.median(low_energy[i - half : i + half + 1])), eps)
        bb_spike = broadband_energy[i] > threshold * bb_med
        low_spike = low_energy[i] > threshold * low_med
        if bb_spike or low_spike:
            clicks += 1
    return clicks


def clicks_per_second(audio: np.ndarray, sample_rate: int = 24000, **kwargs) -> float:
    duration = len(audio) / sample_rate
    if duration < 0.01:
        return 0.0
    amplitude_clicks = count_clicks(audio, sample_rate=sample_rate, **kwargs)
    spectral_clicks = spectral_transient_clicks(audio, sample_rate=sample_rate)
    return max(amplitude_clicks, spectral_clicks) / duration
