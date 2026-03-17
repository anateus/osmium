import numpy as np
from scipy.ndimage import median_filter


def hpss(
    samples: np.ndarray,
    window_size: int = 2048,
    hop: int | None = None,
    harmonic_kernel: int = 31,
    percussive_kernel: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if hop is None:
        hop = window_size // 4

    window = np.hanning(window_size).astype(np.float64)
    half_win = window_size // 2

    padded = np.pad(samples, (half_win, half_win), mode="constant")
    n_frames = 1 + (len(padded) - window_size) // hop

    stft = np.zeros((window_size // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + window_size] * window
        stft[:, i] = np.fft.rfft(frame)

    magnitude = np.abs(stft)
    mag_power = magnitude ** power

    harmonic_mag = median_filter(mag_power, size=(1, harmonic_kernel))
    percussive_mag = median_filter(mag_power, size=(percussive_kernel, 1))

    harmonic_mag = np.maximum(harmonic_mag, 1e-10)
    percussive_mag = np.maximum(percussive_mag, 1e-10)

    harmonic_mask = (harmonic_mag / (harmonic_mag + percussive_mag)) ** margin
    percussive_mask = (percussive_mag / (harmonic_mag + percussive_mag)) ** margin

    total_mask = harmonic_mask + percussive_mask
    harmonic_mask /= total_mask
    percussive_mask /= total_mask

    harmonic_stft = stft * harmonic_mask
    percussive_stft = stft * percussive_mask

    harmonic_out = _istft(harmonic_stft, window, hop, half_win, len(samples))
    percussive_out = _istft(percussive_stft, window, hop, half_win, len(samples))

    return harmonic_out.astype(np.float32), percussive_out.astype(np.float32)


def _istft(
    stft: np.ndarray,
    window: np.ndarray,
    hop: int,
    half_win: int,
    original_length: int,
) -> np.ndarray:
    window_size = len(window)
    n_frames = stft.shape[1]
    output_length = (n_frames - 1) * hop + window_size
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)

    for i in range(n_frames):
        frame = np.fft.irfft(stft[:, i], n=window_size) * window
        start = i * hop
        output[start:start + window_size] += frame
        window_sum[start:start + window_size] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    output = output[half_win:half_win + original_length]
    return output
