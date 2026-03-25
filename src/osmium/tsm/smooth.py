import numpy as np
from scipy.ndimage import gaussian_filter1d


def adaptive_smooth_mel(
    mel: np.ndarray,
    compression_ratios: np.ndarray,
    sigma_min: float = 0.3,
    sigma_max: float = 2.5,
    n_buckets: int = 4,
    crossfade_frames: int = 4,
) -> np.ndarray:
    sigma_min = min(sigma_min, sigma_max)
    T = mel.shape[1]

    if T < 2 or sigma_max <= 0:
        return mel.copy()

    ratio_min = compression_ratios.min()
    ratio_max = compression_ratios.max()

    if ratio_max <= ratio_min:
        sigma = sigma_min
        return gaussian_filter1d(mel, sigma=max(sigma, 0.01), axis=1)

    norm_ratios = (compression_ratios - ratio_min) / (ratio_max - ratio_min)
    per_frame_sigma = sigma_min + norm_ratios * (sigma_max - sigma_min)

    bucket_sigmas = np.linspace(sigma_min, sigma_max, n_buckets)
    bucket_indices = np.argmin(
        np.abs(per_frame_sigma[:, None] - bucket_sigmas[None, :]), axis=1
    )

    smoothed_layers = np.stack([
        gaussian_filter1d(mel, sigma=max(s, 0.01), axis=1)
        for s in bucket_sigmas
    ])

    result = np.empty_like(mel)
    for t in range(T):
        result[:, t] = smoothed_layers[bucket_indices[t], :, t]

    if crossfade_frames > 0 and T > crossfade_frames * 2:
        changes = np.where(np.diff(bucket_indices) != 0)[0]
        for c in changes:
            half = crossfade_frames // 2
            start = max(0, c + 1 - half)
            end = min(T, c + 1 + half)
            length = end - start
            if length < 2:
                continue
            fade = np.linspace(0, 1, length, dtype=np.float32)
            b_before = bucket_indices[c]
            b_after = bucket_indices[min(c + 1, T - 1)]
            blended = (
                smoothed_layers[b_before, :, start:end] * (1 - fade)[None, :]
                + smoothed_layers[b_after, :, start:end] * fade[None, :]
            )
            result[:, start:end] = blended

    return result


def apply_hf_deemphasis(
    mel: np.ndarray,
    compression_ratios: np.ndarray,
    max_rolloff_db: float = 4.0,
    hf_start_bin: float = 0.5,
) -> np.ndarray:
    n_mels, T = mel.shape
    if T < 2 or max_rolloff_db <= 0:
        return mel

    ratio_min = compression_ratios.min()
    ratio_max = compression_ratios.max()
    if ratio_max <= ratio_min + 0.1:
        return mel

    norm_ratios = np.clip(
        (compression_ratios - ratio_min) / (ratio_max - ratio_min), 0, 1
    )

    start_bin = int(n_mels * hf_start_bin)
    bin_positions = np.linspace(0, 1, n_mels - start_bin)
    rolloff_profile = bin_positions ** 1.5

    gain_db = np.zeros((n_mels, T), dtype=np.float32)
    gain_db[start_bin:, :] = -max_rolloff_db * rolloff_profile[:, None] * norm_ratios[None, :]

    return mel + gain_db
