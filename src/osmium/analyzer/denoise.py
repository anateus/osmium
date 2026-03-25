import numpy as np


def spectral_gate(
    samples: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 2048,
    hop: int = 512,
    noise_percentile: float = 0.12,
    threshold_db: float = 6.0,
    smoothing_frames: int = 3,
    chunk_duration: float = 300.0,
    overlap_duration: float = 0.5,
) -> np.ndarray:
    total_samples = len(samples)
    chunk_samples = int(chunk_duration * sample_rate)

    if total_samples <= chunk_samples:
        return _spectral_gate_chunk(
            samples, sample_rate, n_fft, hop,
            noise_percentile, threshold_db, smoothing_frames,
        )

    overlap_samples = int(overlap_duration * sample_rate)
    parts = []
    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_samples + overlap_samples, total_samples)
        start = max(0, pos - overlap_samples) if pos > 0 else 0
        chunk = samples[start:end]
        processed = _spectral_gate_chunk(
            chunk, sample_rate, n_fft, hop,
            noise_percentile, threshold_db, smoothing_frames,
        )
        ob = pos - start if pos > 0 else 0
        oa = end - (pos + chunk_samples) if end < total_samples else 0
        if pos == 0:
            parts.append(processed[:len(processed) - oa] if oa > 0 else processed)
        else:
            if ob > 0 and parts:
                xf = min(ob, len(parts[-1]), len(processed))
                if xf > 0:
                    fade = np.linspace(0, 1, xf, dtype=np.float32)
                    blended = parts[-1][-xf:] * (1 - fade) + processed[:xf] * fade
                    parts[-1] = parts[-1][:-xf]
                    parts.append(blended)
                trimmed = processed[xf:len(processed) - oa] if oa > 0 else processed[xf:]
            else:
                trimmed = processed[:len(processed) - oa] if oa > 0 else processed
            parts.append(trimmed)
        pos += chunk_samples

    return np.concatenate(parts).astype(np.float32)


def _spectral_gate_chunk(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop: int,
    noise_percentile: float,
    threshold_db: float,
    smoothing_frames: int,
) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    original_len = len(samples)
    padded = np.pad(samples, (n_fft // 2, n_fft // 2), mode="reflect")
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)

    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
    ).copy()
    frames *= window

    spec = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(spec)
    phase = np.angle(spec)

    frame_rms = np.sqrt(np.mean(mag ** 2, axis=1))
    n_noise = max(1, int(n_frames * noise_percentile))
    noise_frame_indices = np.argsort(frame_rms)[:n_noise]
    noise_floor = np.mean(mag[noise_frame_indices], axis=0)

    threshold_linear = 10 ** (threshold_db / 20.0)
    snr = mag / (noise_floor[None, :] + 1e-10)

    gain = np.clip((snr - 1.0) / (threshold_linear - 1.0 + 1e-10), 0.0, 1.0)
    gain = gain ** 2

    if smoothing_frames > 1:
        gain = uniform_filter1d(gain, size=smoothing_frames, axis=0)

    cleaned_mag = mag * gain
    cleaned_spec = cleaned_mag * np.exp(1j * phase)
    cleaned_frames = np.fft.irfft(cleaned_spec, n=n_fft)
    cleaned_frames *= window

    output = np.zeros(len(padded), dtype=np.float32)
    window_sum = np.zeros(len(padded), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        output[start:start + n_fft] += cleaned_frames[i]
        window_sum[start:start + n_fft] += window ** 2

    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum

    trim = n_fft // 2
    output = output[trim:trim + original_len]
    return output.astype(np.float32)
