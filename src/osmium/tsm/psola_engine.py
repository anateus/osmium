import numpy as np


def psola_stretch(
    samples: np.ndarray,
    speed: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    import psola

    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    if speed == 1.0:
        return samples.copy()

    stretched = psola.vocode(
        samples.astype(np.float64),
        sample_rate,
        constant_stretch=speed,
        fmin=40.0,
        fmax=550.0,
    )
    return stretched.astype(np.float32)


def variable_rate_psola(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
    chunk_duration: float = 2.0,
) -> np.ndarray:
    import psola

    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(0.05 * sample_rate)
    output_chunks = []

    pos = 0
    while pos < len(samples):
        end = min(pos + chunk_samples, len(samples))
        chunk = samples[pos:end]
        chunk_time = pos / sample_rate

        local_rate = float(np.interp(chunk_time, rate_times, rate_curve))

        stretched = psola.vocode(
            chunk.astype(np.float64),
            sample_rate,
            constant_stretch=local_rate,
            fmin=40.0,
            fmax=550.0,
        ).astype(np.float32)

        if output_chunks and overlap_samples > 0:
            prev = output_chunks[-1]
            xfade_len = min(overlap_samples, len(prev), len(stretched))
            if xfade_len > 0:
                fade = np.linspace(0, 1, xfade_len, dtype=np.float32)
                stretched[:xfade_len] = prev[-xfade_len:] * (1 - fade) + stretched[:xfade_len] * fade
                output_chunks[-1] = prev[:-xfade_len]

        output_chunks.append(stretched)
        pos = end

    if not output_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(output_chunks)
