import numpy as np
from dataclasses import dataclass


@dataclass
class ChunkSpec:
    index: int
    start_sample: int
    end_sample: int
    overlap_before: int
    overlap_after: int


def plan_chunks(
    total_samples: int,
    sample_rate: int,
    chunk_duration: float = 300.0,
    overlap_duration: float = 1.0,
) -> list[ChunkSpec]:
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)

    if total_samples <= chunk_samples:
        return [ChunkSpec(0, 0, total_samples, 0, 0)]

    chunks = []
    pos = 0
    idx = 0
    while pos < total_samples:
        start = max(0, pos - overlap_samples) if idx > 0 else 0
        end = min(pos + chunk_samples + overlap_samples, total_samples)

        overlap_before = pos - start if idx > 0 else 0
        overlap_after = end - (pos + chunk_samples) if end < total_samples else 0

        chunks.append(ChunkSpec(idx, start, end, overlap_before, max(0, overlap_after)))
        pos += chunk_samples
        idx += 1

    return chunks


def process_chunked(
    samples: np.ndarray,
    speed: float,
    sample_rate: int = 24000,
    chunk_duration: float = 300.0,
    overlap_duration: float = 1.0,
    rate_curve: np.ndarray | None = None,
    rate_times: np.ndarray | None = None,
    smoothing: float = 0.7,
    on_progress=None,
) -> np.ndarray:
    chunks = plan_chunks(len(samples), sample_rate, chunk_duration, overlap_duration)

    try:
        from osmium.tsm.vocos_mlx import vocos_mlx_stretch, vocos_mlx_variable_rate
        stretch_fn = vocos_mlx_stretch
        vr_fn = vocos_mlx_variable_rate
    except (ImportError, Exception):
        from osmium.tsm.vocos_engine import vocos_stretch, vocos_variable_rate
        stretch_fn = vocos_stretch
        vr_fn = vocos_variable_rate

    output_parts = []
    for chunk in chunks:
        chunk_data = samples[chunk.start_sample:chunk.end_sample].copy()

        if rate_curve is not None and rate_times is not None:
            chunk_start_time = chunk.start_sample / sample_rate
            chunk_end_time = chunk.end_sample / sample_rate
            mask = (rate_times >= chunk_start_time) & (rate_times <= chunk_end_time)
            if mask.any():
                chunk_rate_times = rate_times[mask] - chunk_start_time
                chunk_rate_curve = rate_curve[mask]
            else:
                chunk_rate_times = np.array([0.0, chunk_end_time - chunk_start_time])
                avg_rate = float(np.interp(
                    (chunk_start_time + chunk_end_time) / 2, rate_times, rate_curve
                ))
                chunk_rate_curve = np.array([avg_rate, avg_rate])
            output = vr_fn(chunk_data, chunk_rate_curve, chunk_rate_times, sample_rate, smoothing)
        else:
            output = stretch_fn(chunk_data, speed, sample_rate, smoothing)

        ob = int(chunk.overlap_before / speed) if chunk.overlap_before > 0 else 0
        oa = int(chunk.overlap_after / speed) if chunk.overlap_after > 0 else 0

        _append_with_crossfade(output_parts, output, ob, oa, chunk.index, len(chunks))

        if on_progress:
            on_progress(chunk.index + 1, len(chunks))

    return np.concatenate(output_parts) if output_parts else np.array([], dtype=np.float32)


def _append_with_crossfade(parts, data, ob, oa, idx, total):
    if idx == 0:
        parts.append(data[:-oa] if oa > 0 else data)
    else:
        if ob > 0 and parts:
            prev = parts[-1]
            xf = min(ob, len(prev), len(data))
            if xf > 0:
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                blended = prev[-xf:] * (1 - fade) + data[:xf] * fade
                parts[-1] = prev[:-xf]
                parts.append(blended)
            trimmed = data[xf:-oa] if oa > 0 else data[xf:]
        else:
            trimmed = data[:-oa] if oa > 0 else data
        parts.append(trimmed)
