import numpy as np
from concurrent.futures import ProcessPoolExecutor
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
    chunk_duration: float = 3600.0,
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


def _process_chunk(args: tuple) -> tuple[int, np.ndarray]:
    chunk_data, speed, engine, window_size, sample_rate, rate_info, overlap_before, overlap_after = args

    if engine == "psola":
        from osmium.tsm.psola_engine import psola_stretch, variable_rate_psola
        if rate_info is not None:
            rate_curve, rate_times = rate_info
            output = variable_rate_psola(chunk_data, rate_curve, rate_times, sample_rate)
        else:
            output = psola_stretch(chunk_data, speed, sample_rate)
    elif engine == "phase_vocoder":
        from osmium.tsm.phase_vocoder import phase_vocoder_stretch, variable_rate_phase_vocoder
        if rate_info is not None:
            rate_curve, rate_times = rate_info
            output = variable_rate_phase_vocoder(chunk_data, rate_curve, rate_times, window_size, sample_rate)
        else:
            output = phase_vocoder_stretch(chunk_data, speed, window_size, sample_rate)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    out_overlap_before = int(overlap_before / speed) if overlap_before > 0 else 0
    out_overlap_after = int(overlap_after / speed) if overlap_after > 0 else 0

    return (out_overlap_before, out_overlap_after, output)


def process_parallel(
    samples: np.ndarray,
    speed: float,
    engine: str = "phase_vocoder",
    window_size: int = 2048,
    sample_rate: int = 24000,
    chunk_duration: float = 3600.0,
    overlap_duration: float = 1.0,
    rate_curve: np.ndarray | None = None,
    rate_times: np.ndarray | None = None,
    max_workers: int | None = None,
    on_progress=None,
) -> np.ndarray:
    chunks = plan_chunks(len(samples), sample_rate, chunk_duration, overlap_duration)

    if len(chunks) == 1:
        rate_info = (rate_curve, rate_times) if rate_curve is not None else None
        _, _, output = _process_chunk((
            samples, speed, engine, window_size, sample_rate,
            rate_info, 0, 0,
        ))
        return output

    args_list = []
    for chunk in chunks:
        chunk_data = samples[chunk.start_sample:chunk.end_sample]

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
            rate_info = (chunk_rate_curve, chunk_rate_times)
        else:
            rate_info = None

        args_list.append((
            chunk_data, speed, engine, window_size, sample_rate,
            rate_info, chunk.overlap_before, chunk.overlap_after,
        ))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_chunk, args): i for i, args in enumerate(args_list)}
        for future in futures:
            idx = futures[future]
            result = future.result()
            results.append((idx, result))
            if on_progress:
                on_progress(idx + 1, len(chunks))

    results.sort(key=lambda x: x[0])

    output_parts = []
    for i, (_, (ob, oa, data)) in enumerate(results):
        if i == 0:
            if oa > 0:
                output_parts.append(data[:-oa])
            else:
                output_parts.append(data)
        elif i == len(results) - 1:
            if ob > 0:
                prev = output_parts[-1]
                xfade_len = min(ob, len(prev), len(data))
                if xfade_len > 0:
                    fade = np.linspace(0, 1, xfade_len, dtype=np.float32)
                    blended = prev[-xfade_len:] * (1 - fade) + data[:xfade_len] * fade
                    output_parts[-1] = prev[:-xfade_len]
                    output_parts.append(blended)
                output_parts.append(data[xfade_len:])
            else:
                output_parts.append(data)
        else:
            if ob > 0:
                prev = output_parts[-1]
                xfade_len = min(ob, len(prev), len(data))
                if xfade_len > 0:
                    fade = np.linspace(0, 1, xfade_len, dtype=np.float32)
                    blended = prev[-xfade_len:] * (1 - fade) + data[:xfade_len] * fade
                    output_parts[-1] = prev[:-xfade_len]
                    output_parts.append(blended)
                if oa > 0:
                    output_parts.append(data[xfade_len:-oa])
                else:
                    output_parts.append(data[xfade_len:])
            else:
                if oa > 0:
                    output_parts.append(data[:-oa])
                else:
                    output_parts.append(data)

    return np.concatenate(output_parts)
