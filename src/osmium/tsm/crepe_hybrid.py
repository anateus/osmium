import numpy as np
from osmium.analyzer.crepe_mlx import predict as crepe_predict
from osmium.tsm.td_psola import pitch_to_marks, td_psola_stretch, td_psola_variable_rate
from osmium.tsm.phase_vocoder import phase_vocoder_stretch


def crepe_hybrid_stretch(
    samples: np.ndarray,
    speed: float,
    window_size: int = 2048,
    sample_rate: int = 24000,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    pitch_result = crepe_predict(samples, sample_rate, capacity="tiny")

    voiced = pitch_result.confidence > confidence_threshold
    voiced_ratio = voiced.mean()

    if voiced_ratio < 0.05:
        return phase_vocoder_stretch(samples, speed, window_size, sample_rate)

    hop_samples = int(sample_rate * 0.01)
    segments = _segment_by_voicing(
        pitch_result.confidence, confidence_threshold,
        hop_samples, len(samples), sample_rate,
        min_segment_ms=80.0,
    )

    output_parts = []
    crossfade = int(0.005 * sample_rate)

    for seg_start, seg_end, is_voiced in segments:
        chunk = samples[seg_start:seg_end]

        if is_voiced and len(chunk) > int(0.05 * sample_rate):
            chunk_pitch = crepe_predict(chunk, sample_rate, capacity="tiny")
            marks = pitch_to_marks(
                chunk_pitch.pitch, chunk_pitch.confidence,
                sample_rate, hop_samples, confidence_threshold,
            )
            if len(marks) > 2:
                stretched = td_psola_stretch(chunk, marks, speed, sample_rate)
            else:
                stretched = phase_vocoder_stretch(chunk, speed, window_size, sample_rate)
        else:
            ws = min(window_size, max(256, len(chunk) // 2))
            stretched = phase_vocoder_stretch(chunk, speed, ws, sample_rate)

        _crossfade_append(output_parts, stretched, crossfade)

    if not output_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(output_parts)


def crepe_hybrid_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    window_size: int = 2048,
    sample_rate: int = 24000,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    pitch_result = crepe_predict(samples, sample_rate, capacity="tiny")

    hop_samples = int(sample_rate * 0.01)
    segments = _segment_by_voicing(
        pitch_result.confidence, confidence_threshold,
        hop_samples, len(samples), sample_rate,
        min_segment_ms=80.0,
    )

    output_parts = []
    crossfade = int(0.005 * sample_rate)

    for seg_start, seg_end, is_voiced in segments:
        chunk = samples[seg_start:seg_end]
        seg_time = seg_start / sample_rate
        local_rate = float(np.interp(seg_time, rate_times, rate_curve))

        if is_voiced and len(chunk) > int(0.05 * sample_rate):
            chunk_pitch = crepe_predict(chunk, sample_rate, capacity="tiny")
            marks = pitch_to_marks(
                chunk_pitch.pitch, chunk_pitch.confidence,
                sample_rate, hop_samples, confidence_threshold,
            )
            if len(marks) > 2:
                seg_start_time = seg_start / sample_rate
                seg_end_time = seg_end / sample_rate
                mask = (rate_times >= seg_start_time) & (rate_times <= seg_end_time)
                if mask.any():
                    local_times = rate_times[mask] - seg_start_time
                    local_rates = rate_curve[mask]
                else:
                    local_times = np.array([0.0])
                    local_rates = np.array([local_rate])

                stretched = td_psola_variable_rate(
                    chunk, marks, local_rates, local_times, sample_rate,
                )
            else:
                stretched = phase_vocoder_stretch(chunk, local_rate, window_size, sample_rate)
        else:
            ws = min(window_size, max(256, len(chunk) // 2))
            stretched = phase_vocoder_stretch(chunk, local_rate, ws, sample_rate)

        _crossfade_append(output_parts, stretched, crossfade)

    if not output_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(output_parts)


def _segment_by_voicing(
    confidence: np.ndarray,
    threshold: float,
    hop_samples: int,
    total_samples: int,
    sample_rate: int,
    min_segment_ms: float = 80.0,
) -> list[tuple[int, int, bool]]:
    min_frames = max(1, int(min_segment_ms / 10.0))
    voiced = confidence > threshold

    segments = []
    current = voiced[0]
    seg_start_frame = 0

    for i in range(1, len(voiced)):
        if voiced[i] != current:
            seg_start = seg_start_frame * hop_samples
            seg_end = i * hop_samples
            if i - seg_start_frame >= min_frames:
                segments.append((seg_start, min(seg_end, total_samples), bool(current)))
            elif segments:
                prev = segments[-1]
                segments[-1] = (prev[0], min(seg_end, total_samples), prev[2])
            else:
                segments.append((seg_start, min(seg_end, total_samples), bool(current)))
            seg_start_frame = i
            current = voiced[i]

    seg_start = seg_start_frame * hop_samples
    segments.append((seg_start, total_samples, bool(current)))

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[2] == merged[-1][2]:
            merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
        else:
            merged.append(seg)

    return merged


def _crossfade_append(parts: list, new_chunk: np.ndarray, crossfade: int):
    if parts and crossfade > 0 and len(new_chunk) > crossfade:
        prev = parts[-1]
        xf = min(crossfade, len(prev), len(new_chunk))
        if xf > 1:
            fade = np.linspace(0, 1, xf, dtype=np.float32)
            blended = prev[-xf:] * (1 - fade) + new_chunk[:xf] * fade
            parts[-1] = prev[:-xf]
            parts.append(blended)
            parts.append(new_chunk[xf:])
            return
    parts.append(new_chunk)
