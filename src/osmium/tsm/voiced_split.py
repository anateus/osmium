import numpy as np
from dataclasses import dataclass


@dataclass
class VoicedSegment:
    start: int
    end: int
    is_voiced: bool


def detect_voiced_segments(
    samples: np.ndarray,
    sample_rate: int = 24000,
    frame_size_ms: float = 20.0,
    energy_threshold: float = 0.02,
    zcr_threshold: float = 0.15,
    min_segment_ms: float = 50.0,
) -> list[VoicedSegment]:
    frame_size = int(frame_size_ms * sample_rate / 1000)
    n_frames = len(samples) // frame_size

    voiced = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        start = i * frame_size
        frame = samples[start:start + frame_size]

        energy = np.sqrt(np.mean(frame ** 2))
        if energy < energy_threshold:
            voiced[i] = False
            continue

        signs = np.sign(frame)
        zcr = np.sum(np.abs(np.diff(signs)) > 0) / (len(frame) - 1)

        voiced[i] = zcr < zcr_threshold

    min_frames = max(1, int(min_segment_ms / frame_size_ms))
    segments = []
    if n_frames == 0:
        return [VoicedSegment(0, len(samples), False)]

    current_voiced = voiced[0]
    seg_start = 0
    for i in range(1, n_frames):
        if voiced[i] != current_voiced:
            seg_end = i * frame_size
            if i - seg_start // frame_size >= min_frames or not segments:
                segments.append(VoicedSegment(seg_start, seg_end, bool(current_voiced)))
            elif segments:
                segments[-1] = VoicedSegment(segments[-1].start, seg_end, segments[-1].is_voiced)
            seg_start = seg_end
            current_voiced = voiced[i]

    segments.append(VoicedSegment(seg_start, len(samples), bool(current_voiced)))

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.is_voiced == merged[-1].is_voiced:
            merged[-1] = VoicedSegment(merged[-1].start, seg.end, seg.is_voiced)
        elif (seg.end - seg.start) < min_frames * frame_size:
            merged[-1] = VoicedSegment(merged[-1].start, seg.end, merged[-1].is_voiced)
        else:
            merged.append(seg)

    return merged


def hybrid_voiced_stretch(
    samples: np.ndarray,
    speed: float,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    from osmium.tsm.phase_vocoder import phase_vocoder_stretch
    from osmium.tsm.psola_engine import psola_stretch

    segments = detect_voiced_segments(samples, sample_rate)
    crossfade_samples = int(0.005 * sample_rate)

    output_parts = []
    for seg in segments:
        chunk = samples[seg.start:seg.end]
        min_psola_samples = int(0.1 * sample_rate)
        if len(chunk) < 512:
            stretched = phase_vocoder_stretch(chunk, speed, window_size=256, sample_rate=sample_rate)
        elif seg.is_voiced and len(chunk) >= min_psola_samples:
            try:
                stretched = psola_stretch(chunk, speed, sample_rate)
            except Exception:
                stretched = phase_vocoder_stretch(chunk, speed, window_size=window_size, sample_rate=sample_rate)
        else:
            stretched = phase_vocoder_stretch(chunk, speed, window_size=window_size, sample_rate=sample_rate)

        if output_parts and crossfade_samples > 0:
            prev = output_parts[-1]
            xf = min(crossfade_samples, len(prev), len(stretched))
            if xf > 1:
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                blended = prev[-xf:] * (1 - fade) + stretched[:xf] * fade
                output_parts[-1] = prev[:-xf]
                output_parts.append(blended)
                output_parts.append(stretched[xf:])
            else:
                output_parts.append(stretched)
        else:
            output_parts.append(stretched)

    if not output_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(output_parts)


def hybrid_voiced_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    from osmium.tsm.phase_vocoder import phase_vocoder_stretch
    from osmium.tsm.psola_engine import psola_stretch

    segments = detect_voiced_segments(samples, sample_rate)
    crossfade_samples = int(0.005 * sample_rate)

    output_parts = []
    for seg in segments:
        chunk = samples[seg.start:seg.end]
        seg_time = seg.start / sample_rate
        local_rate = float(np.interp(seg_time, rate_times, rate_curve))

        min_psola_samples = int(0.1 * sample_rate)
        if len(chunk) < 512:
            stretched = phase_vocoder_stretch(chunk, local_rate, window_size=256, sample_rate=sample_rate)
        elif seg.is_voiced and len(chunk) >= min_psola_samples:
            try:
                stretched = psola_stretch(chunk, local_rate, sample_rate)
            except Exception:
                stretched = phase_vocoder_stretch(chunk, local_rate, window_size=window_size, sample_rate=sample_rate)
        else:
            stretched = phase_vocoder_stretch(chunk, local_rate, window_size=window_size, sample_rate=sample_rate)

        if output_parts and crossfade_samples > 0:
            prev = output_parts[-1]
            xf = min(crossfade_samples, len(prev), len(stretched))
            if xf > 1:
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                blended = prev[-xf:] * (1 - fade) + stretched[:xf] * fade
                output_parts[-1] = prev[:-xf]
                output_parts.append(blended)
                output_parts.append(stretched[xf:])
            else:
                output_parts.append(stretched)
        else:
            output_parts.append(stretched)

    if not output_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(output_parts)
