import numpy as np
from scipy.signal import butter, sosfiltfilt
from osmium.analyzer.importance import ImportanceMap


def compute_prosodic_envelope(
    samples: np.ndarray,
    sample_rate: int = 24000,
    window_ms: float = 20.0,
    cutoff_hz: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    hop = int(window_ms / 1000.0 * sample_rate)
    n_frames = len(samples) // hop

    rms = np.array([
        np.sqrt(np.mean(samples[i * hop:(i + 1) * hop] ** 2))
        for i in range(n_frames)
    ])

    nyquist = (1000.0 / window_ms) / 2.0
    if cutoff_hz < nyquist and n_frames > 12:
        sos = butter(2, cutoff_hz / nyquist, btype='low', output='sos')
        rms = sosfiltfilt(sos, rms)

    rms = np.maximum(rms, 0.0)
    rms_max = rms.max()
    if rms_max > 0:
        rms /= rms_max

    times = np.arange(n_frames) * (window_ms / 1000.0)
    return rms, times


def apply_prosodic_modulation(
    imp: ImportanceMap,
    samples: np.ndarray,
    sample_rate: int = 24000,
    floor: float = 0.5,
    cutoff_hz: float = 4.0,
) -> ImportanceMap:
    prosody, prosody_times = compute_prosodic_envelope(
        samples, sample_rate, cutoff_hz=cutoff_hz,
    )

    prosody_resampled = np.interp(imp.times, prosody_times, prosody)
    modulated = imp.scores * (floor + (1.0 - floor) * prosody_resampled)
    modulated = np.clip(modulated, 0.0, 1.0)

    return ImportanceMap(
        scores=modulated,
        times=imp.times,
        frame_rate=imp.frame_rate,
        duration=imp.duration,
    )
