import numpy as np
from osmium.tsm.hpss import hpss
from osmium.tsm.phase_vocoder import phase_vocoder_stretch, variable_rate_phase_vocoder
from osmium.tsm.wsola import wsola_stretch, variable_rate_wsola


def hybrid_stretch(
    samples: np.ndarray,
    speed: float,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    harmonic, percussive = hpss(samples, window_size=window_size)

    harmonic_stretched = phase_vocoder_stretch(
        harmonic, speed=speed, window_size=window_size, sample_rate=sample_rate,
    )

    percussive_stretched = wsola_stretch(
        percussive, speed=speed, window_size=window_size // 2,
    )

    min_len = min(len(harmonic_stretched), len(percussive_stretched))
    output = harmonic_stretched[:min_len] + percussive_stretched[:min_len]

    return output


def hybrid_variable_rate_stretch(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    harmonic, percussive = hpss(samples, window_size=window_size)

    harmonic_stretched = variable_rate_phase_vocoder(
        harmonic, rate_curve, rate_times,
        window_size=window_size, sample_rate=sample_rate,
    )

    percussive_stretched = variable_rate_wsola(
        percussive, rate_curve, rate_times,
        window_size=window_size // 2, sample_rate=sample_rate,
    )

    min_len = min(len(harmonic_stretched), len(percussive_stretched))
    output = harmonic_stretched[:min_len] + percussive_stretched[:min_len]

    return output
