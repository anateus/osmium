import numpy as np
from osmium.analyzer.importance import ImportanceMap


def compute_mel_importance(
    mel: np.ndarray,
    duration: float,
    weight_flux: float = 0.5,
    weight_energy: float = 0.5,
) -> ImportanceMap:
    T = mel.shape[1]

    flux = np.sqrt(np.sum(np.diff(mel, axis=1) ** 2, axis=0))
    flux = np.concatenate(([0.0], flux))
    flux_max = flux.max()
    if flux_max > 0:
        flux /= flux_max

    energy = np.sum(mel, axis=0)
    e_min, e_max = energy.min(), energy.max()
    if e_max > e_min:
        energy = (energy - e_min) / (e_max - e_min)
    else:
        energy = np.zeros(T)

    scores = weight_flux * flux + weight_energy * energy
    scores = np.clip(scores, 0.0, 1.0)
    times = np.linspace(0, duration, T)

    return ImportanceMap(
        scores=scores,
        times=times,
        frame_rate=T / duration,
        duration=duration,
    )
