import numpy as np
from osmium.analyzer.importance import ImportanceMap


def compute_mel_importance(
    mel: np.ndarray,
    duration: float,
    weight_flux: float = 0.6,
    weight_energy: float = 0.4,
    hf_boost: float = 2.0,
) -> ImportanceMap:
    T = mel.shape[1]
    n_mels = mel.shape[0]

    hf_weights = np.ones(n_mels, dtype=np.float32)
    hf_weights[n_mels // 2:] = hf_boost

    weighted_mel = mel * hf_weights[:, None]
    flux = np.sqrt(np.sum(np.diff(weighted_mel, axis=1) ** 2, axis=0))
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
