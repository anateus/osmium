import numpy as np
from scipy.ndimage import maximum_filter1d
from osmium.analyzer.importance import ImportanceMap


def compute_mel_importance(
    mel: np.ndarray,
    duration: float,
    weight_flux: float = 0.50,
    weight_energy: float = 0.25,
    weight_hf_energy: float = 0.15,
    weight_silence_protect: float = 0.10,
    hf_boost: float = 2.5,
    peak_spread: int = 7,
    hf_start_bin: float = 0.6,
    silence_threshold: float = 0.05,
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

    if peak_spread > 1:
        flux = maximum_filter1d(flux, size=peak_spread)

    energy = np.sum(mel, axis=0)
    e_min, e_max = energy.min(), energy.max()
    if e_max > e_min:
        energy = (energy - e_min) / (e_max - e_min)
    else:
        energy = np.zeros(T)

    hf_start = int(n_mels * hf_start_bin)
    hf_energy = np.sum(mel[hf_start:, :], axis=0)
    hf_min, hf_max = hf_energy.min(), hf_energy.max()
    if hf_max > hf_min:
        hf_energy = (hf_energy - hf_min) / (hf_max - hf_min)
    else:
        hf_energy = np.zeros(T)

    silence = (energy < silence_threshold).astype(np.float32)
    closure_mask = maximum_filter1d(flux, size=peak_spread) > 0.3
    protected_silence = silence * closure_mask.astype(np.float32)

    scores = (
        weight_flux * flux
        + weight_energy * energy
        + weight_hf_energy * hf_energy
        + weight_silence_protect * protected_silence
    )
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    scores = np.clip(scores, 0.0, 1.0)
    times = np.linspace(0, duration, T)

    return ImportanceMap(
        scores=scores,
        times=times,
        frame_rate=T / duration,
        duration=duration,
    )
