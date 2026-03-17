import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

_vocos = None


def _load_vocos():
    global _vocos
    if _vocos is not None:
        return _vocos
    from vocos import Vocos
    _vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return _vocos


def vocos_stretch(
    samples: np.ndarray,
    speed: float,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    vocos = _load_vocos()

    audio_tensor = torch.from_numpy(samples.copy()).unsqueeze(0)
    features = vocos.feature_extractor(audio_tensor)
    feat_np = features.numpy()

    T = feat_np.shape[2]
    target_T = max(1, int(T / speed))

    source_t = np.arange(T)
    target_t = np.linspace(0, T - 1, target_T)

    mel_2d = feat_np[0]
    interp_fn = interp1d(source_t, mel_2d, axis=1, kind="cubic")
    resampled = interp_fn(target_t)

    if smoothing_sigma > 0:
        resampled = gaussian_filter1d(resampled, sigma=smoothing_sigma, axis=1)

    resampled_tensor = torch.from_numpy(resampled.astype(np.float32)[np.newaxis])
    audio_out = vocos.decode(resampled_tensor)
    return audio_out.squeeze().numpy()


def vocos_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    vocos = _load_vocos()

    audio_tensor = torch.from_numpy(samples.copy()).unsqueeze(0)
    features = vocos.feature_extractor(audio_tensor)
    feat_np = features.numpy()

    T = feat_np.shape[2]
    duration = len(samples) / sample_rate
    mel_times = np.linspace(0, duration, T)

    mel_rates = np.interp(mel_times, rate_times, rate_curve)
    mel_rates = np.maximum(mel_rates, 0.5)

    dt = np.diff(mel_times, prepend=0)
    dt[0] = mel_times[0] if T > 0 else 0
    output_times = np.cumsum(dt / mel_rates)

    total_output_duration = output_times[-1]
    target_T = max(1, int(T * duration / (total_output_duration * mel_rates.mean())))
    target_T = max(1, int(total_output_duration / (duration / T)))

    target_mel_times = np.linspace(0, total_output_duration, target_T)

    source_indices = np.interp(target_mel_times, output_times, np.arange(T))

    mel_2d = feat_np[0]
    resampled = np.zeros((mel_2d.shape[0], target_T), dtype=np.float64)
    for i in range(target_T):
        idx = source_indices[i]
        lo = int(np.floor(idx))
        hi = min(lo + 1, T - 1)
        frac = idx - lo
        resampled[:, i] = mel_2d[:, lo] * (1 - frac) + mel_2d[:, hi] * frac

    interp_fn = interp1d(np.arange(T), mel_2d, axis=1, kind="cubic", fill_value="extrapolate")
    resampled = interp_fn(source_indices)

    if smoothing_sigma > 0:
        resampled = gaussian_filter1d(resampled, sigma=smoothing_sigma, axis=1)

    resampled_tensor = torch.from_numpy(resampled.astype(np.float32)[np.newaxis])
    audio_out = vocos.decode(resampled_tensor)
    return audio_out.squeeze().numpy()
