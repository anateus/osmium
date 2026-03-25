import numpy as np
from osmium.analyzer.denoise import spectral_gate


def _make_noisy_speech(sr=24000, duration=2.0, noise_level=0.03):
    rng = np.random.RandomState(42)
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    f0 = 120 + 30 * np.sin(2 * np.pi * 2 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    harmonics = np.zeros_like(t)
    for k in range(1, 8):
        harmonics += (0.4 / k) * np.sin(k * phase)
    am = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t), 0.1, 1.0).astype(np.float32)
    speech = harmonics.astype(np.float32) * am * 0.3
    speech[int(0.8 * sr):int(1.2 * sr)] = 0
    noise = noise_level * rng.randn(len(t)).astype(np.float32)
    return speech + noise, speech, sr


def test_spectral_gate_reduces_noise():
    noisy, _, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    gap = slice(int(0.8 * sr), int(1.2 * sr))
    input_noise = np.sqrt(np.mean(noisy[gap] ** 2))
    output_noise = np.sqrt(np.mean(denoised[gap] ** 2))
    assert output_noise < input_noise * 0.5


def test_spectral_gate_preserves_speech():
    noisy, _, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    speech_region = slice(int(0.2 * sr), int(0.7 * sr))
    input_rms = np.sqrt(np.mean(noisy[speech_region] ** 2))
    output_rms = np.sqrt(np.mean(denoised[speech_region] ** 2))
    assert output_rms > input_rms * 0.2


def test_spectral_gate_returns_same_length():
    noisy, _, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    assert len(denoised) == len(noisy)


def test_spectral_gate_handles_silence():
    silence = np.zeros(24000, dtype=np.float32)
    result = spectral_gate(silence, 24000)
    assert len(result) == 24000
    assert np.max(np.abs(result)) < 1e-6
