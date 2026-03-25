import numpy as np
from osmium.analyzer.denoise import spectral_gate


def _make_noisy_speech(sr=24000, duration=2.0, noise_level=0.05):
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    speech = 0.5 * np.sin(2 * np.pi * 300 * t)
    envelope = np.ones_like(t)
    envelope[int(0.8 * sr):int(1.2 * sr)] = 0
    speech *= envelope
    noise = noise_level * np.random.RandomState(42).randn(len(t)).astype(np.float32)
    return speech + noise, speech, sr


def test_spectral_gate_reduces_noise():
    noisy, clean, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    gap = slice(int(0.8 * sr), int(1.2 * sr))
    input_noise = np.sqrt(np.mean(noisy[gap] ** 2))
    output_noise = np.sqrt(np.mean(denoised[gap] ** 2))
    assert output_noise < input_noise * 0.5


def test_spectral_gate_preserves_speech():
    noisy, clean, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    speech_region = slice(0, int(0.8 * sr))
    input_rms = np.sqrt(np.mean(noisy[speech_region] ** 2))
    output_rms = np.sqrt(np.mean(denoised[speech_region] ** 2))
    assert output_rms > input_rms * 0.7


def test_spectral_gate_returns_same_length():
    noisy, _, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    assert len(denoised) == len(noisy)


def test_spectral_gate_handles_silence():
    silence = np.zeros(24000, dtype=np.float32)
    result = spectral_gate(silence, 24000)
    assert len(result) == 24000
    assert np.max(np.abs(result)) < 1e-6
