import numpy as np
import pytest


def test_spectral_transient_detects_impulse():
    """A single-sample impulse should register as a transient click."""
    from scripts.vocos_finetune.click_detector import spectral_transient_clicks
    sr = 24000
    audio = np.zeros(sr, dtype=np.float32)
    audio[sr // 2] = 1.0
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks >= 1


def test_spectral_transient_silent_audio():
    """Silent audio should have zero transient clicks."""
    from scripts.vocos_finetune.click_detector import spectral_transient_clicks
    sr = 24000
    audio = np.zeros(sr, dtype=np.float32)
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks == 0


def test_spectral_transient_smooth_sine():
    """A smooth sine wave should have zero transient clicks."""
    from scripts.vocos_finetune.click_detector import spectral_transient_clicks
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks == 0


def test_clean_audio_has_no_clicks():
    from scripts.vocos_finetune.click_detector import count_clicks
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    clicks = count_clicks(audio, sample_rate=24000)
    assert clicks == 0


def test_audio_with_spikes_detected():
    from scripts.vocos_finetune.click_detector import count_clicks
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.1
    audio[5000] = 0.9
    audio[12000] = 0.9
    audio[18000] = 0.9
    clicks = count_clicks(audio, sample_rate=24000)
    assert clicks >= 2


def test_clicks_per_second():
    from scripts.vocos_finetune.click_detector import clicks_per_second
    t = np.linspace(0, 2, 48000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.1
    for pos in [5000, 15000, 25000, 40000]:
        audio[pos] = 0.9
    cps = clicks_per_second(audio, sample_rate=24000)
    assert 1.0 <= cps <= 3.0
