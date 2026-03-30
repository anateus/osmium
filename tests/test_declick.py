import numpy as np
from osmium.tsm.declick import declick


def test_silent_audio_unchanged():
    audio = np.zeros(24000, dtype=np.float32)
    result = declick(audio, sample_rate=24000)
    np.testing.assert_array_equal(audio, result)


def test_smooth_sine_unchanged():
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    result = declick(audio, sample_rate=sr)
    np.testing.assert_allclose(audio, result, atol=1e-6)


def test_impulse_is_attenuated():
    sr = 24000
    audio = np.full(sr, 0.01, dtype=np.float32)
    mid = sr // 2
    audio[mid - 5 : mid + 5] = 1.0
    result = declick(audio, sample_rate=sr)
    original_peak = np.max(np.abs(audio[mid - 5 : mid + 5]))
    result_peak = np.max(np.abs(result[mid - 5 : mid + 5]))
    assert result_peak < original_peak * 0.8


def test_output_same_length():
    sr = 24000
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.1
    audio[sr] = 5.0
    result = declick(audio, sample_rate=sr)
    assert len(result) == len(audio)


def test_short_audio_returned_unchanged():
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = declick(audio, sample_rate=24000)
    np.testing.assert_array_equal(audio, result)


def test_multiple_clicks_all_attenuated():
    sr = 24000
    audio = np.full(sr, 0.01, dtype=np.float32)
    for pos in [sr // 4, sr // 2, 3 * sr // 4]:
        audio[pos - 3 : pos + 3] = 1.0
    result = declick(audio, sample_rate=sr)
    for pos in [sr // 4, sr // 2, 3 * sr // 4]:
        assert np.max(np.abs(result[pos - 3 : pos + 3])) < 0.8
