import numpy as np
from osmium.tsm.smooth import adaptive_smooth_mel


def test_uniform_compression_matches_fixed_sigma():
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 200).astype(np.float32)
    ratios = np.ones(200)

    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.7, sigma_max=0.7)
    expected = gaussian_filter1d(mel, sigma=0.7, axis=1)
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_high_compression_gets_more_smoothing():
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 200).astype(np.float32)
    ratios = np.ones(200)
    ratios[100:] = 8.0

    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.3, sigma_max=2.5)
    var_low = np.var(np.diff(result[:, :90], axis=1))
    var_high = np.var(np.diff(result[:, 110:], axis=1))
    assert var_high < var_low


def test_returns_same_shape():
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 50).astype(np.float32)
    ratios = np.ones(50) * 3.0
    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.3, sigma_max=2.5)
    assert result.shape == mel.shape


def test_sigma_min_clamped_to_sigma_max():
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 50).astype(np.float32)
    ratios = np.ones(50) * 5.0
    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.2, sigma_max=0.2)
    assert result.shape == mel.shape
