import numpy as np
import inspect
from osmium.analyzer.mel_importance import compute_mel_importance


def test_default_hf_boost_is_2_5():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["hf_boost"].default == 2.5


def test_default_flux_weight_is_0_50():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_flux"].default == 0.50


def test_default_energy_weight_is_0_25():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_energy"].default == 0.25


def test_default_peak_spread_is_7():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["peak_spread"].default == 7


def test_higher_hf_boost_increases_consonant_importance():
    n_mels, T = 100, 30
    mel = np.zeros((n_mels, T), dtype=np.float32)
    mel[:n_mels // 2, 5] = 3.0
    mel[n_mels // 2:, 15] = 1.0

    imp_low = compute_mel_importance(mel, 1.0, hf_boost=2.0)
    imp_high = compute_mel_importance(mel, 1.0, hf_boost=2.5)
    assert imp_high.scores[15] > imp_low.scores[15]


def test_scores_normalized_to_full_range():
    n_mels, T = 100, 50
    np.random.seed(42)
    mel = np.random.randn(n_mels, T).astype(np.float32)
    imp = compute_mel_importance(mel, 1.0)
    assert imp.scores.max() >= 0.99
    assert imp.scores.min() <= 0.01


def test_peak_spread_widens_consonant_peaks():
    n_mels, T = 100, 50
    mel = np.zeros((n_mels, T), dtype=np.float32)
    mel[:, 25] = 5.0

    imp_no_spread = compute_mel_importance(mel, 1.0, peak_spread=1)
    imp_spread = compute_mel_importance(mel, 1.0, peak_spread=7)
    above_half_no_spread = np.sum(imp_no_spread.scores > 0.5)
    above_half_spread = np.sum(imp_spread.scores > 0.5)
    assert above_half_spread > above_half_no_spread
