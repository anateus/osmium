import numpy as np
import inspect
from osmium.analyzer.mel_importance import compute_mel_importance


def test_default_hf_boost_is_2_5():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["hf_boost"].default == 2.5


def test_default_flux_weight_is_0_65():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_flux"].default == 0.65


def test_default_energy_weight_is_0_35():
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_energy"].default == 0.35


def test_higher_hf_boost_increases_consonant_importance():
    n_mels, T = 100, 30
    mel = np.zeros((n_mels, T), dtype=np.float32)
    mel[:n_mels // 2, 5] = 3.0
    mel[n_mels // 2:, 15] = 1.0

    imp_low = compute_mel_importance(mel, 1.0, hf_boost=2.0)
    imp_high = compute_mel_importance(mel, 1.0, hf_boost=2.5)
    assert imp_high.scores[15] > imp_low.scores[15]
