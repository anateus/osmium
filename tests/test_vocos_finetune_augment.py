import torch
import pytest


def test_roundtrip_preserves_shape():
    from scripts.vocos_finetune.augment import resample_roundtrip
    mel = torch.randn(2, 100, 94)
    result = resample_roundtrip(mel, rate=3.0)
    assert result.shape == mel.shape


def test_roundtrip_modifies_content():
    from scripts.vocos_finetune.augment import resample_roundtrip
    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=3.0)
    assert not torch.allclose(mel, result, atol=1e-3)


def test_roundtrip_at_rate_1_is_near_identity():
    from scripts.vocos_finetune.augment import resample_roundtrip
    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=1.001)
    assert torch.allclose(mel, result, atol=0.1)


def test_higher_rate_produces_more_distortion():
    from scripts.vocos_finetune.augment import resample_roundtrip
    mel = torch.randn(1, 100, 200)
    result_2x = resample_roundtrip(mel, rate=2.0)
    result_5x = resample_roundtrip(mel, rate=5.0)
    dist_2x = (mel - result_2x).abs().mean()
    dist_5x = (mel - result_5x).abs().mean()
    assert dist_5x > dist_2x


def test_roundtrip_with_presmooth():
    from scripts.vocos_finetune.augment import resample_roundtrip
    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=3.0, presmooth_sigma=2.0)
    assert result.shape == mel.shape
    no_smooth = resample_roundtrip(mel, rate=3.0, presmooth_sigma=0.0)
    smooth_var = result.diff(dim=-1).var()
    no_smooth_var = no_smooth.diff(dim=-1).var()
    assert smooth_var < no_smooth_var


def test_random_roundtrip_produces_valid_output():
    from scripts.vocos_finetune.augment import random_resample_roundtrip
    mel = torch.randn(4, 100, 94)
    result = random_resample_roundtrip(mel, min_rate=1.5, max_rate=5.0)
    assert result.shape == mel.shape
    assert torch.isfinite(result).all()
