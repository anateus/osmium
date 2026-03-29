import pytest
import torch
import numpy as np


@pytest.fixture
def dummy_audio_batch():
    return torch.randn(4, 24000)


def test_augment_features_changes_mel(dummy_audio_batch):
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model
    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 1.0
    features = model.feature_extractor(dummy_audio_batch)
    augmented = model._maybe_augment(features)
    assert not torch.allclose(features, augmented, atol=1e-3)


def test_augment_features_preserves_shape(dummy_audio_batch):
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model
    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 1.0
    features = model.feature_extractor(dummy_audio_batch)
    augmented = model._maybe_augment(features)
    assert augmented.shape == features.shape


def test_no_augment_when_ratio_zero(dummy_audio_batch):
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model
    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 0.0
    features = model.feature_extractor(dummy_audio_batch)
    result = model._maybe_augment(features)
    assert torch.allclose(features, result)


def test_aug_ratio_ramp():
    from scripts.vocos_finetune.train import compute_aug_ratio
    assert abs(compute_aug_ratio(0) - 0.3) < 1e-6
    assert abs(compute_aug_ratio(2000) - 0.3) < 1e-6
    assert abs(compute_aug_ratio(3000) - 0.4) < 1e-6
    assert abs(compute_aug_ratio(4000) - 0.5) < 1e-6
    assert abs(compute_aug_ratio(8000) - 0.5) < 1e-6


def test_validation_step_returns_separate_metrics(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_model
    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.eval()
    with torch.no_grad():
        result = model.validation_step(dummy_audio_batch, batch_idx=0)
    assert "mel_loss_normal" in result
    assert "mel_loss_augmented" in result
    assert "mel_loss_2_0x" in result
    assert "mel_loss_3_0x" in result
    assert "mel_loss_4_0x" in result
    assert "click_rate_2_0x" in result
    assert isinstance(result["mel_loss_normal"], torch.Tensor)
