import torch
import pytest


@pytest.fixture
def dummy_audio_batch():
    return torch.randn(4, 24000)


def test_phase_reg_model_creates(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    assert model is not None


def test_backbone_is_frozen():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    for param in model.backbone.parameters():
        assert not param.requires_grad


def test_head_is_trainable():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    for param in model.head.parameters():
        assert param.requires_grad


def test_single_optimizer():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.trainer = type("MockTrainer", (), {"max_steps": 10000})()
    opts = model.configure_optimizers()
    if isinstance(opts, tuple):
        optimizers = opts[0]
    else:
        optimizers = opts
    assert len(optimizers) == 1


def test_forward_produces_audio(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.eval()
    with torch.no_grad():
        audio = model(dummy_audio_batch)
    assert audio.shape[0] == 4
    assert audio.dim() == 2


def test_phase_extraction_matches_forward(dummy_audio_batch):
    """Direct phase extraction should produce same audio as head.forward()."""
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(dummy_audio_batch)
        backbone_out = model.backbone(features)

        audio_normal = model.head(backbone_out)

        x_proj = model.head.out(backbone_out).transpose(1, 2)
        mag_raw, phase = x_proj.chunk(2, dim=1)
        mag = torch.exp(mag_raw).clip(max=1e2)
        S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio_manual = model.head.istft(S)

    assert torch.allclose(audio_normal, audio_manual, atol=1e-5)
