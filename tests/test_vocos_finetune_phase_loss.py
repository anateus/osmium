import torch
import math
import pytest

from scripts.vocos_finetune.phase_loss import InstantaneousFrequencyDeviationLoss


@pytest.fixture
def ifd_loss():
    return InstantaneousFrequencyDeviationLoss(n_fft=1024, hop_length=256)


def test_zero_loss_for_expected_phase_advance(ifd_loss):
    """Phase that advances exactly at bin-center frequency should give near-zero loss."""
    B, N, T = 2, 513, 20
    n_fft, hop = 1024, 256
    k = torch.arange(N).float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    expected_advance = 2 * math.pi * k * hop / n_fft
    t = torch.arange(T).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    phase = expected_advance * t  # (1, N, T) -> broadcast to (B, N, T)
    phase = phase.expand(B, N, T)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() < 1e-5


def test_nonzero_loss_for_random_phase(ifd_loss):
    """Random phase should give nonzero loss."""
    B, N, T = 2, 513, 20
    phase = torch.randn(B, N, T) * math.pi
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() > 0.01


def test_dc_and_nyquist_excluded(ifd_loss):
    """DC (k=0) and Nyquist (k=512) should not contribute to loss."""
    B, N, T = 1, 513, 20
    n_fft, hop = 1024, 256
    k = torch.arange(N).float().unsqueeze(0).unsqueeze(-1)
    expected_advance = 2 * math.pi * k * hop / n_fft
    t = torch.arange(T).float().unsqueeze(0).unsqueeze(0)
    phase = (expected_advance * t).expand(B, N, T).clone()
    phase[:, 0, :] = torch.randn(1, T) * 100  # wild DC phase
    phase[:, -1, :] = torch.randn(1, T) * 100  # wild Nyquist phase
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() < 1e-5


def test_magnitude_weighting_suppresses_silent_bins(ifd_loss):
    """Bins with zero magnitude should not contribute to loss."""
    B, N, T = 1, 513, 20
    phase = torch.randn(B, N, T) * math.pi
    mag_silent = torch.zeros(B, N, T)
    mag_silent[:, 100, :] = 1.0  # only one bin has energy
    loss_silent = ifd_loss(phase, mag_silent)

    mag_full = torch.ones(B, N, T)
    loss_full = ifd_loss(phase, mag_full)
    assert loss_silent < loss_full


def test_loss_is_differentiable(ifd_loss):
    """Loss should be differentiable w.r.t. phase."""
    B, N, T = 1, 513, 10
    phase = torch.randn(B, N, T, requires_grad=True)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    loss.backward()
    assert phase.grad is not None
    assert not torch.isnan(phase.grad).any()


def test_output_is_scalar(ifd_loss):
    """Loss should return a scalar tensor."""
    B, N, T = 2, 513, 15
    phase = torch.randn(B, N, T)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.dim() == 0
