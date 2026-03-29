import math

import torch
from torch import nn


class InstantaneousFrequencyDeviationLoss(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_bins = n_fft // 2 + 1
        k = torch.arange(1, n_bins - 1).float()
        expected = 2 * math.pi * k * hop_length / n_fft
        self.register_buffer("expected_advance", expected)

    def forward(self, phase: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        phase_inner = phase[:, 1:-1, :]
        mag_inner = mag[:, 1:-1, :]

        actual_advance = phase_inner[:, :, 1:] - phase_inner[:, :, :-1]
        expected = self.expected_advance[None, :, None]
        raw_deviation = actual_advance - expected
        deviation = torch.atan2(torch.sin(raw_deviation), torch.cos(raw_deviation))

        mag_for_weight = mag_inner[:, :, 1:]
        mag_max = mag.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        mag_weight = mag_for_weight / mag_max

        weighted_dev_sq = mag_weight * deviation.square()
        return weighted_dev_sq.mean()
