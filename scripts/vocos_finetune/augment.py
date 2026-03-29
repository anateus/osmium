import torch
import torch.nn.functional as F


def resample_roundtrip(mel: torch.Tensor, rate: float, presmooth_sigma: float = 0.0) -> torch.Tensor:
    B, C, T = mel.shape
    if presmooth_sigma > 0:
        mel = _gaussian_smooth_1d(mel, presmooth_sigma)
    compressed_T = max(1, round(T / rate))
    compressed = F.interpolate(mel, size=compressed_T, mode="linear", align_corners=True)
    restored = F.interpolate(compressed, size=T, mode="linear", align_corners=True)
    return restored


def random_resample_roundtrip(mel: torch.Tensor, min_rate: float = 1.5, max_rate: float = 5.0, presmooth_sigma: float = 2.0) -> torch.Tensor:
    rate = torch.empty(1).uniform_(min_rate, max_rate).item()
    return resample_roundtrip(mel, rate=rate, presmooth_sigma=presmooth_sigma)


def _gaussian_smooth_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    B, C, T = x.shape
    x_flat = x.reshape(B * C, 1, T)
    pad = kernel_size // 2
    x_padded = F.pad(x_flat, (pad, pad), mode="reflect")
    kernel_3d = kernel.reshape(1, 1, -1)
    smoothed = F.conv1d(x_padded, kernel_3d)
    return smoothed.reshape(B, C, T)
