# Vocoder Comparison for Time-Resampled Mel Synthesis

Research compiled for evaluating alternatives to Vocos for click-free synthesis.

Last updated: 2026-03-28

---

## Problem

Vocos (ConvNeXt backbone + ISTFT head) produces clicks when synthesizing from time-resampled mel spectrograms. The ISTFT head predicts magnitude and phase independently per frame — when adjacent mel frames have been interpolated, the predicted phases don't align, causing destructive interference in overlap-add. This is fundamental to the ISTFT architecture, not a parameter tuning issue.

Pre-smoothing the mel (sigma=2.0) and using linear interpolation (instead of cubic) reduce but don't eliminate the artifacts.

## Vocoder Comparison

| Vocoder | Approach | Phase Clicks? | CPU RTF | GPU RTF | MLX Port | Mel Format Match |
|---|---|---|---|---|---|---|
| **Vocos** | ConvNeXt + ISTFT | Yes | 170x | 6697x | Yes (ours) | Native |
| **BigVGAN-full** (112M) | Time-domain upsample + Snake | No | 0.4x | 99x | Early ([yrom](https://github.com/yrom/mlx-bigvgan)) | 24kHz/100band ✓ |
| **BigVGAN-base** (14M) | Same, smaller | No | ~2-3x | ~70x | Same port | 24kHz/100band ✓ |
| **HiFi-GAN** | Time-domain upsample + LeakyReLU | No | 14x | 496x | No | Various |
| **iSTFTNet** | Hybrid upsample + ISTFT | Yes (same issue) | 14x | 1046x | No | — |
| **UnivNet** | Location-variable conv | No | ~20x | — | No | Various |

## Why BigVGAN

- Never predicts phase — waveform is generated directly through transposed convolutions
- Snake activation provides periodic inductive bias for harmonic structure
- Anti-aliased convolutions suppress artifacts from unusual input patterns
- Proven OOD robustness: trained on LibriTTS, generalizes to unseen languages, singing, music
- 24kHz/100band model uses identical mel format to our Vocos setup — drop-in replacement

## Integration Strategy

1. Quick validation: BigVGAN-base (14M) via PyTorch MPS to confirm click-free output
2. If validated: port/improve MLX implementation for BigVGAN-base (est. 10-30x realtime)
3. If quality requires full model: evaluate MLX performance (est. 3-10x realtime)
4. Dual vocoder support: keep Vocos as fast path, BigVGAN as quality path

## Alternative: Phase Continuity Enforcement on Vocos

Instead of switching vocoders, smooth the predicted phase along time after the ConvNeXt backbone but before ISTFT. Unwrap phase per frequency bin, apply Gaussian smoothing, re-wrap. This enforces temporal coherence without changing magnitude. Under evaluation — if effective, avoids the speed regression of BigVGAN.

## References

- [BigVGAN paper (ICLR 2023)](https://ar5iv.labs.arxiv.org/html/2206.04658)
- [BigVGAN v2 24kHz model](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
- [MLX BigVGAN port](https://github.com/yrom/mlx-bigvgan)
- [Vocos paper](https://arxiv.org/html/2306.00814v3)
