# Vocos Phase Smoothness Regularization

Design spec for fine-tuning the Vocos ISTFT head with an instantaneous frequency deviation (IFD) loss to reduce phase discontinuity clicks on time-resampled mel.

Date: 2026-03-29

## Problem

Vocos's ISTFT head predicts phase via a single linear projection (512 -> 1026, chunked into magnitude + phase). Phase is predicted per-frame with no explicit inter-frame temporal linking. The ConvNeXt backbone provides temporal context through its receptive field, but the head's projection can still produce incoherent phase between adjacent frames when the input mel has been temporally interpolated for time-stretching.

The prior augmentation-only approach (resample-roundtrip + mel loss) gave marginal click reduction at the cost of voice quality. The 50/50 weight blend with pretrained is the current best, shipped as `--vocos-blended`. The fundamental bottleneck is that mel-domain loss cannot directly supervise phase coherence.

## Approach: Instantaneous Frequency Deviation Loss

Add a regularization loss that directly penalizes unexpected phase jumps between adjacent STFT frames, computed on augmented (resample-roundtrip) forward passes where phase actually breaks down.

### IFD Loss Formulation

After the head's linear projection and `chunk(2, dim=1)`, both magnitude and phase tensors have shape `(B, 513, T)` where 513 = n_fft/2 + 1 frequency bins (DC at k=0 through Nyquist at k=512). The IFD loss operates on bins k=1 through k=511 (excluding DC and Nyquist), using the STFT bin index `k` directly in the expected advance formula.

For predicted phase `phi[k, t]` at frequency bin `k` (1 <= k <= 511), time frame `t`:

```
expected_advance[k] = 2 * pi * k * hop_length / n_fft    (k is the STFT bin index)
actual_advance[k, t] = phi[k, t+1] - phi[k, t]
raw_deviation[k, t] = actual_advance[k, t] - expected_advance[k]
deviation[k, t] = atan2(sin(raw_deviation[k, t]), cos(raw_deviation[k, t]))
```

Magnitude-weighted to focus on audible bins and ignore phase in silence/noise. Normalization is per-sample (max over frequency and time, not across batch):

```
mag_weight[k, t] = mag[k, t] / (mag.amax(dim=(freq, time), keepdim=True) + eps)
L_phase = mean(mag_weight * deviation^2)
```

Key properties:
- Subtracts natural per-bin phase rotation so high-frequency bins aren't unfairly penalized
- Principal-angle wrapping via atan2 avoids discontinuities in the loss surface
- Magnitude weighting ensures phase loss is only active where it matters perceptually
- L2 penalty acts as a gentle regularizer, not a hard constraint -- the model can still predict nonzero residual IF for off-bin harmonics, just at a cost

### Why IFD, Not Zero Phase Gradient

Raw phase difference `||d(phi)/dt||^2` would penalize all phase advance equally, requiring per-bin normalization to avoid high-frequency bins dominating. IFD removes the expected advance analytically. Codex review flagged that real harmonics are off-bin (nonzero residual IF is normal), which is why this is a soft L2 regularizer with a small coefficient rather than a hard constraint.

## Architecture

### Phase Extraction in Training

Rather than subclassing ISTFTHead, the training step extracts phase directly by calling the head's components separately:

```python
# In training_step, after backbone forward:
x_proj = head.out(backbone_output).transpose(1, 2)  # (B, 1026, T)
mag_raw, phase = x_proj.chunk(2, dim=1)              # each (B, 513, T)
mag = torch.exp(mag_raw).clip(max=1e2)

# Reconstruct audio via ISTFT (same as head.forward)
S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
audio = head.istft(S)

# Phase and mag are now available for IFD loss
# No intermediate tensors stashed on the module
```

This avoids storing `_last_phase`/`_last_mag` as instance attributes (which is fragile under gradient checkpointing or DDP). The head's checkpoint keys are unchanged since we call the same `head.out` and `head.istft` — no subclass needed, no new parameters.

### Frozen Backbone

```python
backbone.requires_grad_(False)
```

Only `head.out` (Linear 512 -> 1026) receives gradients. 526,338 trainable parameters (512*1026 + 1026) vs ~13M total. Rationale: the backbone already produces good features with temporal context; the phase prediction problem is in the final projection. Lower regression risk, faster training. If results are limited at 3x/4x, unfreeze last 1-2 ConvNeXt blocks at lower LR as a second pass.

## Training Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total loss | mel_coeff * L_mel + phase_coeff * L_phase | Reconstruction + phase regularizer |
| mel_coeff | 45 | Standard Vocos value |
| phase_coeff | 0.05 | Phase loss operates close to logits (short gradient path); start low to avoid dominating. Log both terms to calibrate. |
| Discriminator | None (pretrain_mel_steps=999999) | Avoids crackle from fresh discriminators |
| Augmentation | resample-roundtrip, fixed 30% of batches | Phase loss on augmented passes, mel loss on all. Fixed rate (not ramped) so phase loss gets consistent signal from step 0. Rate range: Uniform(1.5, 5.0), presmooth sigma=2.0, matching existing augment.py. |
| LR | 1e-5 (pass `--lr 1e-5` explicitly; code default is 2e-5) | Same as V3 (didn't catastrophically regress) |
| Optimizer | AdamW, betas=(0.8, 0.9) | Match upstream Vocos |
| Steps | 5K | Sufficient based on V3 experience |
| Data | LibriTTS train-clean-100 | Already downloaded |
| Trainable params | Head only (backbone frozen) | Low regression risk |
| Gradient clipping | 1.0 | Match existing training |
| Batch size | 16 | Match existing training |

### Loss Application Strategy

- **Non-augmented batches**: mel loss only (phase is already coherent on normal data)
- **Augmented batches**: mel loss + phase IFD loss (target the actual failure mode)
- Log `generator/mel_loss`, `generator/phase_ifd_loss`, `generator/total_loss` separately

### Monitoring

- Log raw phase loss and mel loss values per step
- Log gradient norms for head.out.weight to verify phase loss isn't dominating
- Validation: per-rate mel loss + click detection at 2x/3x/4x
- Early stopping if val/mel_loss_normal regresses beyond 110% of baseline (QualityGateCallback)

## Validation

Reuse existing validation infrastructure with enhancements:

### Per-Rate Metrics (existing)
- mel_loss at 2x, 3x, 4x resample-roundtrip
- click_rate at 2x, 3x, 4x

### Enhanced Click Detector

Add spectral transient detection alongside existing peak-to-median ratio:
- Compute short-time energy in low-frequency band (< 300Hz) and broadband
- Detect frames where both spike simultaneously (characteristic of phase discontinuity clicks, similar to plosive pops)
- This catches clicks that manifest as broadband energy spikes rather than just amplitude peaks

### A/B Sample Generation

Same as previous: generate comparison samples at 2x/3x/4x from Moby Dick clips every 2000 steps.

## Code Changes

1. **`scripts/vocos_finetune/phase_loss.py`** (new)
   - `InstantaneousFrequencyDeviationLoss` module
   - Takes phase tensor (B, N, T) and magnitude tensor (B, N, T)
   - Computes magnitude-weighted IFD, excluding DC and Nyquist bins
   - Returns scalar loss

2. **`scripts/vocos_finetune/train.py`** (modify)
   - New `VocosPhaseRegExp` class (subclass of `VocosExp`)
   - Freeze backbone in `__init__` via `backbone.requires_grad_(False)`
   - Override `training_step`: call `head.out()` and `head.istft()` directly to extract phase/mag and audio; compute phase IFD loss on augmented batches. Single-optimizer design: no `optimizer_idx` branching (unlike parent class which branches on optimizer_idx 0/1 for disc/gen).
   - Override `configure_optimizers`: return single AdamW optimizer for `head.parameters()` only (no discriminator optimizer). Since there's one optimizer, PL calls `training_step` with `optimizer_idx=0` always.
   - New CLI flag `--phase-coeff` (default 0.05)
   - Starting checkpoint: pretrained `charactr/vocos-mel-24khz` (not V3 fine-tuned), since we want to isolate the effect of phase regularization

3. **`scripts/vocos_finetune/click_detector.py`** (enhance)
   - Add `spectral_transient_clicks()` function
   - Broadband + low-frequency energy spike detection
   - Integrate into existing `clicks_per_second()` as additional detection mode

4. **`scripts/vocos_finetune/convert_mlx.py`** (no changes)
   - Head weights have identical keys/shapes to base ISTFTHead
   - Conversion path works unchanged

## Failure Modes and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Phasey/metallic artifacts from oversmoothed phase | Medium | Low coefficient (0.05), magnitude weighting, log and monitor |
| No improvement at 3x/4x (head too limited) | Medium | Unfreeze last 1-2 ConvNeXt blocks as second pass |
| Normal-rate timbre regression | Low | Backbone frozen, QualityGateCallback, mel loss on all batches |
| Phase loss dominates gradients | Low | Start at 0.05, log gradient norms |
| Improvement on roundtrip val but not real inference | Medium | A/B listening tests on real time-stretched audio remain the gate |

## Future Directions (not in this iteration)

- **Phase distillation**: target pretrained model's phase delta instead of zero IFD. Addresses the off-bin harmonics concern more directly.
- **Residual phase head**: add a small temporal conv on top of pretrained logits rather than fine-tuning the linear projection.
- **Post-processing de-click filter**: inference-time transient suppressor as defense-in-depth. Detects broadband energy spikes (similar mechanism to plosive pop filters) and attenuates them.
- **Hybrid vocoder**: use Vocos for <=2x, switch to BigVGAN for >3x where clicks dominate.
- **Partial backbone unfreeze**: if head-only is insufficient, unfreeze last 1-2 ConvNeXt blocks at 10x lower LR.
