# Vocos Fine-Tuning for Resampled Mel Phase Coherence

Design spec for fine-tuning the Vocos vocoder to eliminate phase discontinuity clicks when synthesizing from time-resampled mel spectrograms.

Date: 2026-03-28

## Problem

Vocos (ConvNeXt backbone + ISTFT head) produces clicks on resampled mel input. The ISTFT head predicts magnitude and phase independently per frame. When adjacent mel frames have been interpolated during time-scale modification, the predicted phases don't align, causing destructive interference in overlap-add. Pre-smoothing (sigma=2.0) and linear interpolation reduce but don't eliminate the artifacts. Clicks are mild at 2x, noticeable at 3x+.

BigVGAN eliminates clicks but introduces worse timbre degradation. Vocos is the better vocoder overall — the fix should target its specific weakness.

## Approach: Resample-Roundtrip Augmentation

Fine-tune the pretrained Vocos on a mix of normal and augmented mel-waveform pairs. The augmentation simulates the interpolation artifacts of resampling without changing the output length, so the original waveform remains a valid target.

For each augmented sample:
1. Extract mel from audio segment (T frames)
2. Downsample mel by random rate R ~ Uniform(1.5, 5.0) to T/R frames using linear interpolation + Gaussian pre-smooth (sigma=2.0), matching the Osmium pipeline
3. Upsample back to T frames with the same interpolation
4. The roundtrip mel has interpolation artifacts (temporal smoothing, spectral bleeding) but matches the original waveform length

The model learns to predict coherent phase from mel that exhibits the characteristic patterns of temporal interpolation.

### Alternatives Considered

**Phase smoothness regularization**: Add a loss penalizing phase jumps between adjacent frames. Simpler but doesn't expose the model to actual resampling patterns. Risk of oversmoothing phase on normal input, muffling audio.

**Self-supervised resampled training**: Actually resample mel at various rates and train with mel reconstruction + adversarial loss only (no waveform target). Most directly addresses the real use case but harder to stabilize — two different loss regimes, variable-length outputs, no waveform grounding for the resampled case.

## Decisions

- **Hardware**: Apple Silicon (MPS backend). PyTorch training, MLX conversion for inference.
- **Data**: LibriTTS train-clean-100 (~100 hours, ~6GB). Same distribution as original Vocos pretraining.
- **Framework**: PyTorch Lightning 1.8.6 (pinned — the `vocos` package uses the PL 1.x multi-optimizer `training_step(batch, batch_idx, optimizer_idx)` API removed in PL 2.0). Custom `VocosFineTuneExp` subclass of `VocosExp`. Convert best checkpoint to MLX weights for the existing `VocosMLX` inference path.
- **Evaluation**: Automated metrics + generated A/B audio samples for perceptual check-in.

## Data Pipeline

### Dataset

The dataset returns raw audio only, based on the upstream `VocosDataset` pattern. The resample-roundtrip augmentation happens inside the training step, between the feature extractor and the backbone, to guarantee mel format consistency with the model's own feature extractor.

For each training sample:
1. Load audio file from LibriTTS filelist
2. Crop random 1-second segment (24000 samples at 24kHz)
3. Normalize gain using pure-torch amplitude scaling (`audio * 10**(gain_db/20)` with random gain -1 to -6 dB) — avoids dependency on `torchaudio.sox_effects` which may not be available on macOS

### Augmentation (in training step, not dataset)

The custom `VocosFineTuneExp` subclass overrides `training_step` to intercept mel after the feature extractor:

1. `features = self.feature_extractor(audio_input)` — standard mel extraction
2. With probability `aug_ratio`: apply resample-roundtrip on the mel tensor (in PyTorch, on device)
3. `x = self.backbone(features)` then `audio_hat = self.head(x)` — standard decode
4. Loss computed against original `audio_input`

The augmentation ratio ramps: 30% for steps 0-2000, linearly increasing to 50% by step 4000, held at 50% thereafter. Implemented via a Lightning `on_train_batch_start` callback that updates `self.aug_ratio` based on `self.global_step`.

### Data Acquisition

A download script fetches LibriTTS train-clean-100 from OpenSLR and generates train/val filelists. The val set is 200 utterances held out for deterministic evaluation.

### Validation Set

200 held-out utterances, each augmented at fixed rates (2x, 3x, 4x) for deterministic tracking across checkpoints.

## Training Configuration

### Model

Load pretrained `charactr/vocos-mel-24khz` (ConvNeXt backbone, 8 layers, dim=512). The feature extractor is the non-learnable `MelSpectrogramFeatures` — used as-is for mel extraction. Backbone and ISTFT head weights are unfrozen for fine-tuning. Discriminators (MultiPeriodDiscriminator + MultiResolutionDiscriminator) initialized fresh — they weren't included in the pretrained release.

### Schedule

Uses the upstream `VocosExp` schedule infrastructure: `transformers.get_cosine_schedule_with_warmup` with `num_warmup_steps=1000`. The discriminator is disabled for the first 1000 steps via `pretrain_mel_steps=1000` (existing upstream parameter). After warmup, LR cosine-decays over the remaining 9000 steps.

| Phase | Steps | Learning Rate | Discriminator | Purpose |
|-------|-------|---------------|---------------|---------|
| Mel-only warmup | 0-1000 | 0 → 2e-5 | Off (pretrain_mel_steps) | Backbone adapts to augmented mel without adversarial instability |
| GAN + cosine decay | 1000-10000 | 2e-5 → 0 cosine | On | Full loss: mel recon (coeff=45) + adversarial + feature matching |

### Hyperparameters

- Batch size: 16
- Segment length: 24000 samples (1 second)
- Optimizer: AdamW, betas=(0.8, 0.9)
- Initial learning rate: 2e-5
- Mel loss coefficient: 45 (same as original)
- MRD loss coefficient: 1.0
- Total steps: 10000 effective batches (`trainer.max_steps=20000` — PL 1.x with two optimizers halves per optimizer in `configure_optimizers`)
- Warmup steps: 1000
- Pretrain mel steps: 1000
- Estimated wall-clock: 3-5 hours on Apple Silicon MPS

### Checkpointing

- Save every 1000 steps
- Keep best 3 checkpoints by composite metric: `0.5 * mel_loss_augmented + 0.5 * mel_loss_normal` (prevents trading normal quality for augmented improvement). Hard gate: discard checkpoints where `mel_loss_normal > baseline_mel_loss + 0.05`.
- Full state (model, optimizers, schedulers, step counter, RNG) for pause/resume

### Pause and Resume

PyTorch Lightning checkpoints include full training state. The runner script accepts `--resume` and passes `ckpt_path="last"` to `trainer.fit()`. The script detects existing data and checkpoints, skipping completed stages on re-run.

## Evaluation

### Automated Metrics (every 1000 steps)

- **Val mel loss (normal)**: reconstruction quality on unmodified mel — must not regress
- **Val mel loss (augmented)**: reconstruction quality on roundtrip-augmented mel — primary optimization target
- **Val mel loss per rate** (2x, 3x, 4x): breakdown by compression severity

### Click Detector

Custom metric computed on validation outputs:
- Short-term energy envelope (48-sample window and hop at 24kHz = 2ms, no overlap)
- Click = energy spike > 3x local median (50ms window)
- Reported as clicks per second at each rate: `click_rate_2x`, `click_rate_3x`, `click_rate_4x`

### A/B Sample Generation (every 2000 steps)

5 held-out utterances processed at 2x, 3x, 4x speeds through Osmium's actual pipeline:
- **Baseline**: current pretrained Vocos (for comparison)
- **Fine-tuned**: checkpoint at current step

Saved to `training/eval_samples/step_{N}/`:
- `{utterance}_{speed}x_baseline.wav`
- `{utterance}_{speed}x_finetuned.wav`
- `README.txt` with mel loss and click count per sample

### Check-in Guide

When reviewing results, look at `training/eval_samples/`. Listen to the 4x samples first — clicks are worst there and improvement will be most obvious. The `README.txt` files summarize metrics per sample.

## MLX Weight Conversion

After training completes:
1. Load best checkpoint (by composite metric)
2. Convert PyTorch state dict → MLX weights using the existing `_convert_weights` pattern from `vocos_mlx.py`
3. Save to `training/models/vocos-mel-24khz-finetuned/weights.npz`
4. Generate final comparison: 10 utterances x 3 speeds x {baseline, finetuned} → `training/final_comparison/`

## Integration (post-training, requires user decision)

`vocos_mlx.py` `_load_model()` gains support for a `--vocos-checkpoint` CLI flag to load fine-tuned weights. The existing pipeline (mel extraction, adaptive smoothing, HF de-emphasis) is unchanged.

## Autonomous Execution

Single entry point: `scripts/vocos_finetune.sh [--resume]`

Stages executed in order (each skipped if already complete):
1. Download LibriTTS train-clean-100 (~15 min)
2. Generate train/val filelists
3. Install training dependencies (`pytorch-lightning==1.8.6`, `transformers`)
4. Run training (10k steps, ~3 hours)
5. Convert best checkpoint to MLX weights
6. Generate final A/B comparison samples
7. Print summary (best val loss, click rates, sample paths)

All output in `training/` (gitignored). TensorBoard logs in `training/logs/`. Plain-text summaries alongside.

## File Structure

```
scripts/
  vocos_finetune.sh          entry point
  vocos_finetune/
    download_data.py          LibriTTS download + filelist generation
    dataset.py                audio dataset (raw audio only, augmentation in training step)
    train.py                  VocosFineTuneExp subclass + training loop (Lightning 1.8.6)
    evaluate.py               click detector + A/B sample generation
    convert_mlx.py            best checkpoint → MLX weights

training/                     (gitignored)
  data/
    LibriTTS/train-clean-100/
    filelists/
      train.txt
      val.txt
  checkpoints/
    step_1000.ckpt
    ...
    best.ckpt
    last.ckpt
  logs/                       TensorBoard
  eval_samples/
    step_2000/
    step_4000/
    ...
  final_comparison/
  models/
    vocos-mel-24khz-finetuned/
      weights.npz
```
