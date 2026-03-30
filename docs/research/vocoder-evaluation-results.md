# Vocoder Evaluation Results

A/B listening tests, 2026-03-28.

## Summary

Vocos is the better vocoder for time-resampled mel despite occasional clicks. BigVGAN eliminates clicks but introduces wobble/timbre degradation that's subjectively worse.

## Vocos (current)

- Excellent timbre at all speeds
- Occasional clicks from ISTFT phase discontinuities on resampled mel
- Pre-smooth sigma=2.0 + linear interpolation reduces clicks (committed)
- Phase continuity enforcement (unwrap + smooth) eliminates clicks but introduces phasey artifacts — not usable
- Clicks are mild at 2x, noticeable at 4x

## BigVGAN v2 (112M)

- No clicks (time-domain upsampling has no phase prediction to break)
- Amplitude wobble from mel interpolation artifacts
- Prefers cubic interpolation + presmooth sigma=1.0 (opposite of Vocos)
- Even at best settings, timbre is noticeably worse than Vocos — robotic/distant quality
- The wobble is more distracting than Vocos's clicks
- 5-9x realtime on MPS vs Vocos's ~200x on MLX

## BigVGAN-base (14M)

- Same click-free behavior as full model
- Much worse timbre ("robot slightly far away")
- Not competitive

## Key insight: mel format mismatch

Initial BigVGAN tests sounded terrible because our `extract_mel` (raw np.log, range -10.6 to +4.1) doesn't match BigVGAN's expected mel format (dynamic_range_compression, range -11.5 to +0.5). Using BigVGAN's native `get_mel_spectrogram` fixed this. Any future vocoder integration must use the vocoder's own mel extraction.

## Vocos fine-tuning results (2026-03-29)

Fine-tuned Vocos on LibriTTS train-clean-100 with resample-roundtrip mel augmentation (downsample mel by random rate then upsample back to original length, creating interpolation artifacts the model learns to handle).

### V1: Full training (discriminator + mel loss, 10k steps, 30-50% aug, LR 2e-5)
- Fewer clicks (10-20% reduction at 3x-4x on Moby Dick clips)
- BUT: pronounced crackle, metallic artifacts, robotic/hollow at slower speeds
- Root cause: fresh discriminators (not pretrained) provided harmful adversarial gradients
- Not usable

### V3: Mel-only (no discriminator, 5k steps, 15-25% aug, LR 1e-5)
- Crackle mostly gone
- Clicks reduced at 3x, subtle improvement at 4x
- Still some light metallic ringing at 2x
- Voice quality slightly worse than pretrained — not a clear win

### Weight blending (50% pretrained + 50% V3)
- Best perceptual result — improved timbre quality over both pretrained and pure fine-tuned
- 4x more listenable than baseline
- 3x still noticeably clicky
- Sweet spot for practical use if any fine-tuned weights are used

### Conclusion

Resample-roundtrip augmentation provides marginal click reduction but at a cost to voice quality. The ISTFT phase prediction architecture is the fundamental bottleneck — mel-domain augmentation can't fully teach the model to predict coherent phase for temporally interpolated input. The 50/50 blend is the best practical option but the improvement doesn't clearly justify the complexity.

## Phase smoothness regularization results (2026-03-29)

Added instantaneous frequency deviation (IFD) loss penalizing phase jumps between adjacent STFT frames on augmented mel inputs. Magnitude-weighted, atan2-wrapped, excluding DC/Nyquist. Trained head-only (backbone frozen, 526K params), mel_coeff=45, phase_coeff=0.05, LR=1e-5, 5K steps.

### Head-only (phase_coeff=0.05, 5K steps)
- Phase IFD loss: 0.024 → 0.021 (12% reduction, modest)
- Click rates on Moby Dick clips: ~4% reduction on average, mixed at 3x
- Perceptual: "maybe a tad better but not way better"
- mel_loss_normal didn't regress (0.241 → 0.223)
- Codex review predicted "small improvement possible, not a real fix" for head-only — confirmed

### Assessment
Head-only training doesn't have enough capacity to reshape phase prediction. The linear projection is the wrong bottleneck to target — the backbone features determine what information is available for phase coherence. Next steps: (1) post-processing de-click filter, (2) bump phase_coeff, (3) unfreeze backbone layers.

### Unexplored directions
- **Post-hoc phase smoothing**: smooth the predicted phase AFTER the backbone but before ISTFT, at inference time. Zero training required, but earlier experiments with this ("phase continuity enforcement") produced phasey artifacts.
- **Hybrid vocoder**: use Vocos for <=2x, switch to BigVGAN for >3x where clicks dominate. Trades speed for quality at high rates only.
- **Phase distillation**: target pretrained model's phase delta instead of zero IFD.
