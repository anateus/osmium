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

## Next step: fine-tune Vocos on resampled mel

The ISTFT architecture is right for speech quality. The clicks come from the phase predictor seeing out-of-distribution temporal patterns. Fine-tuning on (resampled-mel, original-waveform) pairs should teach it to predict coherent phase for stretched spectrograms.

Training data: take clean speech (LibriTTS or our own samples), compute mel, resample at various rates (1.5x-5x) with the same pipeline we use (linear interp, pre-smooth, adaptive smooth), pair with the original waveform segment. The loss is the same as Vocos's original training (multi-resolution STFT loss + mel loss + adversarial).
