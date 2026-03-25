# Adaptive Smoothing & Voice Cleanup

Two features that address quality issues at high speed factors (3.8x+): reducing peakiness/choppiness in the output, and cleaning up noisy source audio before processing.

## Problem

### Peakiness (clicks + choppy rhythm)

At 3.8x the rate schedule swings between ~1x (important consonants) and ~8-10x (silences, sustained vowels). This produces two audible artifacts:

1. **Transient spikes/clicks**: When many input mel frames are compressed into few output frames via cubic interpolation, the Vocos vocoder sees abrupt spectral transitions that produce pops at the waveform level. The current fixed mel smoothing (`sigma=0.7`) is insufficient for the highest compression ratios.

2. **Choppy rhythm**: The wide rate swing (1x to 10x) makes the listener perceive a lurching alternation between slow and fast sections, rather than a smooth accelerated flow.

### Noisy sources

Audiobook recordings vary in quality. Common issues are constant room noise / mic hiss (the primary concern) and compression artifacts from low-bitrate source files. These degrade both the importance analysis (noise gets scored as "important") and the Vocos output (the vocoder faithfully reproduces the noise, sometimes amplified).

## Design

### Adaptive mel smoothing + rate contrast compression

Two complementary changes to the existing pipeline:

**Adaptive mel sigma** (`vocos_mlx.py`, `vocos_engine.py`):

Replace the fixed `smoothing_sigma` applied to the resampled mel spectrogram with a per-frame sigma that scales with the local compression ratio.

- Compute the local compression ratio from the source index mapping (how many input frames map to each output frame). Where the rate is high (e.g. 8x), many input frames collapse into one output frame, so more smoothing is needed. Where the rate is near 1x, minimal smoothing preserves consonant attack.
- Sigma range: `sigma_min=0.3` (near 1x rate) to `sigma_max=2.5` (near max rate), interpolated linearly with the local compression ratio.
- **Algorithm**: Approximate variable-sigma smoothing using 4 sigma buckets evenly spaced between `sigma_min` and `sigma_max` (e.g., 0.3, 1.0, 1.7, 2.5 for defaults). Partition output frames into regions by their nearest sigma bucket, apply `gaussian_filter1d` at that sigma to each region, and blend at region boundaries using a short linear crossfade (3-5 frames). This avoids per-frame convolution while achieving adaptive behavior. Extract this as a shared helper in `tsm/smooth.py`: `adaptive_smooth_mel(mel, compression_ratios, sigma_min, sigma_max)` used by both `vocos_mlx.py` and `vocos_engine.py`.

**Rate contrast compression** (`rate_schedule.py`):

Replace the linear importance-to-rate mapping with a compressive curve. Insert `inv_importance = inv_importance ** gamma` on a new line between line 32 (`inv_importance = 1.0 - imp`) and line 33 (`raw_rates = min_rate + ...`):

```
current:  rate = min_rate + inv_importance * (max_rate - min_rate)
proposed: rate = min_rate + inv_importance^gamma * (max_rate - min_rate)
```

With `gamma=1.5` (configurable), mid-range `inv_importance` values produce lower rates than linear. For example, with `inv_importance=0.5` and `max_rate=10`: linear gives rate 5.5x, but `0.5^1.5 = 0.354` gives rate 4.2x. The effect: content that is "somewhat unimportant" (unstressed vowels, brief pauses) gets less aggressive compression, reducing the perceived lurching between fast and slow sections. The extremes (importance=0 silence → rate 10x, importance=1 consonants → rate 1x) are unchanged.

Note: `gamma < 1` would bow the curve the *other* direction (more aggressive compression of mid-range content), which is the opposite of the goal. `gamma=1.0` is linear (current behavior, serves as the off switch).

Expose as `--rate-gamma` CLI option, default 1.5.

**Consonant sensitivity boost** (`mel_importance.py`):

Bump `hf_boost` from 2.0 to 2.5 and shift the flux/energy weights from 0.6/0.4 to 0.65/0.35. This gives slightly more weight to high-frequency spectral changes (consonant onsets) vs. raw energy. The existing Whisper WER harness can validate this doesn't regress intelligibility. Note: update the eval harness baseline configs to reflect the new defaults so A/B comparisons remain meaningful.

### Three-tier voice cleanup

A new preprocessing stage that runs after decode and before importance analysis. Three levels of denoising, selectable via the `--denoise` CLI flag:

```
audio file → decode → [denoise] → importance → rate schedule → stretch → encode
```

Denoise runs in `cli.py` on the full decoded audio *before* `process_chunked` is called. For files over ~30 minutes, the denoise step itself should be chunked internally (e.g., 5-minute segments with 0.5s overlap and crossfade) to bound memory usage. Each denoise module handles its own internal chunking.

**Tier 1: Spectral gating** (`--denoise` or `--denoise gate`)

DSP-only, no new dependencies. Uses numpy/scipy.

1. Compute STFT of the input audio (n_fft=2048, hop=512).
2. Estimate noise profile from the quietest 10-15% of frames (by RMS energy). Average their magnitude spectra to get the noise floor per frequency bin.
3. For each frame, compute per-bin SNR. Attenuate bins where the signal is within a threshold (default 6dB) of the noise floor using a soft gain curve (smooth transition to avoid musical noise).
4. Reconstruct via ISTFT with overlap-add.

New file: `analyzer/denoise.py` with a `spectral_gate(samples, sr, ...)` function.

Expected performance: >100x realtime. Handles constant hiss well. Limited effect on compression artifacts.

**Tier 2: DeepFilterNet** (`--denoise deep`)

Lightweight neural speech enhancement. ~5MB model, runs realtime on CPU.

- New optional dependency group in `pyproject.toml`: `denoise = ["deepfilternet>=0.5,<1.0"]`
- Install: `uv pip install -e '.[denoise]'`
- DeepFilterNet operates at 48kHz internally. Pipeline: upsample 24k→48k via `scipy.signal.resample_poly(samples, 2, 1)`, enhance, downsample 48k→24k via `resample_poly(enhanced, 1, 2)`.
- API: `from df import enhance, init_df; model, state, _ = init_df(); clean = enhance(model, state, audio)` — input is a numpy float32 array of shape `(samples,)` at 48kHz.
- New file: `analyzer/denoise_deep.py`
- Note: pulls in torch as a transitive dependency (~2GB). Document this in help text and README.

Expected performance: ~realtime on CPU. Good on hiss + mild compression artifacts.

**Tier 3: Demucs/HTDemucs** (`--denoise demucs`)

Full neural source separation. Extracts the vocal stem, discarding everything else.

- New optional dependency group: `demucs = ["demucs>=4.0"]`
- Install: `uv pip install -e '.[demucs]'`
- Model: `htdemucs` (~1GB download on first use). Show a progress bar for model download. Uses `demucs.api.Separator(model="htdemucs")` → `separator.separate_tensor(audio)` to extract the `vocals` stem.
- New file: `analyzer/denoise_demucs.py`

Expected performance: ~5-10x realtime on CPU, faster on GPU. Handles everything including music beds and heavy compression artifacts.

### CLI interface

```
--denoise LEVEL      Voice cleanup before processing (omit for no denoising)
                     gate   = spectral gating (DSP, no extra deps)
                     deep   = DeepFilterNet neural denoising (requires: uv pip install -e '.[denoise]')
                     demucs = Demucs source separation (requires: uv pip install -e '.[demucs]')
--rate-gamma FLOAT   Rate contrast compression exponent (default 1.5, 1.0 = linear/off)
```

Click implementation: `@click.option("--denoise", type=click.Choice(["gate", "deep", "demucs"]), default=None)`. No optional-value magic — users must specify the level explicitly. This is simpler and avoids Click footguns.

The existing `--smoothing` flag continues to control mel smoothing. When variable-rate stretching is active (non-uniform mode), `--smoothing` sets the `sigma_max` for adaptive smoothing, with `sigma_min` fixed at `min(0.3, smoothing)`. When uniform mode is active or `--rate-gamma 1.0` is set, `--smoothing` behaves as before (fixed sigma applied uniformly). This is a deliberate change — the old fixed-sigma behavior at variable rates was the source of the peakiness. Users can recover exact old behavior with `--rate-gamma 1.0`.

`--denoise` combined with `--stream` raises a `UsageError` (streaming mode bypasses the batch pipeline; denoising is not supported in streaming mode).

### Files changed

All paths relative to `src/osmium/` unless otherwise noted.

| File | Change |
|------|--------|
| `cli.py` | Add `--denoise` and `--rate-gamma` options, wire denoise into pipeline before importance, error on `--denoise` + `--stream` |
| `tsm/rate_schedule.py` | Add `gamma` parameter to `importance_to_rate_schedule`, apply compressive curve before iterative loop |
| `tsm/vocos_mlx.py` | Adaptive mel smoothing in `vocos_mlx_variable_rate` using shared helper |
| `tsm/vocos_engine.py` | Same adaptive smoothing for PyTorch fallback path using shared helper; clean up pre-existing dead code (duplicate `target_T` computation and unused linear interpolation loop) |
| `tsm/smooth.py` | **New** — shared `adaptive_smooth_mel()` helper used by both vocos paths |
| `analyzer/mel_importance.py` | Bump `hf_boost` to 2.5, flux weight to 0.65 |
| `analyzer/denoise.py` | **New** — spectral gating implementation with internal chunking for large files |
| `analyzer/denoise_deep.py` | **New** — DeepFilterNet wrapper with 24k↔48k resampling (numpy→torch conversion inside wrapper) |
| `analyzer/denoise_demucs.py` | **New** — Demucs wrapper |
| `pyproject.toml` (project root) | Add `denoise` and `demucs` optional dependency groups |
| `scripts/eval_wer.py` (project root) | Update default `hf_boost` to 2.5, `flux_w` to 0.65, `energy_w` to 0.35; add `rate_gamma` config support |

Note: `parallel.py` does not change — denoise runs in `cli.py` before `process_chunked` is called.

### Testing

- Run the existing Whisper WER harness at 3.8x with and without the adaptive smoothing changes. Pass criterion: WER delta < 2 percentage points vs. current output.
- Verify `--rate-gamma 1.0` produces output identical to the pre-change pipeline (backward compatibility regression test).
- Generate sample clips from the Moby Dick source with each denoise tier for ABX comparison.
- Measure processing time overhead for each denoise tier on the full Moby Dick file (~5 hours).
- Test with prosody on/off to verify gamma doesn't interact poorly with the prosodic modulation floor.

### Risks

- **Adaptive sigma bucketing fidelity**: 4 buckets may produce audible transitions at bucket boundaries. If so, increase to 6-8 buckets. The crossfade at boundaries should prevent hard transitions but needs listening tests.
- **DeepFilterNet dependency weight**: Pulls in torch (~2GB). Already present if Mimi is installed, but adds significant size for users who only want lightweight denoising. Document this clearly in `--help` and README.
- **Demucs model download**: First run downloads ~1GB. Show a Rich progress bar during download and warn the user.
- **Rate gamma interaction with prosody**: The prosodic modulation floor (0.5) may need re-tuning with gamma=1.5. Test with prosody on/off.
- **DeepFilterNet API stability**: Pin to `>=0.5,<1.0` and wrap calls in a try/except that gives a clear error message if the API has changed.
