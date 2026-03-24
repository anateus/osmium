# Osmium — Questions & Decision Points

Accumulated during implementation for later review.

---

## 2026-03-16: MLX version pinning

`moshi_mlx==0.3.0` requires `mlx==0.26.5` (downgrade from latest 0.31.1). I proceeded with 0.26.5 to get `rustymimi` working. Once Kyutai updates their package or we extract just the Mimi encoder, we can upgrade MLX.

**Decision made:** Accept MLX downgrade for now.

---

## 2026-03-16: Mimi vs standalone importance analysis

For Phase 2, I'm using Mimi's semantic codebook (codebook 0) from `rustymimi` for importance scoring. An alternative would be using a standalone WavLM or HuBERT model, which would avoid the Moshi dependency entirely. Worth revisiting if the Mimi approach proves insufficient.

**Decision made:** Start with Mimi via `rustymimi`, pivot to standalone model if needed.

---

## 2026-03-16: Phoneme alignment model

For the phoneme aligner, options are:
1. `whisper` (tiny/base) with forced alignment — heavier but well-known
2. A CTC model (wav2vec2/HuBERT) — lighter, faster for pure alignment
3. Skip phoneme alignment in v0.1 and rely on Mimi semantic scoring alone

**Decision made:** Start with Mimi-only importance (option 3), add phoneme alignment as a quality improvement later. This gets us to a working variable-rate system faster.

---

## 2026-03-16: rustymimi API — StreamTokenizer vs Tokenizer

`rustymimi.Tokenizer.encode_step()` has a numpy ABI incompatibility with numpy 2.x (PyO3 binding issue). `rustymimi.StreamTokenizer` works fine with the async encode/get_encoded pattern. Using StreamTokenizer with 1:1 encode→drain pattern.

**Decision made:** Use StreamTokenizer. May revisit if rustymimi updates fix Tokenizer.

---

## 2026-03-16: Importance scoring weights

Current weights: transition=0.35, energy=0.30, multi_cb=0.35. These are initial values — would benefit from perceptual testing to tune. Could expose as CLI flags if needed.

**Open question for Mike:** After listening to neural vs uniform outputs, do the importance weights need adjustment? Would you prefer more or less protection of important segments?

---

## 2026-03-16: Mimi analysis speed

Currently ~3.5x realtime on Apple Silicon. For a 12h audiobook, analysis takes ~3.4h. This is the bottleneck — the TSM phase vocoder itself is ~200x realtime for uniform or ~3.5x for variable-rate (dominated by Mimi analysis time, not TSM).

Options for improvement:
1. Batch multiple chunks to StreamTokenizer (need to test if API supports it)
2. Use the MLX Mimi model directly instead of rustymimi
3. Skip analysis for obvious silence (energy-based pre-filter)
4. Accept the speed and pipeline it — analyze while stretching previous chunks

**Open question:** Is ~3.5x realtime acceptable for batch mode, or should we prioritize speed?

---

## 2026-03-16: Streaming mode doesn't use neural analysis yet

The `--stream` path currently uses uniform-rate only. Adding neural analysis to streaming requires a lookahead buffer approach (analyze N seconds ahead, then stretch). Achievable but adds latency.

**Decision made:** Defer neural streaming to Phase 3. Uniform streaming works now.

---

## 2026-03-16: Mimi analysis speed — RESOLVED

MLX-native Mimi encoder achieves ~500x realtime (0.06s for 30s), vs rustymimi's 3.4x realtime. No longer a bottleneck.

---

## 2026-03-16: PSOLA engine — not better for speech acceleration

ABX testing: PSOLA produces "scrambly" artifacts on unvoiced consonants (t, k, s, f) because pitch detection fails on noise-like segments. Phase vocoder with identity phase locking + transient preservation + noise-band randomization sounds better overall.

PSOLA remains available via `--engine psola` but phase_vocoder is the correct default.

**Decision made:** Keep phase_vocoder as default. PSOLA is experimental.

---

## 2026-03-17: MLX as future direction for PSOLA

Current hybrid engine (voiced_split.py) processes segments independently using CREPE voicing detection, with per-segment rate interpolation. This works but has architectural limits:

- Segments are processed sequentially, context-unaware
- Window sizes are fixed per segment, not optimized across the full signal
- CREPE hop is 10ms; PSOLA mark granularity is limited by pitch period

MLX opens several avenues for improvement:
1. **Batch PSOLA**: Instead of loop-per-mark, compute all synthesis frames as a matrix op (source frames stacked, Hanning windows applied in batch, scattered to output buffer). This would give ~10x throughput on Apple Silicon and enable larger analysis windows.
2. **Larger kernels**: MLX convolutions with large receptive fields (e.g. 3000+ samples = 125ms at 24kHz) could learn pitch-period-scale patterns directly, replacing LP residual + peak-picking GCI detection.
3. **Learned pitch marks**: A small CNN trained to predict GCI positions from raw audio would be more accurate than CREPE + heuristic search, especially at segment boundaries.
4. **Transient-aware voicing**: Current voiced/unvoiced split is binary. MLX could produce continuous voicing scores per sample, enabling smooth blending instead of hard crossfades.

**Near-term plan**: Keep current Praat-style hybrid as the stable engine. Explore batch PSOLA in MLX as a quality/speed improvement once the current engine is well-validated perceptually.

---

## 2026-03-17: Neural vocoder approach (Vocos) — future option

Mel-spectrogram resample + neural vocoder as an alternative to signal-level TSM:
1. Compute mel-spectrogram from input speech
2. Resample mel along time axis (numpy interpolation) — this IS time stretching
3. Feed resampled mel to Vocos (or HiFi-GAN) neural vocoder to generate waveform
4. Variable rate: non-uniform resampling guided by importance map

Advantages: completely sidesteps phase vocoder artifacts and PSOLA pitch mark issues. Mel captures formants, pitch, energy; vocoder handles phase/waveform generation naturally.

Vocos (github.com/gemelo-ai/vocos) works in frequency domain (ISTFT-based), is very fast, and has pre-trained models on HuggingFace. Could potentially port to MLX.

**Decision:** Deferred to future phase. Currently exploring Mimi codec latent-space resynthesis (approach 2) first.

---

## 2026-03-17: Parallel chunked encoding — future optimization

Encode output in parallel chunks then lossless-concat with ffmpeg concat demuxer:
1. Each chunk encodes to a temp file via independent ffmpeg process
2. Write concat manifest: `file 'chunk_001.m4a'\nfile 'chunk_002.m4a'\n...`
3. `ffmpeg -f concat -i manifest.txt -c copy output.m4a` — bitstream copy, no re-encode

For AAC/M4A this works cleanly (independently decodable segments). Would also enable pipelining: stretch chunk N → encode chunk N in background → stretch chunk N+1.

Current encode time is ~10% of total (2.5min of 24.4min for 12h audiobook). Modest standalone gain but enables true pipeline overlap.

**Decision:** Deferred. Mimi analysis (~11min) is the dominant bottleneck.

---

## 2026-03-16: HPSS — introduces discontinuities

Separately stretching harmonic/percussive components and recombining creates timing misalignment artifacts. Worse than single-path phase vocoder.

Available via `--hpss` but not recommended. Potential improvement: better alignment between the two paths before recombination.

**Decision made:** HPSS is opt-in experimental. Phase vocoder is default.

---
