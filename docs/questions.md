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
