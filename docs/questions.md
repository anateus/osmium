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
