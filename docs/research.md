# Osmium — Research & References

## Core Concept

Variable-rate time-scale modification (TSM) for speech, guided by neural importance mapping. Unlike uniform-rate tools (sox tempo, rubberband), osmium allocates its "time budget" to perceptually-critical segments — protecting consonant transients and semantic content while aggressively compressing silences, breaths, and sustained vowels.

## Key Libraries & Tools

### Time-Scale Modification

- **PyTSMod** — https://github.com/KAIST-MACLab/PyTSMod
  - Python library implementing multiple TSM algorithms: OLA, WSOLA, phase vocoder, phase vocoder with identity phase locking (PVIPL), TSM based on HPSS
  - WSOLA works well up to ~2x but introduces echo/doubling at higher ratios
  - Phase vocoder with identity phase locking preserves formant structure better at 3-4x
  - Useful reference implementation, but we need variable-rate support (PyTSMod is uniform-rate)

- **sox tempo** — http://sox.sourceforge.net/
  - Classic WSOLA-based time-stretch
  - Degrades noticeably above 2x — loss of consonant clarity, phasing artifacts

- **Rubber Band Library** — https://breakfastquake.com/rubberband/
  - High-quality pitch-preserving time-stretch using phase vocoder
  - R3 engine (v3+) is significantly better than R2
  - Still uniform-rate — no awareness of speech content

### Neural Speech Analysis

- **Moshi / Mimi** — https://github.com/kyutai-labs/moshi (Kyutai Labs)
  - Mimi is a neural audio codec derived from EnCodec
  - 24 kHz input, 12.5 Hz token rate (80ms frames), 8-level RVQ
  - **Critical property**: codebook 0 captures semantic content (distilled from WavLM), remaining codebooks capture acoustic detail
  - Semantic/acoustic split enables measuring information density per frame
  - MLX backend available for Apple Silicon inference
  - Rust engine (rustymoshi) for high-performance inference
  - ~200MB for Mimi encoder alone

- **WavLM** — https://github.com/microsoft/unilm/tree/master/wavlm
  - Self-supervised speech representation model
  - Mimi's semantic codebook is distilled from WavLM features
  - Could be used directly for importance scoring, but Mimi is more efficient (already quantized to discrete tokens)

- **Whisper** — https://github.com/openai/whisper
  - OpenAI's ASR model, available in multiple sizes
  - Whisper tiny/base (~75-150MB) could provide phoneme-level alignment via forced alignment
  - MLX ports available (mlx-whisper)

- **wav2vec 2.0 / CTC alignment** — https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
  - CTC-based models produce frame-level phoneme posteriors
  - Can be used for forced alignment to get precise phoneme boundaries
  - Smaller and faster than Whisper for pure alignment tasks

### MLX Ecosystem

- **MLX** — https://github.com/ml-explore/mlx (Apple)
  - NumPy-like ML framework optimized for Apple Silicon
  - Unified memory architecture eliminates CPU-GPU transfer overhead
  - First-class support for transformer inference

- **mlx-audio** — MLX ports of audio models
  - Community ports of Whisper, EnCodec-family models to MLX

## Academic Background

### Time-Scale Modification

- Driedger, J., & Müller, M. (2016). "A Review of Time-Scale Modification of Music Signals." Applied Sciences.
  - Comprehensive survey of OLA, WSOLA, phase vocoder variants
  - Identity phase locking (Laroche & Dolson, 1999) is the gold standard for phase vocoder quality

- Verhelst, W., & Roelands, M. (1993). "An overlap-add technique based on waveform similarity (WSOLA)."
  - Foundational WSOLA paper — cross-correlation based segment matching

### Perceptual Importance in Speech

- Shannon, C. E. (1948). "A Mathematical Theory of Communication."
  - Information-theoretic basis for measuring surprisal/entropy of token sequences

- Stilp, C. E., & Kluender, K. R. (2010). "Cochlea-scaled entropy as a measure of speech intelligibility."
  - Perceptual importance correlates with spectral change rate — rapid spectral transitions (consonants) carry more information than steady-state segments (vowels)

- Janse, E. (2004). "Word perception in fast speech: artificially time-compressed vs. naturally produced fast speech."
  - Natural fast speech preserves consonant durations more than vowel durations
  - Artificial compression that mimics this pattern is more intelligible

### Neural Audio Codecs

- Défossez, A., et al. (2022). "High Fidelity Neural Audio Compression." (EnCodec)
  - Foundation for Mimi — RVQ-based neural codec

- Défossez, A., et al. (2024). "Moshi: a speech-text foundation model for real-time dialogue." (Kyutai Labs)
  - Mimi codec with semantic/acoustic codebook separation
  - Demonstrates that first codebook captures WavLM-like semantic features

## Key Insight

Natural fast speakers don't compress all phonemes equally — they maintain consonant clarity while shortening vowels and pauses. Osmium replicates this by using neural models to identify what's perceptually important, then applying non-uniform time compression that mirrors how humans naturally accelerate speech.
