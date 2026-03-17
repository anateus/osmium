# Osmium — Design Document

## Overview

Osmium is a CLI tool for high-quality speech acceleration, targeting 2x–4x speeds on audiobook content. It uses neural importance mapping to apply variable-rate time-scale modification — protecting perceptually-critical speech segments while aggressively compressing redundant content.

## Architecture

Two-stage pipeline:

### Stage 1: Analysis (Neural)

**Mimi encoder (MLX)** — Encodes audio into 8-level RVQ tokens at 12.5 Hz. Codebook 0 captures semantic content. Per-frame surprisal scoring identifies high-information segments.

**Phoneme aligner (CTC model)** — Produces phoneme-level timestamps at ~10ms resolution. Phonemes classified into protection tiers:
- Tier 1 (protect): plosives, affricates, fricative onsets
- Tier 2 (moderate): sustained fricatives, nasals, liquids
- Tier 3 (compress): sustained vowels, silence, breath

**Importance map** — Combined output at configurable resolution (10ms–80ms, default 20ms). Per-frame score 0.0–1.0, computed as max(semantic_score, phoneme_tier_score). Smoothed to prevent jarring rate transitions.

### Stage 2: Time-Scale Modification

**Rate schedule** — Constrained optimization: given target speed and importance map, compute per-frame speed factor where high-importance frames get gentle acceleration and low-importance frames get aggressive acceleration. Rate transitions smoothed (max 0.5x change per 50ms).

**Phase vocoder with identity phase locking** — STFT with variable hop size per frame. Analysis window configurable (default 2048 samples at 24 kHz). Identity phase locking preserves formant structure at high ratios.

## CLI Interface

```
osmium input.mp3 -s 3.0 -o output.mp3
osmium input.m4a -s 2.5 --stream | ffplay -nodisp -
osmium input.mp3 -s 3.5 --resolution 10ms -o output.m4a
osmium input.mp3 --analyze-only -o importance.json
```

Key flags:
- `-s, --speed` — target speed factor (required)
- `-o, --output` — output file (format from extension)
- `--stream` — streaming mode, raw PCM to stdout
- `--resolution` — importance map resolution (default 20ms)
- `--window` — STFT window size (default 2048)
- `--analyze-only` — emit importance map, skip TSM
- `--device` — mlx (default), cuda, cpu
- `--no-model` — uniform-rate fallback, skip neural analysis

## Audio I/O

Decoding and encoding via ffmpeg subprocess. Supports mp3, m4a, wav, flac input/output. Internal processing at 24 kHz mono (Mimi's native rate), resampled as needed.

## Project Structure

```
osmium/
├── pyproject.toml
├── src/osmium/
│   ├── cli.py
│   ├── analyzer/
│   │   ├── mimi.py
│   │   ├── aligner.py
│   │   └── importance.py
│   ├── tsm/
│   │   ├── phase_vocoder.py
│   │   ├── rate_schedule.py
│   │   └── stream.py
│   ├── io/
│   │   ├── decode.py
│   │   └── encode.py
│   └── models/
│       └── download.py
├── samples/
│   ├── full/
│   └── clips/
├── tests/
└── docs/
```

## Dependencies

Python 3.11+, mlx, torch (optional), numpy, soundfile, click. ffmpeg as system dependency. Managed with uv.

## Implementation Phases

1. **MVP** — CLI skeleton, file I/O, uniform-rate phase vocoder
2. **Neural importance** — Mimi encoder, phoneme aligner, variable-rate TSM
3. **Streaming** — Chunked processing, pipe support, real-time playback
4. **Future** — Rust TSM engine, CUDA backend, chapter awareness, benchmarking
