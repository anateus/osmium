# Phoneme-Aware Rate Scheduling

Adds phoneme-class awareness to the importance analysis pipeline so that consonant frames receive minimum importance floors based on their articulatory compressibility. Two tiers: a fast default that classifies frames from CTC emissions, and an opt-in full forced-alignment path for maximum precision.

## Motivation

The current mel importance path (`analyzer/mel_importance.py`) uses spectral flux and energy to infer which frames are "important." This works well for detecting consonant *onsets* (high spectral flux) but poorly for sustained consonants (fricatives, nasals) and consonant *closures* (plosive silence before burst), which have low flux but are perceptually critical.

Decades of speech research (Crystal & House 1988, Gay 1978, Klatt 1979, Janse 2004) establish a clear compressibility hierarchy: pauses > unstressed vowels > stressed vowels > liquids > nasals > fricatives > stop closures > stop bursts. The Klatt duration model formalizes this with per-phoneme minimum durations (D_min) that cannot be compressed further regardless of speaking rate. Our rate schedule should respect these floors.

## Design

### Path C (default): Phoneme-class importance modulation

**New module: `analyzer/phoneme_class.py`**

Uses MMS_FA (torchaudio `MMS_FA` pipeline) to classify each audio frame into a phoneme class, then maps that class to an importance floor.

**MMS_FA properties:**
- Input: 16kHz audio (resampled from 24kHz internally)
- Output: CTC log-probabilities at 50 frames/sec over 29 labels (16kHz / 320 stride)
- Labels: `-` (blank), `a i e n o u t s r m k l d g h y b p w c v j z f ' q x *`
- Model size: ~1.2GB (one-time download, cached)

**Frame classification algorithm:**
1. Run MMS_FA forward pass to get log-probabilities (T_mms × 29). Convert to probabilities via `torch.exp()` (MMS_FA applies `log_softmax` internally — do NOT apply `softmax` again, which is a bug in the existing `aligner.py` that must be fixed).
2. For each frame, check blank probability:
   - If `P(blank) > 0.8` → class = **silence**
3. Otherwise, find the most probable non-blank label and map to class:
   - `{t, d, k, g, b, p, c, '}` → **plosive** (`c` covers /k/ and /tʃ/; `'` is glottal stop)
   - `{s, z, f, v, h, x}` → **fricative** (`x` covers velar/uvular fricatives)
   - `{m, n}` → **nasal**
   - `{l, r, w, y, j}` → **liquid_glide**
   - `{a, e, i, o, u}` → **vowel**
   - `{q, *}` → **plosive** (fallback — conservative, protects unknown segments)

**Importance floor per class** (derived from literature compression hierarchy, section 2.4 of `docs/research/fast-speech-literature.md`):

| Class | Floor | Rationale |
|---|---|---|
| plosive | 0.92 | Stop bursts/closures are near-incompressible |
| fricative | 0.85 | Frication noise needs minimum duration for identity |
| nasal | 0.78 | Less compressible than vowels, more than fricatives |
| liquid_glide | 0.65 | Moderate compressibility |
| vowel | 0.25 | Most compressible speech segment |
| silence | 0.05 | Pauses are the first to go |

**Integration point:** After mel importance and prosodic modulation, as the final step before rate schedule generation in `cli.py::_process_file`:

```
mel importance (94 Hz)
    ↓
prosodic modulation (if enabled)
    ↓
phoneme-class floors (50 Hz, interpolated to 94 Hz)
    ↓
final = max(modulated_importance, phoneme_floor)
    ↓
rate schedule
```

The phoneme floor is applied *after* prosodic modulation, not before. This is critical: prosodic modulation is multiplicative (`scores * (floor + (1-floor) * prosody)`) and would defeat the phoneme floor — e.g., a plosive at 0.92 in an unstressed syllable would drop to `0.92 * 0.5 = 0.46`. By applying the floor last, we guarantee that consonant frames never drop below their class minimum regardless of prosodic context. Mel importance and prosodic modulation can still *raise* frames above the floor.

**Performance:** MMS_FA inference is fast (~50x realtime on CPU for the forward pass). The model is already a torchaudio dependency. Total overhead: ~0.02x realtime added to analysis.

**CLI:** Enabled by default. `--no-phoneme` flag disables it (for A/B comparison and backward compatibility).

### Path B (opt-in): Full forced alignment (`--phoneme-align`)

**New module: `analyzer/phoneme_align.py`**

Uses a transcript (from Whisper or Kyutai STT) plus MMS_FA forced alignment to get exact phoneme boundaries and identities, then assigns importance per-phoneme-segment using the same class-to-floor table.

**Pipeline:**
1. **Transcript generation:**
   - Primary: Kyutai `stt-1b-en_fr` via transformers (if available). Natural fit — uses Mimi tokenization (12.5 Hz), shares ecosystem with existing Mimi analysis path. ~0.5s latency, streaming-capable.
   - Fallback: Whisper (already used in `scripts/eval_wer.py`). `whisper.load_model("base")` for speed.
2. **Tokenization:** Feed transcript through `MMS_FA.get_tokenizer()` to get token sequence.
3. **Forced alignment:** Use `torchaudio.functional.forced_align(emission, tokens)` to get per-token time boundaries.
4. **Phoneme-class lookup:** Map each aligned token to its phoneme class using the same table as Path C.
5. **Importance curve:** Generate a step-function importance curve at phoneme-boundary resolution, with each segment assigned its class floor value. No explicit boundary smoothing needed — the rate schedule's Gaussian smoothing (sigma=15 frames) handles transitions.
6. **Output:** An `ImportanceMap` at mel frame rate (~94 Hz).

**Why better than Path C:** Exact phoneme identity and boundaries vs. frame-level CTC heuristics. The difference matters for:
- Ambiguous frames (nasal-vowel transitions where CTC emission is uncertain)
- Plosive closures (silence before a burst — CTC sees "blank" but forced alignment knows it's part of a /t/)
- Geminate consonants and consonant clusters

**Why not default:** Adds Whisper or Kyutai STT dependency. Analysis time ~2-3x longer than mel-only path. Path C captures ~80% of the benefit with no extra dependencies beyond torchaudio.

**Language limitation:** Kyutai STT supports English and French only. For other languages, fall back to Whisper (99 languages). No automatic language detection — Whisper's `detect_language()` can be used if needed, but for v1 we default to Whisper unless the user explicitly opts into Kyutai STT.

**CLI:** `--phoneme-align` flag. Cannot be combined with `--uniform`. When active, replaces the Path C phoneme-class modulation (no need for both — Path B subsumes Path C). Compatible with `--mimi` (Mimi importance is computed independently; phoneme floor from Path B replaces both Path C and the old `aligner.py` integration in the Mimi path).

### Comparison tooling

**New script: `scripts/compare_phoneme.py`**

Runs osmium on a given input file at multiple speeds (2x, 3x, 4x, 5x) with three configurations:
1. `--no-phoneme` — mel-only (baseline, current behavior)
2. Default — mel + phoneme-class floors (Path C)
3. `--phoneme-align` — full forced alignment (Path B, if deps available)

For each output, runs Whisper WER/CER evaluation (reusing `scripts/eval_wer.py` infrastructure). Writes a markdown report table with WER/CER at each speed × configuration.

Usage: `uv run python scripts/compare_phoneme.py input.wav --whisper-model small`

## Files changed

| File | Change |
|---|---|
| `analyzer/phoneme_class.py` | **New.** MMS_FA emission classifier → phoneme class → importance floor. Returns `ImportanceMap` for pipeline compatibility. |
| `analyzer/phoneme_align.py` | **New.** Transcript + forced alignment → per-phoneme importance. Returns `ImportanceMap`. |
| `analyzer/aligner.py` | **Remove.** Superseded by `phoneme_class.py` (Path C) and `phoneme_align.py` (Path B). The existing `compute_phoneme_importance` function has a double-softmax bug and an inferior classification approach. The Mimi importance path (`importance.py`) should be updated to use `phoneme_class.py` instead, removing the `use_phoneme_alignment` parameter. |
| `analyzer/importance.py` | Remove `use_phoneme_alignment` parameter and the import of `aligner.py`. The phoneme floor is now applied externally in `cli.py`, not inside Mimi importance. |
| `cli.py` | Add `--no-phoneme` and `--phoneme-align` flags. Wire phoneme floor application as final step after mel/mimi importance + prosodic modulation, before rate schedule. Wire phoneme_align as alternative when `--phoneme-align` is set. |
| `scripts/compare_phoneme.py` | **New.** A/B/C comparison script |
| `ARCHITECTURE.md` | Update Stage 1 to document phoneme-class analysis. Add phoneme_class.py and phoneme_align.py to module map. |

**Note:** `parallel.py` does NOT need changes — it receives pre-computed `rate_curve` and `rate_times` from `_process_file`. All phoneme analysis happens before chunking.

## What this does NOT do

- Does not implement the full Klatt D_min duration model (compressible-portion math). The floor-based approach is simpler and achieves the same asymptotic protection. The Klatt model could be a future refinement if floor values prove too coarse.
- Does not use BonnTempo or other corpora for empirical parameter estimation. The floor values are literature-derived approximations. Corpus-driven tuning is a separate future effort.
- Does not modify the Vocos mel-domain stretching or adaptive smoothing — those remain unchanged. The improvement is entirely in the importance/rate-schedule input signal.
- The `--analyze-only` export should include phoneme-floor-adjusted importance (the final signal that feeds the rate schedule), so the JSON output reflects what the rate schedule actually sees.
