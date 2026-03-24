# Architecture

Osmium has three stages: analyze importance, compute a rate schedule, and stretch the audio. Each stage was shaped by specific failures of the alternatives.

## The problem with uniform stretching

Tools like sox and Rubber Band apply the same rate to every part of the signal. At 2x that's fine. At 3x+ consonants lose their attack, sibilants smear, and intelligibility drops off a cliff. This matches what Janse (2004) found: natural fast speakers don't compress evenly. They keep consonant durations roughly intact and eat the time out of vowels and pauses. Osmium tries to do the same thing.

## Pipeline overview

```
audio file
  │
  ├─ decode (ffmpeg → 24kHz mono float32)
  │
  ├─ Stage 1: importance analysis
  │    mel spectrogram → spectral flux + energy → importance[0..1] per frame
  │    (optional: Mimi neural codec, prosodic envelope)
  │
  ├─ Stage 2: rate schedule
  │    importance → variable rate curve → iterative normalization to hit target speed
  │
  ├─ Stage 3: time-scale modification
  │    resample mel along time axis → Vocos neural vocoder → waveform
  │
  └─ encode (ffmpeg → mp3/m4a/wav/flac)
```

Processing runs at 24kHz mono internally because that's what both Mimi and Vocos expect.

## Stage 1: importance analysis

### Mel importance (default)

`analyzer/mel_importance.py`

The mel spectrogram is already computed for the Vocos vocoder, so importance comes nearly free. Two signals get combined:

- **Spectral flux** (weight 0.6): L2 norm of the frame-to-frame mel difference. Consonant onsets produce large flux; sustained vowels produce almost none. This is consistent with Stilp & Kluender (2010), who showed spectral change rate correlates with perceptual importance in speech.
- **Energy** (weight 0.4): sum of mel bands per frame. Separates speech from silence.

The upper half of the mel bands get a 2x weight boost when computing flux. Consonants live in the high frequencies (2-8kHz range for fricatives and plosive bursts), so this makes the importance map more sensitive to them. Adding this brought Whisper WER at 3x down from 42.6% to 39.3%.

### Mimi importance (optional, `--mimi`)

`analyzer/importance.py`, `analyzer/mimi_mlx.py`

Mimi is Kyutai's neural audio codec. Its first codebook is distilled from WavLM, so it captures semantic content, not just acoustics. The remaining seven codebooks capture acoustic detail. Scoring looks at:

- Codebook 0 transitions (0.45 weight): semantic discontinuities
- Multi-codebook change rate (0.40): acoustic complexity
- Voice activity (0.15): energy above noise floor

Mimi importance and mel importance correlate at about 0.73. The mel version is 1000x faster and good enough for most use, so it's the default. Mimi is there for when you want the last bit of quality and don't mind waiting.

### Prosodic envelope

`analyzer/prosody.py`

The mel importance captures frame-level detail (which consonants matter) but has no sense of sentence-level rhythm. The prosodic envelope addresses this by extracting the syllable-rate energy contour (~4Hz low-pass filtered RMS) and using it to modulate importance scores. Stressed syllables get more time budget, unstressed ones get compressed further.

The modulation uses a floor of 0.5, so even unstressed consonants keep at least half their importance. Without the floor, consonants in unstressed syllables would get smeared.

Enabled by default. Disable with `--no-prosody`.

## Stage 2: rate schedule

`tsm/rate_schedule.py`

Converts the importance map into a per-frame playback rate that hits the target speed exactly.

1. Invert importance: high importance → low rate (play slower), low importance → high rate (skip faster). Range is 1x to 10x.
2. Gaussian smooth the importance curve (sigma=15 frames) to prevent the rate from being too spiky frame-to-frame.
3. Scale all rates so the total output duration matches target_speed. This is iterative (up to 20 rounds) because clipping rates to [1x, 10x] throws off the total.
4. Asymmetric slew limiter: rate *increases* (leaving important content) are capped at 0.3 per frame, but rate *decreases* (entering a consonant) are unconstrained. The original symmetric limiter created an audible "reverse speech" artifact. Each consonant had a gradual exponential-feeling fade-in because the rate ramped down slowly over ~260ms before reaching the consonant. Making it asymmetric means the rate snaps to slow immediately at consonant onsets.
5. Light post-smooth (Gaussian, sigma=2 frames) to round off the sharpest remaining spikes without killing contrast.
6. Final re-normalization to hit the target duration.

## Stage 3: time-scale modification

### Vocos neural vocoder (default)

`tsm/vocos_mlx.py`, `tsm/vocos_engine.py`

Instead of trying to fix phase coherence at the waveform level (which is what phase vocoders and PSOLA do, with varying degrees of success), just don't work at the waveform level. Work in the mel domain and let a neural network handle waveform synthesis.

The process:
1. Extract mel spectrogram from the input audio
2. Resample the mel along the time axis using cubic interpolation, following the rate curve from Stage 2
3. Apply light Gaussian smoothing (sigma=0.7) to the resampled mel to prevent staccato artifacts at frame boundaries
4. Feed the resampled mel to the Vocos decoder, which generates a waveform

Vocos (from Charactr/Hugging Face, `charactr/vocos-mel-24khz`) works in the frequency domain via ISTFT, which makes it fast. On Apple Silicon with MLX, the full pipeline runs at ~200x realtime.

The MLX implementation (`vocos_mlx.py`) ports the ConvNeXt backbone and ISTFT head to MLX for GPU inference. Falls back to the PyTorch version if MLX isn't available.

### Why not the classical approaches

Each of these was implemented, ABX-tested, and found wanting.

The **phase vocoder** (`tsm/phase_vocoder.py`) uses identity phase locking (Laroche & Dolson 1999), transient preservation, and noise-band randomization for sibilants. It's decent up to about 3x but sibilants get metallic and vowels pick up a "phasey" quality at higher ratios. Still the fallback when Vocos isn't available.

**TD-PSOLA** (`tsm/td_psola.py`) does pitch-synchronous overlap-add using CREPE pitch detection with glottal closure instant (GCI) refinement. Works well on voiced segments but falls apart on unvoiced consonants (t, k, s, f) where pitch detection is meaningless. The result sounds scrambled on exactly the segments you most want to protect.

The **CREPE hybrid** (`tsm/crepe_hybrid.py`) routes voiced segments to PSOLA and unvoiced to phase vocoder with 20ms crossfades. Marginally better than phase vocoder alone, but the voicing boundaries introduce subtle clicks. ABX testing slightly favored the plain phase vocoder.

**HPSS** (`tsm/hpss.py`) separates the signal into harmonic (vowels) and percussive (consonants) components, stretches them independently, and recombines. The problem is timing: the two streams drift apart during stretching, creating discontinuities when you put them back together. Kept as opt-in but not recommended.

## Post-processing

`cli.py` (_soft_clip_and_normalize)

Vocos can produce samples above 1.0, especially on plosives and sibilants. A soft tanh clipper kicks in above 0.8, then RMS normalization matches the output loudness to the input. Final peak limiting at 0.99 prevents digital clipping.

## Chunked processing

`parallel.py`

Files over 10 minutes get auto-chunked into 300-second segments with 1-second overlap. Each chunk is stretched independently, then the overlaps are crossfaded with a linear blend. This keeps memory usage bounded for 12-hour audiobooks.

## I/O

`io/decode.py`, `io/encode.py`

Decode and encode both shell out to ffmpeg. Decode resamples to 24kHz mono float32. Encode writes whatever format the output extension implies. For large files (>30s), decoding shows a progress bar based on `ffprobe` duration.

Streaming mode (`--stream`) outputs raw f32le PCM to stdout for piping to ffplay or further processing.

## Module map

```
src/osmium/
├── cli.py                  entry point, orchestrates the pipeline
├── parallel.py             chunked processing for large files
├── analyzer/
│   ├── importance.py       Mimi-based importance + ImportanceMap dataclass
│   ├── mel_importance.py   mel spectral flux + energy importance
│   ├── prosody.py          prosodic envelope extraction and modulation
│   ├── mimi.py             Mimi encoder (rustymimi)
│   ├── mimi_mlx.py         Mimi encoder (MLX, ~500x realtime)
│   ├── crepe_mlx.py        CREPE pitch detection (MLX)
│   ├── gci.py              glottal closure instant detection
│   └── aligner.py          phoneme alignment via torchaudio MMS_FA
├── tsm/
│   ├── vocos_mlx.py        Vocos vocoder (MLX) + mel extraction
│   ├── vocos_engine.py     Vocos vocoder (PyTorch fallback)
│   ├── rate_schedule.py    importance → rate curve conversion
│   ├── phase_vocoder.py    phase vocoder with identity phase locking
│   ├── td_psola.py         time-domain PSOLA
│   ├── crepe_hybrid.py     PSOLA/PV hybrid with voicing segmentation
│   ├── crepe_enhanced_pv.py CREPE-guided phase vocoder
│   ├── voiced_split.py     voicing segmentation
│   ├── hybrid.py           earlier HPSS-based hybrid
│   ├── hpss.py             harmonic-percussive separation
│   ├── wsola.py            waveform similarity OLA
│   └── stream.py           streaming processor
└── io/
    ├── decode.py           ffmpeg decode to 24kHz mono
    └── encode.py           ffmpeg encode to mp3/m4a/wav/flac
```
