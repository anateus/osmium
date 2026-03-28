# Architecture

Osmium has four stages: denoise, analyze importance, compute a rate schedule, and stretch the audio. Each stage was shaped by specific failures of the alternatives.

## The problem with uniform stretching

Tools like sox and Rubber Band apply the same rate to every part of the signal. At 2x that's fine. At 3x+ consonants lose their attack, sibilants smear, and intelligibility drops off a cliff. This matches what Janse (2004) found: natural fast speakers don't compress evenly. They keep consonant durations roughly intact and eat the time out of vowels and pauses. Osmium tries to do the same thing.

## Pipeline overview

```
audio file
  │
  ├─ decode (ffmpeg → 24kHz mono float32)
  │
  ├─ Stage 0: voice cleanup (on by default)
  │    spectral gating via noisereduce (--denoise gate/deep/demucs/none)
  │
  ├─ Stage 1: importance analysis
  │    mel spectrogram → spectral flux + energy → importance[0..1] per frame
  │    (optional: Mimi neural codec, prosodic envelope)
  │
  ├─ Stage 2: rate schedule
  │    importance → rate contrast compression (gamma) → variable rate curve
  │    → iterative normalization to hit target speed
  │
  ├─ Stage 3: time-scale modification
  │    resample mel → adaptive smoothing → HF de-emphasis → Vocos vocoder
  │    → spectral tilt correction → waveform
  │
  └─ encode (ffmpeg → mp3/m4a/wav/flac)
```

Processing runs at 24kHz mono internally because that's what both Mimi and Vocos expect.

## Stage 0: voice cleanup

`analyzer/denoise.py`, `analyzer/denoise_demucs.py`

Runs before importance analysis so the importance map and vocoder both see cleaner input. Three tiers, all using the `--denoise` flag:

- **gate** (default): Stationary spectral gating via noisereduce. Good for constant room noise and mic hiss. No extra dependencies, >100x realtime.
- **deep**: Non-stationary spectral gating via noisereduce with `stationary=False`. Adapts to varying noise profiles, better for compression artifacts. Same speed as gate.
- **demucs**: Full neural source separation using HTDemucs. Extracts the vocal stem, discarding everything else. Handles music beds and heavy artifacts. ~3x realtime on CPU. Requires `uv pip install -e '.[demucs]'`.
- **none**: Skip denoising entirely.

Output loudness is normalized to the pre-denoise RMS so all modes produce consistent volume.

## Stage 1: importance analysis

### Mel importance (default)

`analyzer/mel_importance.py`

The mel spectrogram is already computed for the Vocos vocoder, so importance comes nearly free. Two signals get combined:

- **Spectral flux** (weight 0.65): L2 norm of the frame-to-frame mel difference. Consonant onsets produce large flux; sustained vowels produce almost none. This is consistent with Stilp & Kluender (2010), who showed spectral change rate correlates with perceptual importance in speech.
- **Energy** (weight 0.35): sum of mel bands per frame. Separates speech from silence.

The upper half of the mel bands get a 2.5x weight boost when computing flux. Consonants live in the high frequencies (2-8kHz range for fricatives and plosive bursts), so this makes the importance map more sensitive to them. The boost was tuned from 2.0 to 2.5 and flux weight from 0.6 to 0.65 based on Whisper WER testing at 3x+.

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

### Phoneme-class importance floors (default)

`analyzer/phoneme_class.py`

The mel importance captures spectral change but can't distinguish phoneme types. Sustained fricatives and plosive closures have low spectral flux but are perceptually critical and nearly incompressible (Klatt 1979).

MMS_FA (torchaudio's forced alignment model, used for its CTC emissions only) classifies each frame into a phoneme class based on the most probable non-blank emission label. Each class has a literature-derived importance floor:

- **plosive** (0.92): stop bursts/closures are near-incompressible
- **fricative** (0.85): frication noise needs minimum duration
- **nasal** (0.78): less compressible than vowels
- **liquid/glide** (0.65): moderate compressibility
- **vowel** (0.25): most compressible speech segment
- **silence** (0.05): pauses compress first and most

The floor is applied after mel importance and prosodic modulation via `max(importance, phoneme_floor)`, guaranteeing consonant frames never drop below their class minimum regardless of prosodic context.

Enabled by default. Disable with `--no-phoneme`.

### Phoneme-aligned importance (optional, `--phoneme-align`)

`analyzer/phoneme_align.py`

For maximum precision, Whisper generates a transcript and MMS_FA force-aligns it to get exact phoneme boundaries and identities. The same class-to-floor table is applied per-segment. More accurate than CTC emission classification (catches plosive closures that CTC sees as silence) but adds a Whisper dependency and ~2-3x analysis time.

## Stage 2: rate schedule

`tsm/rate_schedule.py`

Converts the importance map into a per-frame playback rate that hits the target speed exactly.

1. Invert importance: high importance → low rate (play slower), low importance → high rate (skip faster). Range is 1x to 10x.
2. Gaussian smooth the importance curve (sigma=15 frames) to prevent the rate from being too spiky frame-to-frame.
3. Scale all rates so the total output duration matches target_speed. This is iterative (up to 20 rounds) because clipping rates to [1x, 10x] throws off the total.
4. **Rate contrast compression** (`--rate-gamma`, default 1.5): applies a power curve to the normalized rate distribution. With gamma > 1, mid-range rates get pulled down — content that is "somewhat unimportant" (unstressed vowels, brief pauses) gets less aggressive compression than a linear mapping would give. This reduces the perceived lurching between slow and fast sections at high speeds. gamma=1.0 gives the original linear behavior.
5. Asymmetric slew limiter: rate *increases* (leaving important content) are capped at 0.3 per frame, but rate *decreases* (entering a consonant) are unconstrained. The original symmetric limiter created an audible "reverse speech" artifact. Each consonant had a gradual exponential-feeling fade-in because the rate ramped down slowly over ~260ms before reaching the consonant. Making it asymmetric means the rate snaps to slow immediately at consonant onsets.
6. Light post-smooth (Gaussian, sigma=2 frames) to round off the sharpest remaining spikes without killing contrast.
7. Final re-normalization to hit the target duration.

## Stage 3: time-scale modification

### Vocos neural vocoder (default)

`tsm/vocos_mlx.py`, `tsm/vocos_engine.py`, `tsm/smooth.py`

Instead of trying to fix phase coherence at the waveform level (which is what phase vocoders and PSOLA do, with varying degrees of success), just don't work at the waveform level. Work in the mel domain and let a neural network handle waveform synthesis.

The process:
1. Extract mel spectrogram from the input audio
2. Resample the mel along the time axis using cubic interpolation, following the rate curve from Stage 2
3. **Adaptive smoothing** (`tsm/smooth.py`): instead of a fixed Gaussian sigma, scale smoothing based on the local compression ratio. Where the rate is high (many input frames compressed into few output frames), more smoothing is needed to prevent spectral discontinuities that Vocos turns into clicks. Where the rate is near 1x, minimal smoothing preserves consonant attacks. Implemented with 4 sigma buckets (ranging from sigma_min to sigma_max) with crossfade blending at bucket boundaries.
4. **HF de-emphasis** (`tsm/smooth.py`): applies a gentle high-frequency rolloff to the mel before Vocos synthesis, proportional to the local compression ratio. Sibilants in heavily-compressed regions get their brightness tamed (up to 5dB at the highest mel bins), plus a 1dB baseline rolloff everywhere to compensate for the vocoder's tendency to brighten. Starts at mel bin 40% (roughly 3kHz). This was tuned by ABX comparison at 3.8x.
5. Feed the processed mel to the Vocos decoder, which generates a waveform

Vocos (from Charactr/Hugging Face, `charactr/vocos-mel-24khz`) works in the frequency domain via ISTFT, which makes it fast. On Apple Silicon with MLX, the full pipeline runs at ~200x realtime.

The MLX implementation (`vocos_mlx.py`) ports the ConvNeXt backbone and ISTFT head to MLX for GPU inference. Falls back to the PyTorch version if MLX isn't available.

### Why not the classical approaches

Each of these was implemented, ABX-tested, and removed from the codebase. They're documented here so we don't repeat the experiments.

The **phase vocoder** used identity phase locking (Laroche & Dolson 1999), transient preservation, and noise-band randomization for sibilants. Decent up to about 3x but sibilants get metallic and vowels pick up a "phasey" quality at higher ratios.

**TD-PSOLA** did pitch-synchronous overlap-add using CREPE pitch detection (ported to MLX) with glottal closure instant (GCI) refinement. Works well on voiced segments but falls apart on unvoiced consonants (t, k, s, f) where pitch detection is meaningless. The result sounds scrambled on exactly the segments you most want to protect.

The **CREPE hybrid** routed voiced segments to PSOLA and unvoiced to phase vocoder with 20ms crossfades. Marginally better than phase vocoder alone, but the voicing boundaries introduce subtle clicks. ABX testing slightly favored the plain phase vocoder.

**CREPE-enhanced phase vocoder** used CREPE pitch tracking to improve phase locking in the phase vocoder. Marginal gains over the plain phase vocoder, not enough to justify the CREPE dependency.

**WSOLA** (waveform similarity overlap-add) selects overlap positions by maximizing cross-correlation. Avoids phase issues but produces audible echoing at high ratios because the similarity search locks onto the wrong pitch periods.

**HPSS** (harmonic-percussive source separation) separated the signal into harmonic (vowels) and percussive (consonants) components, stretched them independently, and recombined. The problem is timing: the two streams drift apart during stretching, creating discontinuities when you put them back together.

## Post-processing

`cli.py` (_match_spectral_tilt, _soft_clip_and_normalize)

Two post-processing steps after Vocos synthesis:

1. **Spectral tilt correction**: compares the average spectral shape of the input audio against the output and applies a smooth corrective EQ (capped at ±4.5dB per frequency bin, Gaussian-smoothed across bins). This catches any tonal shifts the vocoder introduces regardless of cause.

2. **Soft clipping and normalization**: Vocos can produce samples above 1.0, especially on plosives and sibilants. A soft tanh clipper kicks in above 0.8, then RMS normalization matches the output loudness to the original input (measured before denoising for consistency). Final peak limiting at 0.99 prevents digital clipping.

## Chunked processing

`parallel.py`

Files over 10 minutes get auto-chunked into 300-second segments with 1-second overlap. Each chunk is stretched independently, then the overlaps are crossfaded with a linear blend. This keeps memory usage bounded for 12-hour audiobooks.

## I/O

`io/decode.py`, `io/encode.py`

Decode and encode both shell out to ffmpeg. Decode resamples to 24kHz mono float32. Encode writes whatever format the output extension implies. For large files (>30s), decoding shows a progress bar based on `ffprobe` duration.

## Module map

```
src/osmium/
├── cli.py                  entry point, orchestrates the pipeline
├── parallel.py             chunked processing for large files
├── analyzer/
│   ├── importance.py       Mimi-based importance + ImportanceMap dataclass
│   ├── mel_importance.py   mel spectral flux + energy importance
│   ├── prosody.py          prosodic envelope extraction and modulation
│   ├── phoneme_class.py    MMS_FA phoneme class detection + importance floors
│   ├── phoneme_align.py    forced alignment phoneme importance (Whisper + MMS_FA)
│   ├── denoise.py          spectral gating via noisereduce (gate/deep modes)
│   ├── denoise_demucs.py   Demucs HTDemucs source separation wrapper
│   ├── mimi.py             Mimi encoder (rustymimi)
│   └── mimi_mlx.py         Mimi encoder (MLX, ~500x realtime)
├── tsm/
│   ├── vocos_mlx.py        Vocos vocoder (MLX) + mel extraction
│   ├── vocos_engine.py     Vocos vocoder (PyTorch fallback)
│   ├── smooth.py           adaptive mel smoothing + HF de-emphasis
│   └── rate_schedule.py    importance → rate curve with gamma compression
└── io/
    ├── decode.py           ffmpeg decode to 24kHz mono
    └── encode.py           ffmpeg encode to mp3/m4a/wav/flac
```
