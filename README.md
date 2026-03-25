# osmium

Speech acceleration that doesn't butcher consonants. Speeds up audiobooks and podcasts at 2x-4x while keeping the words intelligible.

Most time-stretching tools (sox, Rubber Band) apply a uniform rate across the whole signal. That's fine up to about 1.5x, but past 2x the consonants get mushy and everything sounds like it's underwater. Osmium allocates its time budget the way a natural fast speaker does: it protects consonant transients and compresses the parts you don't need (sustained vowels, pauses, breaths).

## Install

Requires Python 3.11+ and ffmpeg.

```bash
git clone https://github.com/user/osmium
cd osmium
uv sync
```

For Mimi neural importance (optional, slower but slightly better):
```bash
uv pip install -e '.[neural]'
```

## Usage

```bash
osmium audiobook.mp3 -s 3.0 -o output.mp3
osmium podcast.m4a -s 2.5 -o output.m4a
osmium chapter.wav -s 2.0 --uniform -o output.wav  # skip importance analysis
```

Stream to speakers:
```bash
osmium input.mp3 -s 3.0 --stream | ffplay -nodisp -f f32le -ar 24000 -ac 1 -
```

Export the importance map without processing:
```bash
osmium input.mp3 -s 3.0 --analyze-only -o importance.json
```

### Options

| Flag | Default | What it does |
|------|---------|--------------|
| `-s, --speed` | (required) | Target speed factor |
| `-o, --output` | | Output file path |
| `--stream` | | Raw PCM to stdout |
| `--uniform` | | Uniform rate, no importance analysis |
| `--mimi` | | Use Mimi neural codec for importance |
| `--no-prosody` | | Disable sentence-level rhythm preservation |
| `--resolution` | 20ms | Importance map time resolution |
| `--smoothing` | 0.7 | Mel temporal smoothing (0 = off) |
| `--chunk-size` | auto | Process in chunks of N seconds |
| `--analyze-only` | | Dump importance map as JSON |

## How it works

1. **Analyze** -- compute per-frame importance from the mel spectrogram (spectral flux + energy, with a boost for high-frequency consonant bands)
2. **Schedule** -- convert importance to a variable rate curve that hits the target speed while giving more time to important frames
3. **Stretch** -- resample the mel spectrogram according to the rate curve, then reconstruct audio with the Vocos neural vocoder

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## Sample clips

`samples/clips/` contains short clips (15s and 30s) extracted from the public domain LibriVox recording of [Moby Dick by Herman Melville](https://librivox.org/moby-dick-by-herman-melville). These are used for evaluation and listening comparisons.

Generate accelerated versions across all speeds and modes:
```bash
scripts/generate_accelerated.sh
```

This produces MP3s in `samples/clips/accelerated/{speed}/{mode}/` for speeds 2x–3.8x and three modes:
- **uniform** — flat rate, no importance analysis (`--uniform`)
- **no-mimi** — mel-based importance (default)
- **neural** — Mimi neural codec importance (`--mimi`)

## Evaluation

```bash
uv run scripts/eval_wer.py samples/clips/*.wav -s 3.0            # Whisper WER
uv run scripts/abx_test.py version_a.wav version_b.wav            # ABX listening test
```

## License

MIT
