# Phoneme-Aware Rate Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add phoneme-class importance floors so consonant frames are protected from over-compression, with a fast default (CTC emission classification) and an opt-in precise mode (forced alignment).

**Architecture:** MMS_FA CTC emissions classify each frame into a phoneme class (plosive/fricative/nasal/liquid/vowel/silence), each with a literature-derived importance floor. The floor is applied as `max(existing_importance, phoneme_floor)` after all other importance processing, guaranteeing consonants can't drop below their class minimum. An opt-in `--phoneme-align` mode uses Whisper transcription + MMS_FA forced alignment for exact phoneme boundaries.

**Tech Stack:** torchaudio (MMS_FA pipeline), torch, numpy. Optional: whisper (for `--phoneme-align`).

**Spec:** `docs/specs/2026-03-27-phoneme-aware-rate-scheduling-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/osmium/analyzer/phoneme_class.py` | **New.** Phoneme class constants, label→class mapping, MMS_FA inference, floor computation. Returns `ImportanceMap`. |
| `src/osmium/analyzer/phoneme_align.py` | **New.** Whisper transcription + MMS_FA forced alignment → per-phoneme `ImportanceMap`. |
| `src/osmium/analyzer/importance.py` | **Modify.** Remove `use_phoneme_alignment` param and `aligner.py` import. |
| `src/osmium/analyzer/aligner.py` | **Delete.** Superseded by new modules. |
| `src/osmium/cli.py` | **Modify.** Add `--no-phoneme` and `--phoneme-align` flags. Apply phoneme floor after prosodic modulation. |
| `tests/test_phoneme_class.py` | **New.** Unit tests for phoneme classification and floor application. |
| `tests/test_phoneme_align.py` | **New.** Unit tests for forced alignment path. |
| `scripts/compare_phoneme.py` | **New.** A/B/C comparison script. |
| `ARCHITECTURE.md` | **Modify.** Document phoneme-class analysis in Stage 1, update module map. |

---

### Task 1: Phoneme class constants and label mapping

**Files:**
- Create: `src/osmium/analyzer/phoneme_class.py`
- Create: `tests/test_phoneme_class.py`

- [ ] **Step 1: Write tests for phoneme class mapping**

```python
# tests/test_phoneme_class.py
import numpy as np
from osmium.analyzer.phoneme_class import (
    PHONEME_CLASS_FLOORS,
    LABEL_TO_CLASS,
    classify_frame,
)


def test_all_mms_fa_labels_are_mapped():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    for label in labels:
        if label == "-":
            continue
        assert label in LABEL_TO_CLASS, f"Label '{label}' not mapped"


def test_plosive_labels():
    for label in "tdkgbpc'":
        assert LABEL_TO_CLASS[label] == "plosive", f"'{label}' should be plosive"


def test_fricative_labels():
    for label in "szfvhx":
        assert LABEL_TO_CLASS[label] == "fricative", f"'{label}' should be fricative"


def test_nasal_labels():
    for label in "mn":
        assert LABEL_TO_CLASS[label] == "nasal"


def test_liquid_glide_labels():
    for label in "lrwyj":
        assert LABEL_TO_CLASS[label] == "liquid_glide"


def test_vowel_labels():
    for label in "aeiou":
        assert LABEL_TO_CLASS[label] == "vowel"


def test_fallback_labels_are_plosive():
    for label in "q*":
        assert LABEL_TO_CLASS[label] == "plosive"


def test_floor_values_are_ordered():
    floors = PHONEME_CLASS_FLOORS
    assert floors["plosive"] > floors["fricative"]
    assert floors["fricative"] > floors["nasal"]
    assert floors["nasal"] > floors["liquid_glide"]
    assert floors["liquid_glide"] > floors["vowel"]
    assert floors["vowel"] > floors["silence"]


def test_classify_frame_silence():
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.9)  # blank is index 0
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "silence"


def test_classify_frame_plosive():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    t_idx = labels.index("t")
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.1)  # blank low
    log_probs[t_idx] = np.log(0.7)  # t high
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "plosive"


def test_classify_frame_vowel():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    a_idx = labels.index("a")
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.1)
    log_probs[a_idx] = np.log(0.6)
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "vowel"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_phoneme_class.py -v
```

Expected: FAIL — `ImportError: cannot import name 'PHONEME_CLASS_FLOORS' from 'osmium.analyzer.phoneme_class'`

- [ ] **Step 3: Implement phoneme class constants and classify_frame**

```python
# src/osmium/analyzer/phoneme_class.py
import numpy as np

MMS_FA_LABELS = list("-aienuotsrmkldghybpwcvjzf'qx*")

LABEL_TO_CLASS = {
    # plosives
    "t": "plosive", "d": "plosive", "k": "plosive",
    "g": "plosive", "b": "plosive", "p": "plosive",
    "c": "plosive", "'": "plosive",
    # fricatives
    "s": "fricative", "z": "fricative", "f": "fricative",
    "v": "fricative", "h": "fricative", "x": "fricative",
    # nasals
    "m": "nasal", "n": "nasal",
    # liquids/glides
    "l": "liquid_glide", "r": "liquid_glide", "w": "liquid_glide",
    "y": "liquid_glide", "j": "liquid_glide",
    # vowels
    "a": "vowel", "e": "vowel", "i": "vowel",
    "o": "vowel", "u": "vowel",
    # fallback — conservative
    "q": "plosive", "*": "plosive",
}

PHONEME_CLASS_FLOORS = {
    "plosive": 0.92,
    "fricative": 0.85,
    "nasal": 0.78,
    "liquid_glide": 0.65,
    "vowel": 0.25,
    "silence": 0.05,
}


def classify_frame(
    log_probs: np.ndarray,
    blank_threshold: float = 0.8,
) -> str:
    probs = np.exp(log_probs)
    if probs[0] > blank_threshold:
        return "silence"
    non_blank_probs = probs[1:]
    best_idx = int(np.argmax(non_blank_probs))
    label = MMS_FA_LABELS[best_idx + 1]
    return LABEL_TO_CLASS.get(label, "plosive")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_phoneme_class.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/osmium/analyzer/phoneme_class.py tests/test_phoneme_class.py
git commit -m "feat: add phoneme class constants and frame classifier"
```

---

### Task 2: MMS_FA inference and floor computation

**Files:**
- Modify: `src/osmium/analyzer/phoneme_class.py`
- Modify: `tests/test_phoneme_class.py`

- [ ] **Step 1: Write tests for compute_phoneme_floors**

Add to `tests/test_phoneme_class.py`:

```python
from osmium.analyzer.phoneme_class import compute_phoneme_floors
from osmium.analyzer.importance import ImportanceMap


def test_compute_phoneme_floors_returns_importance_map():
    # Use synthetic log-probs to avoid loading MMS_FA model
    n_frames = 100
    n_labels = 29
    # All frames are blank (silence)
    log_probs = np.full((n_frames, n_labels), -10.0, dtype=np.float32)
    log_probs[:, 0] = np.log(0.95)
    result = compute_phoneme_floors(log_probs, duration=2.0)
    assert isinstance(result, ImportanceMap)
    assert len(result.scores) == n_frames
    assert result.duration == 2.0
    np.testing.assert_allclose(result.scores, 0.05, atol=1e-6)


def test_compute_phoneme_floors_plosive_frames():
    n_frames = 50
    n_labels = 29
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    t_idx = labels.index("t")
    log_probs = np.full((n_frames, n_labels), -10.0, dtype=np.float32)
    log_probs[:, 0] = np.log(0.1)
    log_probs[:, t_idx] = np.log(0.7)
    result = compute_phoneme_floors(log_probs, duration=1.0)
    np.testing.assert_allclose(result.scores, 0.92, atol=1e-6)


def test_compute_phoneme_floors_mixed():
    n_frames = 20
    n_labels = 29
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    a_idx = labels.index("a")
    t_idx = labels.index("t")
    log_probs = np.full((n_frames, n_labels), -10.0, dtype=np.float32)
    # First 10 frames: vowel
    log_probs[:10, 0] = np.log(0.1)
    log_probs[:10, a_idx] = np.log(0.7)
    # Last 10 frames: plosive
    log_probs[10:, 0] = np.log(0.1)
    log_probs[10:, t_idx] = np.log(0.7)
    result = compute_phoneme_floors(log_probs, duration=0.4)
    np.testing.assert_allclose(result.scores[:10], 0.25, atol=1e-6)
    np.testing.assert_allclose(result.scores[10:], 0.92, atol=1e-6)


def test_compute_phoneme_floors_zero_duration():
    log_probs = np.zeros((0, 29), dtype=np.float32)
    result = compute_phoneme_floors(log_probs, duration=0.0)
    assert len(result.scores) == 0


def test_compute_phoneme_floors_single_frame():
    n_labels = 29
    log_probs = np.full((1, n_labels), -10.0, dtype=np.float32)
    log_probs[0, 0] = np.log(0.95)
    result = compute_phoneme_floors(log_probs, duration=0.02)
    assert len(result.scores) == 1
    assert result.scores[0] == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_phoneme_class.py::test_compute_phoneme_floors_returns_importance_map -v
```

Expected: FAIL — `ImportError: cannot import name 'compute_phoneme_floors'`

- [ ] **Step 3: Implement compute_phoneme_floors**

Add to `src/osmium/analyzer/phoneme_class.py`:

```python
from osmium.analyzer.importance import ImportanceMap


def compute_phoneme_floors(
    log_probs: np.ndarray,
    duration: float,
    blank_threshold: float = 0.8,
) -> ImportanceMap:
    n_frames = log_probs.shape[0]
    if n_frames == 0 or duration <= 0:
        return ImportanceMap(
            scores=np.array([], dtype=np.float32),
            times=np.array([], dtype=np.float32),
            frame_rate=0.0,
            duration=max(duration, 0.0),
        )
    floors = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        cls = classify_frame(log_probs[i], blank_threshold)
        floors[i] = PHONEME_CLASS_FLOORS[cls]
    times = np.linspace(0, duration, n_frames)
    return ImportanceMap(
        scores=floors,
        times=times,
        frame_rate=n_frames / duration,
        duration=duration,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_phoneme_class.py -v
```

Expected: All PASS

- [ ] **Step 5: Add MMS_FA inference function**

Add to `src/osmium/analyzer/phoneme_class.py`:

```python
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from torchaudio.pipelines import MMS_FA as bundle
    _model = bundle.get_model()
    _model.eval()
    return _model


def analyze_phoneme_class(
    samples: np.ndarray,
    sample_rate: int = 24000,
    blank_threshold: float = 0.8,
) -> ImportanceMap:
    import torch
    import torchaudio

    model = _load_model()
    duration = len(samples) / sample_rate
    target_sr = 16000

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    with torch.no_grad():
        emission, _ = model(waveform)

    log_probs = emission[0].numpy()
    return compute_phoneme_floors(log_probs, duration, blank_threshold)
```

- [ ] **Step 6: Commit**

```bash
git add src/osmium/analyzer/phoneme_class.py tests/test_phoneme_class.py
git commit -m "feat: add MMS_FA phoneme floor computation"
```

---

### Task 3: Remove aligner.py and clean up importance.py

**Files:**
- Delete: `src/osmium/analyzer/aligner.py`
- Modify: `src/osmium/analyzer/importance.py`

- [ ] **Step 1: Delete aligner.py**

```bash
rm src/osmium/analyzer/aligner.py
```

- [ ] **Step 2: Update importance.py — remove use_phoneme_alignment**

Read `src/osmium/analyzer/importance.py` and remove:
- The `use_phoneme_alignment` parameter from `compute_importance()`
- The `weight_mimi` and `weight_phoneme` parameters
- The entire `if use_phoneme_alignment:` block and the `from osmium.analyzer.aligner import` inside it
- Simplify to just return `_mimi_importance(codes, samples)`

The function becomes:

```python
def compute_importance(
    codes: MimiCodes,
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> ImportanceMap:
    return _mimi_importance(codes, samples)
```

- [ ] **Step 3: Verify no other imports of aligner.py or use_phoneme_alignment remain**

```bash
uv run grep -r "aligner\|use_phoneme_alignment" src/osmium/ --include="*.py"
```

Expected: no results

- [ ] **Step 4: Run existing tests to verify nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add -u src/osmium/analyzer/aligner.py src/osmium/analyzer/importance.py
git commit -m "refactor: remove aligner.py, simplify importance.py

Superseded by phoneme_class.py. The old aligner had a double-softmax
bug (applied softmax on log-softmax output) and an inferior
classification approach."
```

---

### Task 4: Wire phoneme floors into cli.py (Path C only)

**Files:**
- Modify: `src/osmium/cli.py`

This task wires only the default Path C (phoneme-class floors). The `--phoneme-align` flag is added here but the import is deferred to Task 6 after `phoneme_align.py` exists.

- [ ] **Step 1: Add CLI flags**

Add two new click options to the `main` function in `src/osmium/cli.py`:

```python
@click.option("--no-phoneme", is_flag=True, help="Disable phoneme-class importance floors")
@click.option("--phoneme-align", is_flag=True, help="Use forced alignment for precise phoneme boundaries (requires whisper)")
```

Add `no_phoneme` and `phoneme_align` to the `main()` signature, pass through to `_batch_mode()` and `_process_file()`.

- [ ] **Step 2: Add validation**

In `main()`, after existing validation:

```python
if phoneme_align and uniform:
    raise click.UsageError("--phoneme-align cannot be combined with --uniform")
```

- [ ] **Step 3: Wire phoneme floor into _process_file**

In `_process_file()`, after prosodic modulation (line ~138) and before `resample_importance` (line ~140), add:

```python
if not no_phoneme and not phoneme_align:
    from osmium.analyzer.phoneme_class import analyze_phoneme_class
    phoneme_task = progress.add_task("Analyzing", total=None, status="phoneme class")
    phoneme_floors = analyze_phoneme_class(audio.samples, audio.sample_rate)
    progress.remove_task(phoneme_task)
    phoneme_resampled = np.interp(imp.times, phoneme_floors.times, phoneme_floors.scores)
    imp = ImportanceMap(
        scores=np.maximum(imp.scores, phoneme_resampled),
        times=imp.times,
        frame_rate=imp.frame_rate,
        duration=imp.duration,
    )
```

This goes *after* prosodic modulation but *before* `resample_importance` and `importance_to_rate_schedule`. This placement means `--analyze-only` output will include the phoneme-floor-adjusted importance, as the spec requires.

- [ ] **Step 4: Test the CLI flags parse correctly**

```bash
uv run osmium --help 2>&1 | grep -E "phoneme|no-phoneme"
```

Expected: both `--no-phoneme` and `--phoneme-align` appear in help output

- [ ] **Step 5: Commit**

```bash
git add src/osmium/cli.py
git commit -m "feat: wire phoneme-class floors into CLI (Path C default)

MMS_FA CTC emission classification sets per-frame importance floors.
--no-phoneme disables for A/B comparison.
--phoneme-align flag added (wired in later task)."
```

---

### Task 5: Implement phoneme_align.py (Path B)

**Files:**
- Create: `src/osmium/analyzer/phoneme_align.py`
- Create: `tests/test_phoneme_align.py`

- [ ] **Step 1: Write tests for the alignment path**

```python
# tests/test_phoneme_align.py
import numpy as np
from osmium.analyzer.phoneme_align import (
    phoneme_segments_to_importance,
)
from osmium.analyzer.importance import ImportanceMap
from osmium.analyzer.phoneme_class import PHONEME_CLASS_FLOORS


def test_segments_to_importance_single_vowel():
    segments = [("a", 0.0, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=50.0)
    assert isinstance(result, ImportanceMap)
    np.testing.assert_allclose(
        result.scores, PHONEME_CLASS_FLOORS["vowel"], atol=1e-6
    )


def test_segments_to_importance_plosive_vowel():
    segments = [("t", 0.0, 0.1), ("a", 0.1, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=100.0)
    n_plosive = int(0.1 * 100)
    assert result.scores[0] == PHONEME_CLASS_FLOORS["plosive"]
    assert result.scores[-1] == PHONEME_CLASS_FLOORS["vowel"]


def test_segments_to_importance_covers_full_duration():
    segments = [("s", 0.0, 0.5), ("a", 0.5, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=50.0)
    assert len(result.scores) == 50
    assert result.duration == 1.0


def test_segments_to_importance_empty():
    result = phoneme_segments_to_importance([], duration=1.0, frame_rate=50.0)
    assert len(result.scores) == 50
    np.testing.assert_allclose(
        result.scores, PHONEME_CLASS_FLOORS["silence"], atol=1e-6
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_phoneme_align.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement phoneme_segments_to_importance**

```python
# src/osmium/analyzer/phoneme_align.py
import numpy as np
from osmium.analyzer.importance import ImportanceMap
from osmium.analyzer.phoneme_class import LABEL_TO_CLASS, PHONEME_CLASS_FLOORS


def phoneme_segments_to_importance(
    segments: list[tuple[str, float, float]],
    duration: float,
    frame_rate: float = 50.0,
) -> ImportanceMap:
    n_frames = max(1, int(duration * frame_rate))
    scores = np.full(n_frames, PHONEME_CLASS_FLOORS["silence"], dtype=np.float32)
    times = np.linspace(0, duration, n_frames)

    for label, start, end in segments:
        cls = LABEL_TO_CLASS.get(label, "plosive")
        floor = PHONEME_CLASS_FLOORS[cls]
        i_start = int(start * frame_rate)
        i_end = min(int(end * frame_rate), n_frames)
        scores[i_start:i_end] = floor

    return ImportanceMap(
        scores=scores,
        times=times,
        frame_rate=frame_rate,
        duration=duration,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_phoneme_align.py -v
```

Expected: All PASS

- [ ] **Step 5: Add the full analyze_phoneme_aligned function**

Add to `src/osmium/analyzer/phoneme_align.py`.

**Note:** Kyutai STT (`stt-1b-en_fr`) is deferred to a future task. V1 uses Whisper only.

```python
def analyze_phoneme_aligned(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> ImportanceMap:
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle
    from osmium.analyzer.phoneme_class import _load_model

    duration = len(samples) / sample_rate
    target_sr = 16000

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    try:
        import whisper
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(
            samples.astype(np.float32), language="en",
        )
        transcript = result["text"].strip()
    except ImportError:
        raise ImportError(
            "--phoneme-align requires whisper: uv pip install openai-whisper"
        )

    if not transcript:
        return phoneme_segments_to_importance([], duration)

    model = _load_model()  # reuse singleton from phoneme_class.py
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.no_grad():
        emission, _ = model(waveform)

    tokens = tokenizer(transcript)
    token_spans = aligner(emission[0], torch.tensor(tokens))

    labels = bundle.get_labels()
    frame_dur = duration / emission.shape[1]
    segments = []
    for span in token_spans:
        label = labels[span.token]
        start = span.start * frame_dur
        end = (span.end + 1) * frame_dur
        segments.append((label, start, end))

    return phoneme_segments_to_importance(segments, duration)
```

- [ ] **Step 6: Commit**

```bash
git add src/osmium/analyzer/phoneme_align.py tests/test_phoneme_align.py
git commit -m "feat: add forced-alignment phoneme importance (Path B)

Uses Whisper transcription + MMS_FA forced alignment for exact
phoneme boundaries. Opt-in via --phoneme-align flag."
```

---

### Task 6: Wire --phoneme-align into cli.py (Path B)

**Files:**
- Modify: `src/osmium/cli.py`

Now that `phoneme_align.py` exists (Task 5), wire the `--phoneme-align` flag.

- [ ] **Step 1: Add phoneme-align block to _process_file**

In `_process_file()`, right after the Path C phoneme-floor block added in Task 4, add:

```python
if phoneme_align:
    from osmium.analyzer.phoneme_align import analyze_phoneme_aligned
    align_task = progress.add_task("Analyzing", total=None, status="forced alignment")
    aligned_imp = analyze_phoneme_aligned(audio.samples, audio.sample_rate)
    progress.remove_task(align_task)
    aligned_resampled = np.interp(imp.times, aligned_imp.times, aligned_imp.scores)
    imp = ImportanceMap(
        scores=np.maximum(imp.scores, aligned_resampled),
        times=imp.times,
        frame_rate=imp.frame_rate,
        duration=imp.duration,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/osmium/cli.py
git commit -m "feat: wire --phoneme-align forced alignment into CLI (Path B)"
```

---

### Task 7: Comparison script

**Files:**
- Create: `scripts/compare_phoneme.py`

- [ ] **Step 1: Implement the comparison script**

```python
#!/usr/bin/env python
"""Compare phoneme-aware rate scheduling configurations via WER/CER."""
import argparse
import subprocess
import tempfile
import json
from pathlib import Path


def run_osmium(input_file, speed, output_file, extra_args=None):
    cmd = ["uv", "run", "osmium", input_file, "-s", str(speed), "-o", output_file]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def run_wer(ref_file, test_file, whisper_model="small"):
    cmd = [
        "uv", "run", "python", "scripts/eval_wer.py",
        "--ref", ref_file, "--test", test_file,
        "--whisper-model", whisper_model,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in result.stdout.strip().split("\n"):
        if "WER" in line and "CER" in line:
            parts = line.split()
            wer = float(parts[parts.index("WER") + 1].rstrip("%,"))
            cer = float(parts[parts.index("CER") + 1].rstrip("%,"))
            return wer, cer
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Compare phoneme-aware configurations")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--speeds", default="2.0,3.0,4.0,5.0", help="Comma-separated speeds")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--output-dir", default=None, help="Dir for temp outputs")
    args = parser.parse_args()

    speeds = [float(s) for s in args.speeds.split(",")]
    configs = [
        ("mel-only", ["--no-phoneme"]),
        ("phoneme-class", []),
        ("phoneme-align", ["--phoneme-align"]),
    ]

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(args.output_dir) if args.output_dir else Path(tmpdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for speed in speeds:
            results[speed] = {}
            for name, extra in configs:
                outfile = str(outdir / f"{name}_{speed}x.wav")
                try:
                    run_osmium(args.input, speed, outfile, extra)
                    wer, cer = run_wer(args.input, outfile, args.whisper_model)
                    results[speed][name] = {"wer": wer, "cer": cer}
                except Exception as e:
                    results[speed][name] = {"wer": None, "cer": None, "error": str(e)}

    print("\n## Phoneme-Aware Rate Scheduling Comparison\n")
    print(f"Input: `{args.input}`\n")
    print("| Speed | Config | WER (%) | CER (%) |")
    print("|---|---|---|---|")
    for speed in speeds:
        for name in ["mel-only", "phoneme-class", "phoneme-align"]:
            r = results[speed].get(name, {})
            wer = f"{r['wer']:.1f}" if r.get("wer") is not None else "N/A"
            cer = f"{r['cer']:.1f}" if r.get("cer") is not None else "N/A"
            err = f" ({r['error'][:40]})" if r.get("error") else ""
            print(f"| {speed}x | {name} | {wer}{err} | {cer} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

```bash
uv run python scripts/compare_phoneme.py --help
```

Expected: help text with `input`, `--speeds`, `--whisper-model` args

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_phoneme.py
git commit -m "feat: add phoneme A/B/C comparison script

Runs osmium at multiple speeds with mel-only, phoneme-class, and
phoneme-align configs, reports WER/CER via Whisper."
```

---

### Task 8: Update ARCHITECTURE.md

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Add phoneme-class analysis to Stage 1 section**

After the "Prosodic envelope" subsection in Stage 1, add:

```markdown
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
```

- [ ] **Step 2: Update the module map**

Add `phoneme_class.py` and `phoneme_align.py` to the analyzer section, remove `aligner.py`:

```
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
```

- [ ] **Step 3: Verify the doc reads correctly**

Read through the full ARCHITECTURE.md and confirm the flow makes sense.

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: add phoneme-class analysis to architecture

Documents phoneme-class importance floors (default) and forced
alignment path (opt-in) in Stage 1. Updates module map."
```

---

### Task 9: Integration test — run the full pipeline

- [ ] **Step 1: Run with default phoneme floors on a sample file**

```bash
uv run osmium samples/some_sample.wav -s 3.0 -o /tmp/test_phoneme_default.wav
```

Verify it completes without errors and the "phoneme class" analysis step appears in output.

- [ ] **Step 2: Run with --no-phoneme for comparison**

```bash
uv run osmium samples/some_sample.wav -s 3.0 -o /tmp/test_no_phoneme.wav --no-phoneme
```

Verify it completes without the phoneme analysis step.

- [ ] **Step 3: Verify --phoneme-align + --uniform rejects**

```bash
uv run osmium samples/some_sample.wav -s 3.0 -o /tmp/test.wav --phoneme-align --uniform 2>&1
```

Expected: `Error: --phoneme-align cannot be combined with --uniform`

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 5: Final commit with any fixes**

If any integration issues were found and fixed, commit them.
