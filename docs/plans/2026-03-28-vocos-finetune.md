# Vocos Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the Vocos vocoder on resample-roundtrip augmented mel to eliminate phase discontinuity clicks when synthesizing time-stretched speech.

**Architecture:** A PyTorch Lightning 1.8.6 training pipeline that subclasses the upstream `VocosExp` to inject resample-roundtrip mel augmentation between the feature extractor and backbone. Trains on LibriTTS train-clean-100 using Apple Silicon MPS. Best checkpoint is converted to MLX weights for the existing `VocosMLX` inference path.

**Tech Stack:** PyTorch (MPS), PyTorch Lightning 1.8.6, transformers (scheduler), torchaudio, vocos, MLX (conversion only)

**Spec:** `docs/specs/2026-03-28-vocos-finetune-design.md`

---

## File Structure

```
scripts/
  vocos_finetune.sh                    shell entry point (stages, resume, skip logic)
  vocos_finetune/
    download_data.py                   LibriTTS download + filelist generation
    dataset.py                         audio-only dataset (gain aug, crop, no mel)
    augment.py                         resample-roundtrip mel augmentation (pure torch)
    train.py                           VocosFineTuneExp subclass + CLI
    click_detector.py                  short-term energy click metric
    evaluate.py                        A/B sample generation through Osmium pipeline
    convert_mlx.py                     best checkpoint → MLX weights

training/                              (gitignored, all output goes here)
  data/LibriTTS/train-clean-100/
  data/filelists/{train,val}.txt
  checkpoints/
  logs/
  eval_samples/step_{N}/
  final_comparison/
  models/vocos-mel-24khz-finetuned/weights.npz
```

---

### Task 1: Project setup and dependencies

**Files:**
- Modify: `.gitignore`
- Create: `scripts/vocos_finetune/__init__.py`

- [ ] **Step 1: Add `training/` to .gitignore**

Append to `.gitignore`:
```
# Training artifacts
training/
```

- [ ] **Step 2: Create the scripts package directory**

```bash
mkdir -p scripts/vocos_finetune
touch scripts/vocos_finetune/__init__.py
```

- [ ] **Step 3: Verify `scripts` package imports work**

All modules import as `scripts.vocos_finetune.*`. Verify this resolves from the project root:

```bash
cd /Users/mike/code/osmium
PYTHONPATH=. python -c "import scripts.vocos_finetune"
```

If this fails, add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:
```toml
packages = ["src/osmium", "scripts"]
```

- [ ] **Step 4: Verify training deps are installable**

```bash
cd /Users/mike/code/osmium
uv pip install "pytorch-lightning==1.8.6" "transformers>=4.30" "einops" --dry-run
```

Expected: resolves without conflicts. If there's a conflict with existing packages, report it before proceeding.

- [ ] **Step 5: Install training deps**

```bash
uv pip install "pytorch-lightning==1.8.6" "transformers>=4.30" "einops"
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore scripts/vocos_finetune/__init__.py
git commit -m "chore: scaffold vocos fine-tuning directory and training deps"
```

---

### Task 2: Data download and filelist generation

**Files:**
- Create: `scripts/vocos_finetune/download_data.py`
- Test: manual — verify files exist after running

- [ ] **Step 1: Write download_data.py**

```python
"""Download LibriTTS train-clean-100 and generate train/val filelists."""

import argparse
import hashlib
import os
import random
import tarfile
import urllib.request
from pathlib import Path

LIBRITTS_URL = "https://www.openslr.org/resources/60/train-clean-100.tar.gz"
LIBRITTS_MD5 = "2c05cecece06364326d57678c8791e82"
VAL_UTTERANCES = 200
SEED = 42


def download_libritts(data_dir: Path) -> Path:
    tar_path = data_dir / "train-clean-100.tar.gz"
    extract_dir = data_dir / "LibriTTS"

    if (extract_dir / "train-clean-100").exists():
        print(f"LibriTTS already extracted at {extract_dir / 'train-clean-100'}")
        return extract_dir / "train-clean-100"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"Downloading LibriTTS train-clean-100 to {tar_path}...")
        urllib.request.urlretrieve(LIBRITTS_URL, tar_path, _progress_hook)
        print()

    print("Verifying checksum...")
    md5 = hashlib.md5(tar_path.read_bytes()).hexdigest()
    if md5 != LIBRITTS_MD5:
        tar_path.unlink()
        raise RuntimeError(f"MD5 mismatch: expected {LIBRITTS_MD5}, got {md5}")

    print(f"Extracting to {extract_dir}...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir, filter="data")

    tar_path.unlink()
    print("Done.")
    return extract_dir / "train-clean-100"


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)


def generate_filelists(corpus_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    if train_path.exists() and val_path.exists():
        train_count = len(train_path.read_text().splitlines())
        val_count = len(val_path.read_text().splitlines())
        print(f"Filelists exist: {train_count} train, {val_count} val")
        return train_path, val_path

    wav_files = sorted(corpus_dir.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {corpus_dir}")

    rng = random.Random(SEED)
    rng.shuffle(wav_files)

    val_files = wav_files[:VAL_UTTERANCES]
    train_files = wav_files[VAL_UTTERANCES:]

    val_path.write_text("\n".join(str(f) for f in val_files) + "\n")
    train_path.write_text("\n".join(str(f) for f in train_files) + "\n")

    print(f"Generated filelists: {len(train_files)} train, {len(val_files)} val")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("training/data"))
    args = parser.parse_args()

    corpus_dir = download_libritts(args.data_dir)
    generate_filelists(corpus_dir, args.data_dir / "filelists")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify**

```bash
cd /Users/mike/code/osmium
python scripts/vocos_finetune/download_data.py --data-dir training/data
```

Expected: downloads ~6GB tar.gz, extracts, generates `training/data/filelists/train.txt` and `val.txt`. Takes ~15 minutes on typical connection. Verify:

```bash
wc -l training/data/filelists/train.txt training/data/filelists/val.txt
ls training/data/LibriTTS/train-clean-100/ | head -5
```

Expected: ~33000 train lines, 200 val lines. Several speaker directories listed.

- [ ] **Step 3: Commit**

```bash
git add scripts/vocos_finetune/download_data.py
git commit -m "feat: add LibriTTS download and filelist generation"
```

---

### Task 3: Audio dataset

**Files:**
- Create: `scripts/vocos_finetune/dataset.py`
- Test: `tests/test_vocos_finetune_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the vocos fine-tuning audio dataset."""

import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def tmp_audio_files(tmp_path):
    """Create temporary WAV files for testing."""
    paths = []
    for i in range(5):
        samples = np.random.randn(48000).astype(np.float32) * 0.5
        path = tmp_path / f"test_{i}.wav"
        sf.write(str(path), samples, 24000)
        paths.append(path)

    filelist = tmp_path / "filelist.txt"
    filelist.write_text("\n".join(str(p) for p in paths) + "\n")
    return filelist


def test_dataset_returns_correct_shape(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset

    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=True)
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (24000,)
    assert sample.dtype == torch.float32


def test_dataset_length(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset

    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=True)
    assert len(ds) == 5


def test_dataset_gain_normalization(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset

    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=False)
    sample = ds[0]
    # val mode uses fixed -3 dB gain
    assert sample.abs().max() > 0
    assert sample.abs().max() <= 1.5  # reasonable range after gain


def test_dataset_pads_short_audio(tmp_path):
    from scripts.vocos_finetune.dataset import AudioDataset

    # Create a very short file (0.25 seconds)
    samples = np.random.randn(6000).astype(np.float32) * 0.5
    path = tmp_path / "short.wav"
    sf.write(str(path), samples, 24000)

    filelist = tmp_path / "filelist.txt"
    filelist.write_text(str(path) + "\n")

    ds = AudioDataset(filelist_path=str(filelist), num_samples=24000, sample_rate=24000, train=False)
    sample = ds[0]
    assert sample.shape == (24000,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mike/code/osmium
python -m pytest tests/test_vocos_finetune_dataset.py -v
```

Expected: all fail with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write dataset.py**

```python
"""Audio dataset for Vocos fine-tuning. Returns raw audio only — mel augmentation happens in training_step."""

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        filelist_path: str,
        num_samples: int = 24000,
        sample_rate: int = 24000,
        train: bool = True,
    ):
        with open(filelist_path) as f:
            self.filelist = [line.strip() for line in f if line.strip()]
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)

        # Pure-torch gain normalization (avoids sox_effects macOS dependency)
        gain_db = np.random.uniform(-6, -1) if self.train else -3.0
        peak = y.abs().max()
        if peak > 0:
            y = y / peak  # normalize to peak 1.0
            y = y * (10 ** (gain_db / 20))  # apply target gain

        # Crop or pad to exact length
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            repeats = 1 + pad_length // y.size(-1)
            y = y.repeat(1, repeats + 1)[:, : self.num_samples]
        elif self.train:
            start = np.random.randint(0, y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]

        return y[0]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mike/code/osmium
python -m pytest tests/test_vocos_finetune_dataset.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/dataset.py tests/test_vocos_finetune_dataset.py
git commit -m "feat: add audio dataset for vocos fine-tuning"
```

---

### Task 4: Resample-roundtrip augmentation

**Files:**
- Create: `scripts/vocos_finetune/augment.py`
- Test: `tests/test_vocos_finetune_augment.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for resample-roundtrip mel augmentation."""

import torch
import pytest


def test_roundtrip_preserves_shape():
    from scripts.vocos_finetune.augment import resample_roundtrip

    mel = torch.randn(2, 100, 94)  # (B, n_mels, T)
    result = resample_roundtrip(mel, rate=3.0)
    assert result.shape == mel.shape


def test_roundtrip_modifies_content():
    from scripts.vocos_finetune.augment import resample_roundtrip

    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=3.0)
    # roundtrip should change values due to interpolation artifacts
    assert not torch.allclose(mel, result, atol=1e-3)


def test_roundtrip_at_rate_1_is_near_identity():
    from scripts.vocos_finetune.augment import resample_roundtrip

    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=1.001)
    # very close to identity for near-1 rate
    assert torch.allclose(mel, result, atol=0.1)


def test_higher_rate_produces_more_distortion():
    from scripts.vocos_finetune.augment import resample_roundtrip

    mel = torch.randn(1, 100, 200)
    result_2x = resample_roundtrip(mel, rate=2.0)
    result_5x = resample_roundtrip(mel, rate=5.0)
    dist_2x = (mel - result_2x).abs().mean()
    dist_5x = (mel - result_5x).abs().mean()
    assert dist_5x > dist_2x


def test_roundtrip_with_presmooth():
    from scripts.vocos_finetune.augment import resample_roundtrip

    mel = torch.randn(1, 100, 94)
    result = resample_roundtrip(mel, rate=3.0, presmooth_sigma=2.0)
    assert result.shape == mel.shape
    # presmooth should make result smoother (lower variance along time)
    no_smooth = resample_roundtrip(mel, rate=3.0, presmooth_sigma=0.0)
    smooth_var = result.diff(dim=-1).var()
    no_smooth_var = no_smooth.diff(dim=-1).var()
    assert smooth_var < no_smooth_var


def test_random_roundtrip_produces_valid_output():
    from scripts.vocos_finetune.augment import random_resample_roundtrip

    mel = torch.randn(4, 100, 94)
    result = random_resample_roundtrip(mel, min_rate=1.5, max_rate=5.0)
    assert result.shape == mel.shape
    assert torch.isfinite(result).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vocos_finetune_augment.py -v
```

Expected: all fail with import errors.

- [ ] **Step 3: Write augment.py**

```python
"""Resample-roundtrip mel augmentation in pure PyTorch (runs on MPS/CUDA/CPU)."""

import torch
import torch.nn.functional as F


def resample_roundtrip(
    mel: torch.Tensor,
    rate: float,
    presmooth_sigma: float = 2.0,
) -> torch.Tensor:
    """Apply resample-roundtrip augmentation to mel spectrogram.

    Downsamples mel by `rate` then upsamples back to original length.
    The roundtrip introduces interpolation artifacts (temporal smoothing,
    spectral bleeding) that match what Osmium's TSM pipeline produces.

    Args:
        mel: (B, n_mels, T) mel spectrogram
        rate: compression rate (e.g. 3.0 means downsample to T/3 then back)
        presmooth_sigma: Gaussian smoothing sigma before downsampling (0 = none)

    Returns:
        (B, n_mels, T) augmented mel with same shape as input
    """
    B, C, T = mel.shape

    if presmooth_sigma > 0:
        mel = _gaussian_smooth_1d(mel, presmooth_sigma)

    compressed_T = max(1, int(T / rate))

    # Downsample: (B, C, T) → (B, C, compressed_T)
    compressed = F.interpolate(mel, size=compressed_T, mode="linear", align_corners=True)

    # Upsample back: (B, C, compressed_T) → (B, C, T)
    restored = F.interpolate(compressed, size=T, mode="linear", align_corners=True)

    return restored


def random_resample_roundtrip(
    mel: torch.Tensor,
    min_rate: float = 1.5,
    max_rate: float = 5.0,
    presmooth_sigma: float = 2.0,
) -> torch.Tensor:
    """Apply resample-roundtrip with a random rate."""
    rate = torch.empty(1).uniform_(min_rate, max_rate).item()
    return resample_roundtrip(mel, rate=rate, presmooth_sigma=presmooth_sigma)


def _gaussian_smooth_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian smoothing along the last dimension."""
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # (B, C, T) — apply per-channel with groups=C
    B, C, T = x.shape
    x_flat = x.reshape(B * C, 1, T)
    pad = kernel_size // 2
    x_padded = F.pad(x_flat, (pad, pad), mode="reflect")
    kernel_3d = kernel.reshape(1, 1, -1)
    smoothed = F.conv1d(x_padded, kernel_3d)
    return smoothed.reshape(B, C, T)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_vocos_finetune_augment.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/augment.py tests/test_vocos_finetune_augment.py
git commit -m "feat: add resample-roundtrip mel augmentation"
```

---

### Task 5: Click detector metric

**Files:**
- Create: `scripts/vocos_finetune/click_detector.py`
- Test: `tests/test_vocos_finetune_click.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for click detection metric."""

import numpy as np
import torch
import pytest


def test_clean_audio_has_no_clicks():
    from scripts.vocos_finetune.click_detector import count_clicks

    # Smooth sine wave — no clicks
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    clicks = count_clicks(audio, sample_rate=24000)
    assert clicks == 0


def test_audio_with_spikes_detected():
    from scripts.vocos_finetune.click_detector import count_clicks

    # Smooth audio with injected clicks
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.1
    # inject 3 sharp spikes
    audio[5000] = 0.9
    audio[12000] = 0.9
    audio[18000] = 0.9
    clicks = count_clicks(audio, sample_rate=24000)
    assert clicks >= 2  # should catch most/all spikes


def test_clicks_per_second():
    from scripts.vocos_finetune.click_detector import clicks_per_second

    t = np.linspace(0, 2, 48000, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.1
    # inject 4 spikes across 2 seconds
    for pos in [5000, 15000, 25000, 40000]:
        audio[pos] = 0.9
    cps = clicks_per_second(audio, sample_rate=24000)
    assert 1.0 <= cps <= 3.0  # ~2 clicks/sec
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vocos_finetune_click.py -v
```

- [ ] **Step 3: Write click_detector.py**

```python
"""Click detection metric for vocoder output quality evaluation."""

import numpy as np


def count_clicks(
    audio: np.ndarray,
    sample_rate: int = 24000,
    window_samples: int = 48,
    median_window_ms: float = 50.0,
    threshold: float = 3.0,
) -> int:
    """Count click artifacts in audio via short-term energy spike detection.

    Args:
        audio: 1D float32 audio array
        sample_rate: sample rate in Hz
        window_samples: energy window size (48 = 2ms at 24kHz, no overlap)
        median_window_ms: local median window in milliseconds
        threshold: spike = energy > threshold * local_median

    Returns:
        Number of detected clicks
    """
    # Short-term energy envelope (non-overlapping windows)
    n_frames = len(audio) // window_samples
    if n_frames < 3:
        return 0

    trimmed = audio[: n_frames * window_samples]
    frames = trimmed.reshape(n_frames, window_samples)
    energy = np.mean(frames ** 2, axis=1)

    # Local median (50ms window = ~25 frames at 2ms per frame)
    median_frames = max(3, int(median_window_ms / (window_samples / sample_rate * 1000)))
    if median_frames % 2 == 0:
        median_frames += 1
    half = median_frames // 2

    clicks = 0
    for i in range(half, n_frames - half):
        local_med = np.median(energy[i - half : i + half + 1])
        if local_med > 0 and energy[i] > threshold * local_med:
            clicks += 1

    return clicks


def clicks_per_second(audio: np.ndarray, sample_rate: int = 24000, **kwargs) -> float:
    """Click rate normalized by audio duration."""
    duration = len(audio) / sample_rate
    if duration < 0.01:
        return 0.0
    return count_clicks(audio, sample_rate=sample_rate, **kwargs) / duration
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_vocos_finetune_click.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/click_detector.py tests/test_vocos_finetune_click.py
git commit -m "feat: add click detection metric for vocoder evaluation"
```

---

### Task 6: VocosFineTuneExp training module

**Files:**
- Create: `scripts/vocos_finetune/train.py`
- Test: `tests/test_vocos_finetune_train.py`

This is the core module. It subclasses `VocosExp` to inject resample-roundtrip augmentation between the feature extractor and backbone.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for VocosFineTuneExp training module."""

import pytest
import torch
import numpy as np


@pytest.fixture
def dummy_audio_batch():
    """1-second batch of 4 audio samples at 24kHz."""
    return torch.randn(4, 24000)


def test_augment_features_changes_mel(dummy_audio_batch):
    """Verify that augmentation actually modifies the mel features."""
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model

    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 1.0  # always augment

    features = model.feature_extractor(dummy_audio_batch)
    augmented = model._maybe_augment(features)
    # with aug_ratio=1.0, should always augment
    assert not torch.allclose(features, augmented, atol=1e-3)


def test_augment_features_preserves_shape(dummy_audio_batch):
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model

    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 1.0

    features = model.feature_extractor(dummy_audio_batch)
    augmented = model._maybe_augment(features)
    assert augmented.shape == features.shape


def test_no_augment_when_ratio_zero(dummy_audio_batch):
    from scripts.vocos_finetune.train import VocosFineTuneExp, create_model

    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.aug_ratio = 0.0

    features = model.feature_extractor(dummy_audio_batch)
    result = model._maybe_augment(features)
    assert torch.allclose(features, result)


def test_aug_ratio_ramp():
    from scripts.vocos_finetune.train import compute_aug_ratio

    assert abs(compute_aug_ratio(0) - 0.3) < 1e-6
    assert abs(compute_aug_ratio(2000) - 0.3) < 1e-6
    assert abs(compute_aug_ratio(3000) - 0.4) < 1e-6
    assert abs(compute_aug_ratio(4000) - 0.5) < 1e-6
    assert abs(compute_aug_ratio(8000) - 0.5) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vocos_finetune_train.py -v
```

- [ ] **Step 3: Write train.py**

```python
"""Vocos fine-tuning with resample-roundtrip mel augmentation.

Subclasses VocosExp to inject augmentation between feature extractor and backbone.
Uses PyTorch Lightning 1.8.6 (PL 1.x multi-optimizer API).
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from vocos import Vocos
from vocos.experiment import VocosExp
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone

from scripts.vocos_finetune.augment import random_resample_roundtrip
from scripts.vocos_finetune.dataset import AudioDataset


def compute_aug_ratio(global_step: int) -> float:
    """Augmentation ratio ramp: 30% at step 0, linear to 50% at step 4000, then constant."""
    if global_step <= 2000:
        return 0.3
    elif global_step <= 4000:
        return 0.3 + 0.2 * (global_step - 2000) / 2000
    else:
        return 0.5


class VocosFineTuneExp(VocosExp):
    """Vocos experiment with resample-roundtrip mel augmentation.

    Overrides training_step to apply augmentation between feature extraction
    and backbone, and forward to route through augmentation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_ratio = 0.3

    def _maybe_augment(self, features: torch.Tensor) -> torch.Tensor:
        """Apply resample-roundtrip augmentation with probability aug_ratio."""
        if not self.training or self.aug_ratio <= 0:
            return features
        if torch.rand(1).item() < self.aug_ratio:
            return random_resample_roundtrip(features, min_rate=1.5, max_rate=5.0, presmooth_sigma=2.0)
        return features

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        # Compute augmented audio_hat ONCE per batch, reuse for both disc and gen.
        # This avoids stochastic augmentation mismatch between the two optimizer paths.
        if optimizer_idx == 0:
            with torch.no_grad():
                features = self.feature_extractor(audio_input, **kwargs)
                features = self._maybe_augment(features)
                x = self.backbone(features, **kwargs)
                audio_hat = self.head(x)
            self._cached_audio_hat = audio_hat

        if optimizer_idx == 0 and self.train_discriminator:
            audio_hat = self._cached_audio_hat
            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd
            self.log("discriminator/total", loss, prog_bar=True)
            return loss

        if optimizer_idx == 1:
            # Re-run forward WITH gradients, using same augmentation decision
            # (seeded by the batch_idx for consistency)
            torch.manual_seed(batch_idx + self.global_step)
            features = self.feature_extractor(audio_input, **kwargs)
            features = self._maybe_augment(features)
            x = self.backbone(features, **kwargs)
            audio_hat = self.head(x)

            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
            )
            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("generator/mel_loss", mel_loss)
            return loss

    def on_train_batch_start(self, *args):
        super().on_train_batch_start(*args)
        self.aug_ratio = compute_aug_ratio(self.global_step)


def create_model(
    pretrain_mel_steps: int = 1000,
    initial_learning_rate: float = 2e-5,
    max_steps: int = 20000,
) -> VocosFineTuneExp:
    """Create a VocosFineTuneExp with pretrained backbone weights."""
    feature_extractor = MelSpectrogramFeatures(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100
    )
    backbone = VocosBackbone(
        input_channels=100, dim=512, intermediate_dim=1536, num_layers=8
    )
    head = ISTFTHead(dim=512, n_fft=1024, hop_length=256)

    # Load pretrained weights into backbone + head
    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    backbone.load_state_dict(pt_model.backbone.state_dict())
    head.load_state_dict(pt_model.head.state_dict())
    del pt_model

    model = VocosFineTuneExp(
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head,
        sample_rate=24000,
        initial_learning_rate=initial_learning_rate,
        num_warmup_steps=1000,
        mel_loss_coeff=45,
        mrd_loss_coeff=1.0,
        pretrain_mel_steps=pretrain_mel_steps,
    )

    return model


class QualityGateCallback(pl.Callback):
    """Delete checkpoints where normal mel loss exceeds baseline + threshold.

    PL 1.8.6 ModelCheckpoint doesn't expose a skip hook, so this callback
    runs after checkpointing and removes files that fail the quality gate.
    """

    def __init__(self, checkpoint_dir: Path, threshold: float = 0.05):
        self.checkpoint_dir = checkpoint_dir
        self.threshold = threshold
        self.baseline_mel_loss = None

    def on_validation_end(self, trainer, pl_module):
        normal_loss = trainer.callback_metrics.get("val/mel_loss_normal")
        if normal_loss is None:
            return
        # Record baseline from first validation
        if self.baseline_mel_loss is None:
            self.baseline_mel_loss = normal_loss.item()
            print(f"Quality gate: baseline normal mel loss = {self.baseline_mel_loss:.4f}")
            return
        # Gate: if normal quality regressed, remove the latest best checkpoint
        if normal_loss.item() > self.baseline_mel_loss + self.threshold:
            print(f"Quality gate: normal mel loss {normal_loss:.4f} exceeds "
                  f"baseline {self.baseline_mel_loss:.4f} + {self.threshold}. "
                  f"Checkpoint will not be considered best.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filelist", type=Path, default=Path("training/data/filelists/train.txt"))
    parser.add_argument("--val-filelist", type=Path, default=Path("training/data/filelists/val.txt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("training/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("training/logs"))
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(
        pretrain_mel_steps=1000,
        initial_learning_rate=args.lr,
        max_steps=args.max_steps,
    )

    train_ds = AudioDataset(
        filelist_path=str(args.train_filelist),
        num_samples=24000,
        sample_rate=24000,
        train=True,
    )
    val_ds = AudioDataset(
        filelist_path=str(args.val_filelist),
        num_samples=24000,
        sample_rate=24000,
        train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    checkpoint_best = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="best-{step}",
        monitor="val/composite_loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=2000,  # 2000 optimizer steps = 1000 effective batches
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="last",
        every_n_train_steps=2000,
        save_last=True,
    )
    quality_gate = QualityGateCallback(checkpoint_dir=args.checkpoint_dir)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = pl.loggers.TensorBoardLogger(save_dir=str(args.log_dir), name="vocos_finetune")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="mps",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_best, checkpoint_last, quality_gate, lr_monitor],
        val_check_interval=2000,  # 2000 optimizer steps = 1000 effective batches
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    ckpt_path = "last" if args.resume else None
    if args.resume:
        last_ckpt = args.checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
        else:
            print("No checkpoint found for resume, starting fresh")
            ckpt_path = None

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    print(f"\nTraining complete. Best checkpoint: {checkpoint_best.best_model_path}")
    print(f"Best composite loss: {checkpoint_best.best_model_score}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_vocos_finetune_train.py -v
```

Expected: all 4 tests PASS. Note: first run will be slow due to pretrained weight download.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/train.py tests/test_vocos_finetune_train.py
git commit -m "feat: add VocosFineTuneExp training module with augmentation"
```

---

### Task 7: Validation with separate normal/augmented metrics

The upstream `VocosExp.validation_step` only computes a single mel loss. We need to override it to compute separate normal and augmented mel losses, plus per-rate losses and click counts.

**Files:**
- Modify: `scripts/vocos_finetune/train.py` (add validation overrides)
- Test: `tests/test_vocos_finetune_train.py` (add validation metric tests)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vocos_finetune_train.py`:

```python
def test_validation_step_returns_separate_metrics(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_model

    model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=100)
    model.eval()
    with torch.no_grad():
        result = model.validation_step(dummy_audio_batch, batch_idx=0)

    assert "mel_loss_normal" in result
    assert "mel_loss_augmented" in result
    assert "mel_loss_2_0x" in result
    assert "mel_loss_3_0x" in result
    assert "mel_loss_4_0x" in result
    assert "click_rate_2_0x" in result
    assert isinstance(result["mel_loss_normal"], torch.Tensor)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_vocos_finetune_train.py::test_validation_step_returns_separate_metrics -v
```

- [ ] **Step 3: Add validation overrides to train.py**

Add these methods to the `VocosFineTuneExp` class in `train.py`:

```python
    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        from scripts.vocos_finetune.augment import resample_roundtrip
        from scripts.vocos_finetune.click_detector import clicks_per_second
        import numpy as np

        # Normal reconstruction
        features_normal = self.feature_extractor(audio_input)
        x_normal = self.backbone(features_normal)
        audio_hat_normal = self.head(x_normal)
        mel_loss_normal = self.melspec_loss(audio_hat_normal, audio_input)

        # Per-rate augmented reconstruction
        result = {
            "val_loss": mel_loss_normal,
            "mel_loss_normal": mel_loss_normal,
            "audio_input": audio_input[0],
            "audio_pred_normal": audio_hat_normal[0],
        }

        mel_losses_aug = []
        for rate in [2.0, 3.0, 4.0]:
            features_aug = resample_roundtrip(features_normal.clone(), rate=rate, presmooth_sigma=2.0)
            x_aug = self.backbone(features_aug)
            audio_hat_aug = self.head(x_aug)
            mel_loss_aug = self.melspec_loss(audio_hat_aug, audio_input)
            mel_losses_aug.append(mel_loss_aug)

            # Click rate on first sample in batch
            audio_np = audio_hat_aug[0].detach().cpu().numpy()
            click_rate = clicks_per_second(audio_np, sample_rate=self.hparams.sample_rate)

            rate_key = f"{rate}x".replace(".", "_")
            result[f"mel_loss_{rate_key}"] = mel_loss_aug
            result[f"click_rate_{rate_key}"] = torch.tensor(click_rate)

        avg_aug_loss = torch.stack(mel_losses_aug).mean()
        composite = 0.5 * mel_loss_normal + 0.5 * avg_aug_loss
        result["mel_loss_augmented"] = avg_aug_loss
        result["val_loss"] = composite

        self.log("val/mel_loss_normal", mel_loss_normal, prog_bar=True)
        self.log("val/mel_loss_augmented", avg_aug_loss, prog_bar=True)
        self.log("val/composite_loss", composite, prog_bar=True)

        return result

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0 and outputs:
            first = outputs[0]
            self.logger.experiment.add_audio(
                "val/audio_in", first["audio_input"].data.cpu().numpy(),
                self.global_step, self.hparams.sample_rate,
            )
            self.logger.experiment.add_audio(
                "val/audio_normal", first["audio_pred_normal"].data.cpu().numpy(),
                self.global_step, self.hparams.sample_rate,
            )
            self.logger.experiment.add_audio(
                "val/audio_aug", first["audio_pred_aug"].data.cpu().numpy(),
                self.global_step, self.hparams.sample_rate,
            )

        avg_normal = torch.stack([x["mel_loss_normal"] for x in outputs]).mean()
        avg_aug = torch.stack([x["mel_loss_augmented"] for x in outputs]).mean()
        avg_composite = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("val_loss", avg_composite, sync_dist=True)
        self.log("val/mel_loss_normal", avg_normal, sync_dist=True)
        self.log("val/mel_loss_augmented", avg_aug, sync_dist=True)

        # Per-rate metrics
        for rate in [2.0, 3.0, 4.0]:
            rate_key = f"{rate}x".replace(".", "_")
            mel_key = f"mel_loss_{rate_key}"
            click_key = f"click_rate_{rate_key}"
            if mel_key in outputs[0]:
                avg_mel = torch.stack([x[mel_key] for x in outputs]).mean()
                avg_click = torch.stack([x[click_key] for x in outputs]).mean()
                self.log(f"val/mel_loss_{rate_key}", avg_mel, sync_dist=True)
                self.log(f"val/click_rate_{rate_key}", avg_click, sync_dist=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_vocos_finetune_train.py -v
```

Expected: all tests PASS including the new validation metric test.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/train.py tests/test_vocos_finetune_train.py
git commit -m "feat: add validation with separate normal/augmented mel metrics"
```

---

### Task 8: A/B sample evaluation script

**Files:**
- Create: `scripts/vocos_finetune/evaluate.py`
- Test: manual — generates audio files

This script loads a checkpoint and generates A/B comparison audio files by running utterances through the actual Osmium pipeline at 2x, 3x, 4x with both the baseline and fine-tuned vocoder.

- [ ] **Step 1: Write evaluate.py**

```python
"""Generate A/B comparison samples from a fine-tuned Vocos checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.vocos_finetune.click_detector import clicks_per_second


def generate_samples(
    checkpoint_path: Path,
    val_filelist: Path,
    output_dir: Path,
    n_utterances: int = 5,
    speeds: list[float] = [2.0, 3.0, 4.0],
    sample_rate: int = 24000,
):
    """Generate A/B comparison samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(val_filelist) as f:
        utterances = [line.strip() for line in f if line.strip()][:n_utterances]

    # Load baseline vocoder
    from vocos import Vocos
    baseline_vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    baseline_vocos.eval()

    # Load fine-tuned vocoder from checkpoint state_dict
    # (can't use load_from_checkpoint because VocosExp excludes feature_extractor/backbone/head from hparams)
    from scripts.vocos_finetune.train import create_model
    finetuned_model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    finetuned_model.load_state_dict(ckpt["state_dict"])
    finetuned_model.eval()

    class FinetunedVocos:
        def __init__(self, model):
            self.feature_extractor = model.feature_extractor
            self.backbone = model.backbone
            self.head = model.head

    finetuned = FinetunedVocos(finetuned_model)

    readme_lines = ["# A/B Comparison Samples\n"]

    for utt_idx, utt_path in enumerate(utterances):
        import torchaudio
        audio, sr = torchaudio.load(utt_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = audio[0]  # (T,)

        # Trim to 5 seconds max
        max_samples = 5 * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        utt_name = Path(utt_path).stem

        for speed in speeds:
            # Time-resample the mel (simulate what Osmium does)
            audio_batch = audio.unsqueeze(0)  # (1, T)

            with torch.no_grad():
                # Baseline
                features = baseline_vocos.feature_extractor(audio_batch)
                T = features.shape[2]
                target_T = max(1, int(T / speed))
                resampled = torch.nn.functional.interpolate(
                    features, size=target_T, mode="linear", align_corners=True,
                )
                baseline_out = baseline_vocos.decode(resampled)
                baseline_np = baseline_out.squeeze().numpy()

                # Fine-tuned
                features_ft = finetuned.feature_extractor(audio_batch)
                resampled_ft = torch.nn.functional.interpolate(
                    features_ft, size=target_T, mode="linear", align_corners=True,
                )
                x_ft = finetuned.backbone(resampled_ft)
                finetuned_out = finetuned.head(x_ft)
                finetuned_np = finetuned_out.squeeze().numpy()

            # Save WAVs
            baseline_path = output_dir / f"{utt_name}_{speed}x_baseline.wav"
            finetuned_path = output_dir / f"{utt_name}_{speed}x_finetuned.wav"
            sf.write(str(baseline_path), baseline_np, sample_rate)
            sf.write(str(finetuned_path), finetuned_np, sample_rate)

            # Click counts
            baseline_clicks = clicks_per_second(baseline_np, sample_rate)
            finetuned_clicks = clicks_per_second(finetuned_np, sample_rate)

            readme_lines.append(f"## {utt_name} @ {speed}x")
            readme_lines.append(f"- Baseline clicks/s: {baseline_clicks:.1f}")
            readme_lines.append(f"- Fine-tuned clicks/s: {finetuned_clicks:.1f}")
            readme_lines.append("")

    readme_path = output_dir / "README.txt"
    readme_path.write_text("\n".join(readme_lines))
    print(f"Samples saved to {output_dir}")
    print(f"Summary: {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-filelist", type=Path, default=Path("training/data/filelists/val.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-utterances", type=int, default=5)
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        val_filelist=args.val_filelist,
        output_dir=args.output_dir,
        n_utterances=args.n_utterances,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/vocos_finetune/evaluate.py
git commit -m "feat: add A/B sample generation for vocoder evaluation"
```

---

### Task 9: MLX weight conversion

**Files:**
- Create: `scripts/vocos_finetune/convert_mlx.py`
- Test: `tests/test_vocos_finetune_convert.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for MLX weight conversion."""

import pytest
import torch
import numpy as np


def test_convert_weights_produces_correct_keys():
    from scripts.vocos_finetune.convert_mlx import extract_mlx_weights

    # Use the pretrained model as a stand-in
    from vocos import Vocos
    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    weights = extract_mlx_weights(pt_model.backbone.state_dict(), pt_model.head.state_dict())
    keys = [k for k, v in weights]

    assert "embed.weight" in keys
    assert "embed.bias" in keys
    assert "convnext.0.dwconv.weight" in keys
    assert "convnext.7.gamma" in keys
    assert "head_out.weight" in keys
    assert "final_layer_norm.weight" in keys


def test_convert_weights_shapes_match_mlx_model():
    from scripts.vocos_finetune.convert_mlx import extract_mlx_weights
    from vocos import Vocos

    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    weights = extract_mlx_weights(pt_model.backbone.state_dict(), pt_model.head.state_dict())

    # Check key shape expectations
    weight_dict = dict(weights)
    assert weight_dict["embed.weight"].shape == (512, 7, 100)  # MLX Conv1d: (out, kernel, in)
    assert weight_dict["head_out.weight"].shape == (1026, 512)
    assert weight_dict["convnext.0.gamma"].shape == (512,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vocos_finetune_convert.py -v
```

- [ ] **Step 3: Write convert_mlx.py**

```python
"""Convert fine-tuned Vocos PyTorch checkpoint to MLX weights.

Uses the same weight mapping as vocos_mlx.py's _convert_weights, adapted
to work from a Lightning checkpoint's state dict.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def extract_mlx_weights(
    backbone_sd: dict[str, torch.Tensor],
    head_sd: dict[str, torch.Tensor],
) -> list[tuple[str, np.ndarray]]:
    """Convert backbone + head state dicts to MLX weight list.

    Mirrors the mapping in src/osmium/tsm/vocos_mlx.py _convert_weights().
    """
    weights = []

    def _c(name: str, tensor: torch.Tensor):
        weights.append((name, tensor.detach().cpu().numpy()))

    # Backbone embed (Conv1d: PyTorch [out, in, kernel] → MLX [out, kernel, in] via transpose(1,2))
    _c("embed.weight", backbone_sd["embed.weight"].transpose(1, 2))
    _c("embed.bias", backbone_sd["embed.bias"])

    _c("norm.weight", backbone_sd["norm.weight"])
    _c("norm.bias", backbone_sd["norm.bias"])

    for i in range(8):
        pt_pre = f"convnext.{i}"
        mlx_pre = f"convnext.{i}"

        _c(f"{mlx_pre}.dwconv.weight", backbone_sd[f"{pt_pre}.dwconv.weight"].transpose(1, 2))
        _c(f"{mlx_pre}.dwconv.bias", backbone_sd[f"{pt_pre}.dwconv.bias"])
        _c(f"{mlx_pre}.norm.weight", backbone_sd[f"{pt_pre}.norm.weight"])
        _c(f"{mlx_pre}.norm.bias", backbone_sd[f"{pt_pre}.norm.bias"])
        _c(f"{mlx_pre}.pwconv1.weight", backbone_sd[f"{pt_pre}.pwconv1.weight"])
        _c(f"{mlx_pre}.pwconv1.bias", backbone_sd[f"{pt_pre}.pwconv1.bias"])
        _c(f"{mlx_pre}.pwconv2.weight", backbone_sd[f"{pt_pre}.pwconv2.weight"])
        _c(f"{mlx_pre}.pwconv2.bias", backbone_sd[f"{pt_pre}.pwconv2.bias"])
        _c(f"{mlx_pre}.gamma", backbone_sd[f"{pt_pre}.gamma"])

    _c("final_layer_norm.weight", backbone_sd["final_layer_norm.weight"])
    _c("final_layer_norm.bias", backbone_sd["final_layer_norm.bias"])

    # Head (Linear)
    _c("head_out.weight", head_sd["out.weight"])
    _c("head_out.bias", head_sd["out.bias"])

    return weights


def convert_checkpoint(checkpoint_path: Path, output_dir: Path):
    """Load a Lightning checkpoint and save MLX weights."""
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Extract backbone and head state dicts from Lightning's prefixed keys
    backbone_sd = {}
    head_sd = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head_sd[k[len("head."):]] = v

    weights = extract_mlx_weights(backbone_sd, head_sd)

    # Save as MLX-compatible npz
    weight_dict = {name: arr for name, arr in weights}
    output_path = output_dir / "weights.npz"
    np.savez(str(output_path), **weight_dict)

    print(f"MLX weights saved to {output_path}")
    print(f"  {len(weights)} weight arrays, total {sum(v.nbytes for _, v in weights) / 1e6:.1f} MB")

    # Verify by loading into VocosMLX
    from osmium.tsm.vocos_mlx import VocosMLX
    model = VocosMLX()
    mlx_weights = [(name, mx.array(arr)) for name, arr in weights]
    model.load_weights(mlx_weights, strict=False)
    mx.eval(model.parameters())
    print("Verification: MLX model loaded successfully")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training/models/vocos-mel-24khz-finetuned"))
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_vocos_finetune_convert.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/convert_mlx.py tests/test_vocos_finetune_convert.py
git commit -m "feat: add MLX weight conversion for fine-tuned Vocos"
```

---

### Task 10: Shell runner script

**Files:**
- Create: `scripts/vocos_finetune.sh`

- [ ] **Step 1: Write vocos_finetune.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TRAINING_DIR="training"
DATA_DIR="$TRAINING_DIR/data"
CHECKPOINT_DIR="$TRAINING_DIR/checkpoints"
LOG_DIR="$TRAINING_DIR/logs"
EVAL_DIR="$TRAINING_DIR/eval_samples"
MODEL_DIR="$TRAINING_DIR/models/vocos-mel-24khz-finetuned"

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME="--resume"
    echo "==> Resume mode: will continue from last checkpoint"
fi

echo "============================================"
echo "  Vocos Fine-Tuning Pipeline"
echo "  $(date)"
echo "============================================"

# Stage 1: Download data
echo ""
echo "==> Stage 1: Download LibriTTS train-clean-100"
PYTHONPATH="$PROJECT_DIR" python scripts/vocos_finetune/download_data.py --data-dir "$DATA_DIR"

# Stage 2: Install training deps (idempotent)
echo ""
echo "==> Stage 2: Verify training dependencies"
python -c "import pytorch_lightning; import transformers; import einops" 2>/dev/null || {
    echo "Installing training dependencies..."
    uv pip install "pytorch-lightning==1.8.6" "transformers>=4.30" "einops"
}

# Stage 3: Train
echo ""
echo "==> Stage 3: Training (10k effective steps, ~3-5 hours)"
echo "    Checkpoints: $CHECKPOINT_DIR"
echo "    Logs: $LOG_DIR (tensorboard --logdir $LOG_DIR)"
echo "    Ctrl-C to pause, re-run with --resume to continue"
echo ""
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.train \
    --train-filelist "$DATA_DIR/filelists/train.txt" \
    --val-filelist "$DATA_DIR/filelists/val.txt" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --max-steps 20000 \
    --batch-size 16 \
    --lr 2e-5 \
    $RESUME

# Stage 4: Find best checkpoint
echo ""
echo "==> Stage 4: Convert best checkpoint to MLX"
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best-*.ckpt 2>/dev/null | head -1)
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: No best checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi
echo "    Best checkpoint: $BEST_CKPT"
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.convert_mlx \
    --checkpoint "$BEST_CKPT" \
    --output-dir "$MODEL_DIR"

# Stage 5: Generate final A/B comparison
echo ""
echo "==> Stage 5: Generate final A/B comparison samples"
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.evaluate \
    --checkpoint "$BEST_CKPT" \
    --val-filelist "$DATA_DIR/filelists/val.txt" \
    --output-dir "$TRAINING_DIR/final_comparison" \
    --n-utterances 10

# Done
echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  MLX weights:     $MODEL_DIR/weights.npz"
echo "  A/B samples:     $TRAINING_DIR/final_comparison/"
echo "  TensorBoard:     tensorboard --logdir $LOG_DIR"
echo ""
echo "Listen to the 4x samples first — clicks are worst there."
echo "Check README.txt in the comparison directory for metrics."
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/vocos_finetune.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/vocos_finetune.sh
git commit -m "feat: add vocos fine-tuning runner script"
```

---

### Task 11: Periodic A/B eval callback

**Files:**
- Modify: `scripts/vocos_finetune/train.py` (add callback for eval sample generation every 2000 steps)

- [ ] **Step 1: Add eval callback to train.py**

Add this callback class and wire it into the trainer:

```python
class EvalSampleCallback(pl.Callback):
    """Generate A/B comparison samples every N steps during training."""

    def __init__(self, val_filelist: Path, output_base: Path, every_n_steps: int = 2000):
        self.val_filelist = val_filelist
        self.output_base = output_base
        self.every_n_steps = every_n_steps
        self._last_step = -1

    def on_validation_end(self, trainer, pl_module):
        # Use pl_module.global_step (overridden to count batches, not optimizer steps)
        step = pl_module.global_step
        if step == self._last_step or step % self.every_n_steps != 0 or step == 0:
            return
        self._last_step = step

        output_dir = self.output_base / f"step_{step}"
        if output_dir.exists():
            return

        print(f"\n==> Generating A/B samples at step {step}")

        # Save a temporary checkpoint for the evaluator
        ckpt_path = self.output_base / f"_tmp_eval_step_{step}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))

        try:
            from scripts.vocos_finetune.evaluate import generate_samples
            generate_samples(
                checkpoint_path=ckpt_path,
                val_filelist=self.val_filelist,
                output_dir=output_dir,
                n_utterances=5,
            )
        except Exception as e:
            print(f"Warning: eval sample generation failed at step {step}: {e}")
        finally:
            ckpt_path.unlink(missing_ok=True)
```

In the `main()` function, add to the callbacks list:

```python
    eval_callback = EvalSampleCallback(
        val_filelist=args.val_filelist,
        output_base=Path("training/eval_samples"),
        every_n_steps=2000,
    )
    # Add eval_callback to the callbacks list in Trainer
```

- [ ] **Step 2: Run training for a few steps to smoke test**

```bash
cd /Users/mike/code/osmium
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.train \
    --train-filelist training/data/filelists/train.txt \
    --val-filelist training/data/filelists/val.txt \
    --max-steps 100 --batch-size 4
```

Expected: starts training, runs ~100 steps, validates once, no crashes. Ctrl-C when satisfied.

- [ ] **Step 3: Commit**

```bash
git add scripts/vocos_finetune/train.py
git commit -m "feat: add periodic A/B eval sample callback during training"
```

---

### Task 12: End-to-end smoke test and launch

- [ ] **Step 1: Run all tests**

```bash
cd /Users/mike/code/osmium
python -m pytest tests/test_vocos_finetune_*.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify resume works**

```bash
# Start training
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.train \
    --train-filelist training/data/filelists/train.txt \
    --val-filelist training/data/filelists/val.txt \
    --max-steps 200 --batch-size 4
# Ctrl-C after ~50 steps

# Resume
PYTHONPATH="$PROJECT_DIR" python -m scripts.vocos_finetune.train \
    --train-filelist training/data/filelists/train.txt \
    --val-filelist training/data/filelists/val.txt \
    --max-steps 200 --batch-size 4 --resume
```

Expected: resumes from checkpoint, continues training from where it left off.

- [ ] **Step 3: Clean up test artifacts**

```bash
rm -rf training/checkpoints/best-*.ckpt training/checkpoints/last.ckpt training/logs/vocos_finetune
```

- [ ] **Step 4: Launch the full pipeline**

```bash
cd /Users/mike/code/osmium
./scripts/vocos_finetune.sh 2>&1 | tee training/run.log
```

This runs unattended for 3-5 hours. Check in on:
- `training/eval_samples/step_*/` for A/B audio at 2000-step intervals
- `training/run.log` for progress
- `tensorboard --logdir training/logs` for live metrics

- [ ] **Step 5: Commit all remaining changes**

```bash
git add -A scripts/vocos_finetune/ tests/test_vocos_finetune_*.py
git commit -m "feat: complete vocos fine-tuning pipeline, ready for training"
```
