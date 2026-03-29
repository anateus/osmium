# Vocos Phase Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an instantaneous frequency deviation (IFD) loss to Vocos fine-tuning that penalizes phase discontinuities on augmented mel, training head-only to reduce clicks in time-stretched audio.

**Architecture:** New `VocosPhaseRegExp` class with single-optimizer head-only training, direct phase extraction via `head.out()` + `head.istft()`, and magnitude-weighted IFD loss on augmented batches only. Reuses existing dataset, augmentation, click detection, and MLX conversion infrastructure.

**Tech Stack:** PyTorch, PyTorch Lightning 1.8.6, Vocos, MLX, NumPy, SciPy

**Spec:** `docs/specs/2026-03-29-vocos-phase-regularization-design.md`

---

### Task 1: IFD Phase Loss Module

**Files:**
- Create: `scripts/vocos_finetune/phase_loss.py`
- Create: `tests/test_vocos_finetune_phase_loss.py`

- [ ] **Step 1: Write tests for IFD loss**

```python
import torch
import math
import pytest

from scripts.vocos_finetune.phase_loss import InstantaneousFrequencyDeviationLoss


@pytest.fixture
def ifd_loss():
    return InstantaneousFrequencyDeviationLoss(n_fft=1024, hop_length=256)


def test_zero_loss_for_expected_phase_advance(ifd_loss):
    """Phase that advances exactly at bin-center frequency should give near-zero loss."""
    B, N, T = 2, 513, 20
    n_fft, hop = 1024, 256
    k = torch.arange(N).float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    expected_advance = 2 * math.pi * k * hop / n_fft
    t = torch.arange(T).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    phase = expected_advance * t  # (1, N, T) -> broadcast to (B, N, T)
    phase = phase.expand(B, N, T)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() < 1e-5


def test_nonzero_loss_for_random_phase(ifd_loss):
    """Random phase should give nonzero loss."""
    B, N, T = 2, 513, 20
    phase = torch.randn(B, N, T) * math.pi
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() > 0.01


def test_dc_and_nyquist_excluded(ifd_loss):
    """DC (k=0) and Nyquist (k=512) should not contribute to loss."""
    B, N, T = 1, 513, 20
    phase = torch.zeros(B, N, T)
    phase[:, 0, :] = torch.randn(1, T) * 100  # wild DC phase
    phase[:, -1, :] = torch.randn(1, T) * 100  # wild Nyquist phase
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.item() < 1e-5


def test_magnitude_weighting_suppresses_silent_bins(ifd_loss):
    """Bins with zero magnitude should not contribute to loss."""
    B, N, T = 1, 513, 20
    phase = torch.randn(B, N, T) * math.pi
    mag_silent = torch.zeros(B, N, T)
    mag_silent[:, 100, :] = 1.0  # only one bin has energy
    loss_silent = ifd_loss(phase, mag_silent)

    mag_full = torch.ones(B, N, T)
    loss_full = ifd_loss(phase, mag_full)
    assert loss_silent < loss_full


def test_loss_is_differentiable(ifd_loss):
    """Loss should be differentiable w.r.t. phase."""
    B, N, T = 1, 513, 10
    phase = torch.randn(B, N, T, requires_grad=True)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    loss.backward()
    assert phase.grad is not None
    assert not torch.isnan(phase.grad).any()


def test_output_is_scalar(ifd_loss):
    """Loss should return a scalar tensor."""
    B, N, T = 2, 513, 15
    phase = torch.randn(B, N, T)
    mag = torch.ones(B, N, T)
    loss = ifd_loss(phase, mag)
    assert loss.dim() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vocos_finetune_phase_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.vocos_finetune.phase_loss'`

- [ ] **Step 3: Implement IFD loss module**

```python
import math

import torch
from torch import nn


class InstantaneousFrequencyDeviationLoss(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_bins = n_fft // 2 + 1
        k = torch.arange(1, n_bins - 1).float()
        expected = 2 * math.pi * k * hop_length / n_fft
        self.register_buffer("expected_advance", expected)

    def forward(self, phase: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        phase_inner = phase[:, 1:-1, :]
        mag_inner = mag[:, 1:-1, :]

        actual_advance = phase_inner[:, :, 1:] - phase_inner[:, :, :-1]
        expected = self.expected_advance[None, :, None]
        raw_deviation = actual_advance - expected
        deviation = torch.atan2(torch.sin(raw_deviation), torch.cos(raw_deviation))

        mag_for_weight = mag_inner[:, :, 1:]
        mag_max = mag.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        mag_weight = mag_for_weight / mag_max

        weighted_dev_sq = mag_weight * deviation.square()
        return weighted_dev_sq.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vocos_finetune_phase_loss.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/phase_loss.py tests/test_vocos_finetune_phase_loss.py
git commit -m "feat: add instantaneous frequency deviation (IFD) phase loss module"
```

---

### Task 2: VocosPhaseRegExp Training Class

**Files:**
- Modify: `scripts/vocos_finetune/train.py`
- Create: `tests/test_vocos_finetune_phase_reg.py`

- [ ] **Step 1: Write tests for VocosPhaseRegExp**

```python
import torch
import pytest


@pytest.fixture
def dummy_audio_batch():
    return torch.randn(4, 24000)


def test_phase_reg_model_creates(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    assert model is not None


def test_backbone_is_frozen():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    for param in model.backbone.parameters():
        assert not param.requires_grad


def test_head_is_trainable():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    for param in model.head.parameters():
        assert param.requires_grad


def test_single_optimizer():
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.trainer = type("MockTrainer", (), {"max_steps": 10000})()
    opts = model.configure_optimizers()
    if isinstance(opts, tuple):
        optimizers = opts[0]
    else:
        optimizers = opts
    assert len(optimizers) == 1


def test_forward_produces_audio(dummy_audio_batch):
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.eval()
    with torch.no_grad():
        audio = model(dummy_audio_batch)
    assert audio.shape[0] == 4
    assert audio.dim() == 2


def test_phase_extraction_matches_forward(dummy_audio_batch):
    """Direct phase extraction should produce same audio as head.forward()."""
    from scripts.vocos_finetune.train import create_phase_reg_model
    model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=100)
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(dummy_audio_batch)
        backbone_out = model.backbone(features)

        audio_normal = model.head(backbone_out)

        x_proj = model.head.out(backbone_out).transpose(1, 2)
        mag_raw, phase = x_proj.chunk(2, dim=1)
        mag = torch.exp(mag_raw).clip(max=1e2)
        S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio_manual = model.head.istft(S)

    assert torch.allclose(audio_normal, audio_manual, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vocos_finetune_phase_reg.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_phase_reg_model'`

- [ ] **Step 3: Implement VocosPhaseRegExp and create_phase_reg_model**

Add to `scripts/vocos_finetune/train.py`:

```python
from scripts.vocos_finetune.phase_loss import InstantaneousFrequencyDeviationLoss


class VocosPhaseRegExp(VocosExp):
    def __init__(self, *args, phase_coeff: float = 0.05, aug_ratio: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_coeff = phase_coeff
        self.aug_ratio = aug_ratio
        self.ifd_loss = InstantaneousFrequencyDeviationLoss(
            n_fft=1024, hop_length=256,
        )
        self.backbone.requires_grad_(False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.initial_learning_rate,
            betas=(0.8, 0.9),
        )
        max_steps = self.trainer.max_steps
        scheduler = transformers.get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=0, num_training_steps=max_steps,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    def _forward_extract_phase(self, audio_input):
        features = self.feature_extractor(audio_input)

        is_augmented = self.training and random.random() < self.aug_ratio
        if is_augmented:
            features = random_resample_roundtrip(
                features, min_rate=1.5, max_rate=5.0, presmooth_sigma=2.0,
            )

        backbone_out = self.backbone(features)
        T = audio_input.shape[-1]

        x_proj = self.head.out(backbone_out).transpose(1, 2)
        mag_raw, phase = x_proj.chunk(2, dim=1)
        mag = torch.exp(mag_raw).clip(max=1e2)
        S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio = self.head.istft(S)[..., :T]

        return audio, phase, mag, is_augmented

    def training_step(self, batch, batch_idx, optimizer_idx=0, **kwargs):
        audio_input = batch
        audio_hat, phase, mag, is_augmented = self._forward_extract_phase(audio_input)

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss = self.mel_loss_coeff * mel_loss

        self.log("generator/mel_loss", mel_loss, prog_bar=True)

        if is_augmented:
            phase_loss = self.ifd_loss(phase, mag)
            loss = loss + self.phase_coeff * phase_loss
            self.log("generator/phase_ifd_loss", phase_loss, prog_bar=True)

        self.log("generator/total_loss", loss, prog_bar=True)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            try:
                self.logger.experiment.add_audio(
                    "train/audio_in", audio_input[0].data.cpu(),
                    self.global_step, self.hparams.sample_rate,
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", audio_hat[0].data.cpu(),
                    self.global_step, self.hparams.sample_rate,
                )
            except Exception:
                pass

        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        from scripts.vocos_finetune.augment import resample_roundtrip
        from scripts.vocos_finetune.click_detector import clicks_per_second
        import numpy as np

        T = audio_input.shape[-1]
        features = self.feature_extractor(audio_input)
        backbone_out = self.backbone(features)
        audio_hat_normal = self.head(backbone_out)[..., :T]
        mel_loss_normal = self.melspec_loss(audio_hat_normal, audio_input)

        result = {
            "val_loss": mel_loss_normal,
            "mel_loss_normal": mel_loss_normal,
        }

        mel_losses_aug = []
        for rate in [2.0, 3.0, 4.0]:
            features_aug = resample_roundtrip(features.clone(), rate=rate, presmooth_sigma=2.0)
            backbone_aug = self.backbone(features_aug)

            x_proj = self.head.out(backbone_aug).transpose(1, 2)
            mag_raw, phase = x_proj.chunk(2, dim=1)
            mag = torch.exp(mag_raw).clip(max=1e2)
            S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
            audio_hat_aug = self.head.istft(S)[..., :T]

            mel_loss_aug = self.melspec_loss(audio_hat_aug, audio_input)
            phase_loss_aug = self.ifd_loss(phase, mag)
            mel_losses_aug.append(mel_loss_aug)

            audio_np = audio_hat_aug[0].detach().cpu().numpy()
            click_rate = clicks_per_second(audio_np, sample_rate=self.hparams.sample_rate)

            rate_key = f"{rate}x".replace(".", "_")
            result[f"mel_loss_{rate_key}"] = mel_loss_aug
            result[f"phase_ifd_{rate_key}"] = phase_loss_aug
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
        if not outputs:
            return
        avg_normal = torch.stack([x["mel_loss_normal"] for x in outputs]).mean()
        avg_aug = torch.stack([x["mel_loss_augmented"] for x in outputs]).mean()
        avg_composite = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log("val_loss", avg_composite, sync_dist=True)
        self.log("val/mel_loss_normal", avg_normal, sync_dist=True)
        self.log("val/mel_loss_augmented", avg_aug, sync_dist=True)

        for rate in [2.0, 3.0, 4.0]:
            rate_key = f"{rate}x".replace(".", "_")
            for prefix in ["mel_loss", "phase_ifd", "click_rate"]:
                key = f"{prefix}_{rate_key}"
                if key in outputs[0]:
                    avg = torch.stack([x[key] for x in outputs]).mean()
                    self.log(f"val/{key}", avg, sync_dist=True)

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        grad = self.head.out.weight.grad
        if grad is not None:
            self.log("grad_norm/head_out_weight", grad.norm().item())


def create_phase_reg_model(
    phase_coeff: float = 0.05,
    initial_learning_rate: float = 1e-5,
    max_steps: int = 5000,
) -> VocosPhaseRegExp:
    feature_extractor = MelSpectrogramFeatures(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100
    )
    backbone = VocosBackbone(
        input_channels=100, dim=512, intermediate_dim=1536, num_layers=8
    )
    head = ISTFTHead(dim=512, n_fft=1024, hop_length=256)

    pretrained = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    backbone.load_state_dict(pretrained.backbone.state_dict())
    head.load_state_dict(pretrained.head.state_dict())

    model = VocosPhaseRegExp(
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head,
        sample_rate=24000,
        initial_learning_rate=initial_learning_rate,
        pretrain_mel_steps=999999,
        phase_coeff=phase_coeff,
    )
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vocos_finetune_phase_reg.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/train.py tests/test_vocos_finetune_phase_reg.py
git commit -m "feat: add VocosPhaseRegExp with IFD phase loss and head-only training"
```

---

### Task 3: Enhanced Click Detector

**Files:**
- Modify: `scripts/vocos_finetune/click_detector.py`
- Modify: `tests/test_vocos_finetune_click.py`

- [ ] **Step 1: Write tests for spectral transient detection**

Add to `tests/test_vocos_finetune_click.py`:

```python
import numpy as np
from scripts.vocos_finetune.click_detector import spectral_transient_clicks


def test_spectral_transient_detects_impulse():
    """A single-sample impulse should register as a transient click."""
    sr = 24000
    audio = np.zeros(sr, dtype=np.float32)  # 1 second
    audio[sr // 2] = 1.0  # impulse at 0.5s
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks >= 1


def test_spectral_transient_silent_audio():
    """Silent audio should have zero transient clicks."""
    sr = 24000
    audio = np.zeros(sr, dtype=np.float32)
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks == 0


def test_spectral_transient_smooth_sine():
    """A smooth sine wave should have zero transient clicks."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    clicks = spectral_transient_clicks(audio, sample_rate=sr)
    assert clicks == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vocos_finetune_click.py::test_spectral_transient_detects_impulse -v`
Expected: FAIL with `ImportError: cannot import name 'spectral_transient_clicks'`

- [ ] **Step 3: Implement spectral transient detection and integrate into clicks_per_second**

Add to `scripts/vocos_finetune/click_detector.py`:

```python
def spectral_transient_clicks(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: float = 2.0,
    median_window_ms: float = 50.0,
    threshold: float = 4.0,
    lowcut_hz: float = 300.0,
) -> int:
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    n_fft = frame_samples
    n_frames = len(audio) // frame_samples
    if n_frames < 5:
        return 0

    trimmed = audio[: n_frames * frame_samples]
    frames = trimmed.reshape(n_frames, frame_samples)
    spec = np.abs(np.fft.rfft(frames, axis=1))

    freq_bins = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate)
    low_mask = freq_bins <= lowcut_hz

    broadband_energy = np.mean(spec ** 2, axis=1)
    low_energy = np.mean(spec[:, low_mask] ** 2, axis=1) if low_mask.any() else broadband_energy

    median_frames = max(3, int(median_window_ms / frame_ms))
    if median_frames % 2 == 0:
        median_frames += 1
    half = median_frames // 2

    clicks = 0
    for i in range(half, n_frames - half):
        bb_med = np.median(broadband_energy[i - half : i + half + 1])
        low_med = np.median(low_energy[i - half : i + half + 1])
        bb_spike = bb_med > 0 and broadband_energy[i] > threshold * bb_med
        low_spike = low_med > 0 and low_energy[i] > threshold * low_med
        if bb_spike or low_spike:
            clicks += 1
    return clicks
```

Also modify `clicks_per_second` to combine both detection methods:

```python
def clicks_per_second(audio: np.ndarray, sample_rate: int = 24000, **kwargs) -> float:
    duration = len(audio) / sample_rate
    if duration < 0.01:
        return 0.0
    amplitude_clicks = count_clicks(audio, sample_rate=sample_rate, **kwargs)
    spectral_clicks = spectral_transient_clicks(audio, sample_rate=sample_rate)
    return max(amplitude_clicks, spectral_clicks) / duration
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vocos_finetune_click.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add scripts/vocos_finetune/click_detector.py tests/test_vocos_finetune_click.py
git commit -m "feat: add spectral transient click detection, integrate into clicks_per_second"
```

---

### Task 4: Fix evaluate.py for Phase Reg Checkpoints

**Files:**
- Modify: `scripts/vocos_finetune/evaluate.py`

The existing `load_finetuned_model` calls `create_model()` which creates a `VocosFineTuneExp`. Phase reg checkpoints contain `ifd_loss.expected_advance` and different hparams. Also, `EvalSampleCallback` calls `generate_samples` which uses this loader. Both will crash with phase reg checkpoints.

- [ ] **Step 1: Update evaluate.py to support both model types**

Modify `scripts/vocos_finetune/evaluate.py` `load_finetuned_model`:

```python
def load_finetuned_model(checkpoint_path: Path, model_type: str = "finetune"):
    if model_type == "phase_reg":
        from scripts.vocos_finetune.train import create_phase_reg_model
        model = create_phase_reg_model(phase_coeff=0.05, initial_learning_rate=1e-5, max_steps=1)
    else:
        from scripts.vocos_finetune.train import create_model
        model = create_model(pretrain_mel_steps=0, initial_learning_rate=1e-4, max_steps=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model
```

Add `--model-type` to `evaluate.py` CLI:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-filelist", type=Path, default=Path("training/data/filelists/val.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-utterances", type=int, default=5)
    parser.add_argument("--model-type", choices=["finetune", "phase_reg"], default="finetune")
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        val_filelist=args.val_filelist,
        output_dir=args.output_dir,
        n_utterances=args.n_utterances,
        model_type=args.model_type,
    )
```

Thread `model_type` through `generate_samples` to `load_finetuned_model`.

- [ ] **Step 2: Update EvalSampleCallback to accept model_type**

In `scripts/vocos_finetune/train.py`, modify `EvalSampleCallback.__init__` to accept `model_type` parameter and pass it through to `generate_samples`:

```python
class EvalSampleCallback(pl.Callback):
    def __init__(self, val_filelist, output_base, every_n_steps=2000, model_type="finetune"):
        self.val_filelist = val_filelist
        self.output_base = Path(output_base)
        self.every_n_steps = every_n_steps
        self.model_type = model_type
        self._last_step = -1
```

And in `on_validation_end`, pass `model_type=self.model_type` to `generate_samples`.

- [ ] **Step 3: Verify imports work**

Run: `python -c "from scripts.vocos_finetune.evaluate import load_finetuned_model; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/vocos_finetune/evaluate.py scripts/vocos_finetune/train.py
git commit -m "fix: evaluate.py supports phase_reg model type for checkpoint loading"
```

---

### Task 5: Phase Reg CLI Entry Point (was Task 4)

**Files:**
- Modify: `scripts/vocos_finetune/train.py`

- [ ] **Step 1: Add `main_phase_reg()` entry point**

Add to `scripts/vocos_finetune/train.py` after the existing `main()`:

```python
def main_phase_reg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filelist", required=True)
    parser.add_argument("--val-filelist", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/vocos_phase_reg")
    parser.add_argument("--log-dir", default="logs/vocos_phase_reg")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--phase-coeff", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from scripts.vocos_finetune.dataset import AudioDataset
    from torch.utils.data import DataLoader

    model = create_phase_reg_model(
        phase_coeff=args.phase_coeff,
        initial_learning_rate=args.lr,
        max_steps=args.max_steps,
    )

    train_dataset = AudioDataset(
        filelist_path=args.train_filelist, num_samples=24000, sample_rate=24000, train=True
    )
    val_dataset = AudioDataset(
        filelist_path=args.val_filelist, num_samples=24000, sample_rate=24000, train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val/composite_loss",
        save_top_k=3,
        every_n_train_steps=2000,
        save_last=True,
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name="vocos_phase_reg")

    resume_path = None
    if args.resume:
        last_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_path = last_ckpt

    eval_callback = EvalSampleCallback(
        val_filelist=args.val_filelist,
        output_base=Path("training/eval_samples_phase_reg"),
        model_type="phase_reg",
    )

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_steps=args.max_steps,
        val_check_interval=1000,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, QualityGateCallback(), eval_callback],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)
```

- [ ] **Step 2: Verify the entry point parses args**

Run: `python -c "from scripts.vocos_finetune.train import main_phase_reg; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/vocos_finetune/train.py
git commit -m "feat: add main_phase_reg CLI entry point"
```

---

### Task 6: Training Shell Script

**Files:**
- Create: `scripts/vocos_phase_reg.sh`

- [ ] **Step 1: Write the training script**

```bash
#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILELIST="training/data/filelists/train.txt"
VAL_FILELIST="training/data/filelists/val.txt"
CHECKPOINT_DIR="checkpoints/vocos_phase_reg"
LOG_DIR="logs/vocos_phase_reg"
MAX_STEPS=5000
BATCH_SIZE=16
LR=1e-5
PHASE_COEFF=0.05

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

echo "=== Vocos Phase Regularization Training ==="
echo "Phase coeff: $PHASE_COEFF"
echo "LR: $LR"
echo "Max steps: $MAX_STEPS"
echo ""

# Stage 1: Verify data exists
if [ ! -f "$TRAIN_FILELIST" ]; then
    echo "ERROR: Training filelist not found at $TRAIN_FILELIST"
    echo "Run the vocos_finetune.sh data download first."
    exit 1
fi
echo "[1/3] Data verified."

# Stage 2: Train
echo "[2/3] Starting training..."
python -m scripts.vocos_finetune.train_phase_reg \
    --train-filelist "$TRAIN_FILELIST" \
    --val-filelist "$VAL_FILELIST" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --phase-coeff "$PHASE_COEFF" \
    $RESUME_FLAG

echo "[2/3] Training complete."

# Stage 3: Convert best checkpoint to MLX
echo "[3/3] Converting best checkpoint to MLX..."
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*.ckpt | grep -v last | head -1)
python -m scripts.vocos_finetune.convert_mlx \
    --checkpoint "$BEST_CKPT" \
    --output-dir "training/models/vocos-mel-24khz-phase-reg"

echo "=== Done ==="
echo "MLX weights: training/models/vocos-mel-24khz-phase-reg/weights.npz"
echo "TensorBoard: tensorboard --logdir $LOG_DIR"
```

Note: this script references `scripts.vocos_finetune.train_phase_reg` as a module. We need a small wrapper file for `python -m` invocation.

- [ ] **Step 2: Create the module entry point**

Create `scripts/vocos_finetune/train_phase_reg.py`:

```python
from scripts.vocos_finetune.train import main_phase_reg

if __name__ == "__main__":
    main_phase_reg()
```

- [ ] **Step 3: Make shell script executable and test**

Run: `chmod +x scripts/vocos_phase_reg.sh && head -5 scripts/vocos_phase_reg.sh`
Expected: Shows the shebang line

- [ ] **Step 4: Commit**

```bash
git add scripts/vocos_phase_reg.sh scripts/vocos_finetune/train_phase_reg.py
git commit -m "feat: add phase regularization training script"
```

---

### Task 7: Run Training and Evaluate

- [ ] **Step 1: Verify existing tests still pass**

Run: `python -m pytest tests/test_vocos_finetune_train.py tests/test_vocos_finetune_phase_loss.py tests/test_vocos_finetune_phase_reg.py tests/test_vocos_finetune_click.py -v 2>&1 | tee /tmp/phase_reg_tests.log`
Expected: All tests PASS

- [ ] **Step 2: Launch training**

Run: `bash scripts/vocos_phase_reg.sh 2>&1 | tee /tmp/phase_reg_training.log`
Expected: Training starts, logs mel_loss and phase_ifd_loss to TensorBoard

- [ ] **Step 3: Monitor training progress**

After ~1000 steps, check TensorBoard logs:
Run: `tail -20 /tmp/phase_reg_training.log`
Verify: phase_ifd_loss is being logged and decreasing; mel_loss_normal is not regressing significantly

- [ ] **Step 4: Convert best checkpoint to MLX**

This happens automatically in the shell script (Stage 3). Verify:
Run: `ls -la training/models/vocos-mel-24khz-phase-reg/weights.npz`
Expected: weights.npz file exists

- [ ] **Step 5: Generate A/B comparison samples**

Run:
```bash
python -m scripts.vocos_finetune.evaluate \
    --checkpoint "$(ls -t checkpoints/vocos_phase_reg/*.ckpt | grep -v last | head -1)" \
    --output-dir training/eval_samples_phase_reg/final \
    --n-utterances 5 \
    --model-type phase_reg
```

- [ ] **Step 6: Send samples for listening test**

Copy samples to the user's machine for perceptual evaluation:
```bash
for f in training/eval_samples_phase_reg/final/*.wav; do
    tailscale file cp "$f" crysknife-lapis:
done
```

- [ ] **Step 7: Commit training results documentation**

Update `docs/research/vocoder-evaluation-results.md` with phase regularization results after listening test feedback.
