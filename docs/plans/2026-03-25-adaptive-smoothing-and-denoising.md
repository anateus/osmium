# Adaptive Smoothing & Voice Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce peakiness/choppiness at high speed factors and add three-tier voice cleanup preprocessing.

**Architecture:** Two independent feature groups wired into the existing pipeline. Smoothing changes modify the rate schedule and mel processing (stages 2-3). Denoising adds a new preprocessing step between decode and stage 1. Both are controlled by new CLI flags.

**Tech Stack:** numpy, scipy, Click (existing); DeepFilterNet, Demucs (new optional deps)

**Spec:** `docs/specs/2026-03-25-adaptive-smoothing-and-denoising-design.md`

---

### Task 1: Rate contrast compression

Add `gamma` parameter to `importance_to_rate_schedule` that compresses the rate swing for mid-importance content.

**Files:**
- Modify: `src/osmium/tsm/rate_schedule.py:16-33`
- Create: `tests/test_rate_schedule.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rate_schedule.py
import numpy as np
from osmium.tsm.rate_schedule import importance_to_rate_schedule


def test_gamma_1_produces_linear_rates():
    """gamma=1.0 should produce rates using the linear inv_importance formula."""
    importance = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    times = np.linspace(0, 1.0, 5)
    rates, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.0,
        smoothing_sigma=0, max_rate_change=100.0,
    )
    # With smoothing/slew disabled, rates should follow inv_importance linearly
    # inv_importance = [1.0, 0.75, 0.5, 0.25, 0.0]
    # raw_rates = 1 + inv * 9 = [10, 7.75, 5.5, 3.25, 1]
    # The iterative loop rescales to hit target, but relative ordering is preserved
    assert rates[0] > rates[2] > rates[4]  # monotonically decreasing
    # Ratios between adjacent rates should be roughly linear
    diffs = np.diff(rates)
    assert np.all(diffs < 0)  # all decreasing


def test_gamma_compresses_mid_range_rates():
    """gamma=1.5 should produce lower rates for mid-importance content vs gamma=1.0."""
    importance = np.linspace(0, 1, 50)
    times = np.linspace(0, 5.0, 50)
    rates_linear, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.0,
    )
    rates_compressed, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.5,
    )
    # Mid-range frames (importance ~0.5) should have lower rates with gamma>1
    mid = len(importance) // 2
    mid_slice = slice(mid - 3, mid + 3)
    assert rates_compressed[mid_slice].mean() < rates_linear[mid_slice].mean()


def test_gamma_preserves_target_speed():
    """Total output duration should still match target speed regardless of gamma."""
    importance = np.random.RandomState(42).rand(200)
    times = np.linspace(0, 10.0, 200)
    for gamma in [1.0, 1.5, 2.0]:
        rates, _ = importance_to_rate_schedule(
            importance, times, target_speed=3.0, gamma=gamma,
        )
        dt = np.diff(times, prepend=0)
        dt[0] = times[0]
        output_dur = (dt / rates).sum()
        target_dur = times[-1] / 3.0
        assert abs(output_dur - target_dur) / target_dur < 0.01, (
            f"gamma={gamma}: output_dur={output_dur:.3f} vs target={target_dur:.3f}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rate_schedule.py -v`
Expected: FAIL — `gamma` parameter not accepted

- [ ] **Step 3: Implement gamma parameter**

In `src/osmium/tsm/rate_schedule.py`, add `gamma` parameter and apply power curve:

```python
def importance_to_rate_schedule(
    importance: np.ndarray,
    importance_times: np.ndarray,
    target_speed: float,
    min_rate: float = 1.0,
    max_rate: float = 10.0,
    smoothing_sigma: float = 15.0,
    max_rate_change: float = 0.3,
    gamma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    imp = importance.copy()
    imp = np.clip(imp, 0, 1)

    if smoothing_sigma > 0 and len(imp) > 1:
        imp = gaussian_filter1d(imp, sigma=smoothing_sigma)
        imp = np.clip(imp, 0, 1)

    inv_importance = 1.0 - imp
    if gamma != 1.0:
        inv_importance = inv_importance ** gamma
    raw_rates = min_rate + inv_importance * (max_rate - min_rate)
    # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rate_schedule.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_rate_schedule.py src/osmium/tsm/rate_schedule.py
git commit -m "feat: rate contrast compression (gamma parameter)"
```

---

### Task 2: Consonant sensitivity boost

Bump mel importance defaults for better consonant detection.

**Files:**
- Modify: `src/osmium/analyzer/mel_importance.py:5-11`
- Create: `tests/test_mel_importance.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_mel_importance.py
import numpy as np
from osmium.analyzer.mel_importance import compute_mel_importance


def test_default_hf_boost_is_2_5():
    """Default hf_boost should be 2.5 for better consonant sensitivity."""
    import inspect
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["hf_boost"].default == 2.5


def test_default_flux_weight_is_0_65():
    """Default flux weight should be 0.65 to favor spectral change over energy."""
    import inspect
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_flux"].default == 0.65


def test_default_energy_weight_is_0_35():
    """Default energy weight should be 0.35."""
    import inspect
    sig = inspect.signature(compute_mel_importance)
    assert sig.parameters["weight_energy"].default == 0.35


def test_higher_hf_boost_increases_consonant_importance():
    """Higher hf_boost should increase importance at frames with high-frequency activity."""
    rng = np.random.RandomState(42)
    n_mels, T = 100, 50
    mel = rng.randn(n_mels, T).astype(np.float32) * 0.1
    # Simulate consonant burst at frame 25 in high frequencies
    mel[50:, 25] += 3.0
    mel[50:, 24] += 0.1  # low value before to create high flux

    imp_low = compute_mel_importance(mel, 1.0, hf_boost=2.0)
    imp_high = compute_mel_importance(mel, 1.0, hf_boost=2.5)
    assert imp_high.scores[25] >= imp_low.scores[25]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mel_importance.py -v`
Expected: FAIL on default value checks (currently 2.0, 0.6, 0.4)

- [ ] **Step 3: Update defaults**

In `src/osmium/analyzer/mel_importance.py`:

```python
def compute_mel_importance(
    mel: np.ndarray,
    duration: float,
    weight_flux: float = 0.65,
    weight_energy: float = 0.35,
    hf_boost: float = 2.5,
) -> ImportanceMap:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mel_importance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mel_importance.py src/osmium/analyzer/mel_importance.py
git commit -m "feat: bump consonant sensitivity defaults (hf_boost=2.5, flux=0.65)"
```

---

### Task 3: Adaptive mel smoothing helper

Create shared `adaptive_smooth_mel()` function that applies variable-width Gaussian smoothing based on local compression ratio.

**Files:**
- Create: `src/osmium/tsm/smooth.py`
- Create: `tests/test_smooth.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smooth.py
import numpy as np
from osmium.tsm.smooth import adaptive_smooth_mel


def test_uniform_compression_matches_fixed_sigma():
    """When compression ratio is uniform, result should approximate gaussian_filter1d."""
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 200).astype(np.float32)
    ratios = np.ones(200)  # uniform = ratio 1.0 everywhere

    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.7, sigma_max=0.7)
    expected = gaussian_filter1d(mel, sigma=0.7, axis=1)
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_high_compression_gets_more_smoothing():
    """Regions with high compression ratio should be smoother than low-compression regions."""
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 200).astype(np.float32)
    # First half: low compression (ratio=1), second half: high compression (ratio=8)
    ratios = np.ones(200)
    ratios[100:] = 8.0

    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.3, sigma_max=2.5)
    # High-compression region should have lower variance (more smoothed)
    var_low = np.var(np.diff(result[:, :90], axis=1))
    var_high = np.var(np.diff(result[:, 110:], axis=1))
    assert var_high < var_low


def test_returns_same_shape():
    """Output shape must match input shape."""
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 50).astype(np.float32)
    ratios = np.ones(50) * 3.0
    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.3, sigma_max=2.5)
    assert result.shape == mel.shape


def test_sigma_min_clamped_to_sigma_max():
    """If sigma_max < default sigma_min, sigma_min should clamp to sigma_max."""
    rng = np.random.RandomState(42)
    mel = rng.randn(100, 50).astype(np.float32)
    ratios = np.ones(50) * 5.0
    # sigma_max=0.2 < default sigma_min=0.3, should not error
    result = adaptive_smooth_mel(mel, ratios, sigma_min=0.2, sigma_max=0.2)
    assert result.shape == mel.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smooth.py -v`
Expected: FAIL — module `osmium.tsm.smooth` does not exist

- [ ] **Step 3: Implement adaptive_smooth_mel**

```python
# src/osmium/tsm/smooth.py
import numpy as np
from scipy.ndimage import gaussian_filter1d


def adaptive_smooth_mel(
    mel: np.ndarray,
    compression_ratios: np.ndarray,
    sigma_min: float = 0.3,
    sigma_max: float = 2.5,
    n_buckets: int = 4,
    crossfade_frames: int = 4,
) -> np.ndarray:
    sigma_min = min(sigma_min, sigma_max)
    T = mel.shape[1]

    if T < 2 or sigma_max <= 0:
        return mel.copy()

    ratio_min = compression_ratios.min()
    ratio_max = compression_ratios.max()

    if ratio_max <= ratio_min:
        sigma = sigma_min
        return gaussian_filter1d(mel, sigma=max(sigma, 0.01), axis=1)

    norm_ratios = (compression_ratios - ratio_min) / (ratio_max - ratio_min)
    per_frame_sigma = sigma_min + norm_ratios * (sigma_max - sigma_min)

    bucket_sigmas = np.linspace(sigma_min, sigma_max, n_buckets)
    bucket_indices = np.argmin(
        np.abs(per_frame_sigma[:, None] - bucket_sigmas[None, :]), axis=1
    )

    smoothed_layers = np.stack([
        gaussian_filter1d(mel, sigma=max(s, 0.01), axis=1)
        for s in bucket_sigmas
    ])

    result = np.empty_like(mel)
    for t in range(T):
        result[:, t] = smoothed_layers[bucket_indices[t], :, t]

    if crossfade_frames > 0 and T > crossfade_frames * 2:
        changes = np.where(np.diff(bucket_indices) != 0)[0]
        for c in changes:
            half = crossfade_frames // 2
            start = max(0, c + 1 - half)
            end = min(T, c + 1 + half)
            length = end - start
            if length < 2:
                continue
            fade = np.linspace(0, 1, length, dtype=np.float32)
            b_before = bucket_indices[c]
            b_after = bucket_indices[min(c + 1, T - 1)]
            blended = (
                smoothed_layers[b_before, :, start:end] * (1 - fade)[None, :]
                + smoothed_layers[b_after, :, start:end] * fade[None, :]
            )
            result[:, start:end] = blended

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_smooth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/osmium/tsm/smooth.py tests/test_smooth.py
git commit -m "feat: adaptive mel smoothing helper with bucketed variable sigma"
```

---

### Task 4: Integrate adaptive smoothing into vocos engines

Replace fixed `gaussian_filter1d` in both vocos engines with `adaptive_smooth_mel`, computing compression ratios from the source index mapping. Clean up dead code in `vocos_engine.py`.

**Files:**
- Modify: `src/osmium/tsm/vocos_mlx.py:267-307`
- Modify: `src/osmium/tsm/vocos_engine.py:48-97`

- [ ] **Step 1: Update `vocos_mlx_variable_rate` in `vocos_mlx.py`**

Replace the smoothing section (lines 300-301) with adaptive smoothing. The compression ratio is derived from `source_indices` — the derivative tells us how many input frames map to each output frame:

```python
def vocos_mlx_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    from scipy.interpolate import interp1d
    from osmium.tsm.smooth import adaptive_smooth_mel

    model = _load_model()
    mel = extract_mel(samples, sample_rate)

    T = mel.shape[1]
    duration = len(samples) / sample_rate
    mel_times = np.linspace(0, duration, T)

    mel_rates = np.interp(mel_times, rate_times, rate_curve)
    mel_rates = np.maximum(mel_rates, 0.5)

    dt = np.diff(mel_times, prepend=0)
    dt[0] = mel_times[0] if T > 0 else 0
    output_times = np.cumsum(dt / mel_rates)

    total_output_duration = output_times[-1]
    target_T = max(1, int(total_output_duration / (duration / T)))

    target_mel_times = np.linspace(0, total_output_duration, target_T)
    source_indices = np.interp(target_mel_times, output_times, np.arange(T))

    interp_fn = interp1d(np.arange(T), mel, axis=1, kind="cubic", fill_value="extrapolate")
    resampled = interp_fn(source_indices)

    if smoothing_sigma > 0:
        compression_ratios = np.gradient(source_indices)
        compression_ratios = np.maximum(compression_ratios, 0.1)
        sigma_min = min(0.3, smoothing_sigma)
        resampled = adaptive_smooth_mel(
            resampled, compression_ratios,
            sigma_min=sigma_min, sigma_max=smoothing_sigma,
        )

    features = mx.array(resampled.astype(np.float32)[np.newaxis])
    audio = model(features)
    mx.eval(audio)

    return np.array(audio).squeeze().astype(np.float32)
```

Also update `vocos_mlx_stretch` — uniform stretch has uniform compression, so adaptive smoothing reduces to fixed sigma (no behavioral change, but keeps the code path consistent):

```python
def vocos_mlx_stretch(
    samples: np.ndarray,
    speed: float,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d

    model = _load_model()
    mel = extract_mel(samples, sample_rate)

    T = mel.shape[1]
    target_T = max(1, int(T / speed))

    source_t = np.arange(T)
    target_t = np.linspace(0, T - 1, target_T)
    interp_fn = interp1d(source_t, mel, axis=1, kind="cubic")
    resampled = interp_fn(target_t)

    if smoothing_sigma > 0:
        resampled = gaussian_filter1d(resampled, sigma=smoothing_sigma, axis=1)

    features = mx.array(resampled.astype(np.float32)[np.newaxis])
    audio = model(features)
    mx.eval(audio)

    return np.array(audio).squeeze().astype(np.float32)
```

(No change to `vocos_mlx_stretch` — uniform stretch uses fixed sigma, which is correct.)

- [ ] **Step 2: Update `vocos_variable_rate` in `vocos_engine.py`**

Clean up dead code (duplicate `target_T` on line 73, unused linear interpolation loop lines 80-87) and add adaptive smoothing. **Keep existing top-level imports** (`interp1d`, `gaussian_filter1d`) since `vocos_stretch` still uses them:

```python
def vocos_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    from osmium.tsm.smooth import adaptive_smooth_mel

    vocos = _load_vocos()

    audio_tensor = torch.from_numpy(samples.copy()).unsqueeze(0)
    features = vocos.feature_extractor(audio_tensor)
    feat_np = features.numpy()

    T = feat_np.shape[2]
    duration = len(samples) / sample_rate
    mel_times = np.linspace(0, duration, T)

    mel_rates = np.interp(mel_times, rate_times, rate_curve)
    mel_rates = np.maximum(mel_rates, 0.5)

    dt = np.diff(mel_times, prepend=0)
    dt[0] = mel_times[0] if T > 0 else 0
    output_times = np.cumsum(dt / mel_rates)

    total_output_duration = output_times[-1]
    target_T = max(1, int(total_output_duration / (duration / T)))

    target_mel_times = np.linspace(0, total_output_duration, target_T)
    source_indices = np.interp(target_mel_times, output_times, np.arange(T))

    mel_2d = feat_np[0]
    interp_fn = interp1d(np.arange(T), mel_2d, axis=1, kind="cubic", fill_value="extrapolate")
    resampled = interp_fn(source_indices)

    if smoothing_sigma > 0:
        compression_ratios = np.gradient(source_indices)
        compression_ratios = np.maximum(compression_ratios, 0.1)
        sigma_min = min(0.3, smoothing_sigma)
        resampled = adaptive_smooth_mel(
            resampled, compression_ratios,
            sigma_min=sigma_min, sigma_max=smoothing_sigma,
        )

    resampled_tensor = torch.from_numpy(resampled.astype(np.float32)[np.newaxis])
    audio_out = vocos.decode(resampled_tensor)
    return audio_out.squeeze().numpy()
```

- [ ] **Step 3: Verify no import errors**

Run: `uv run python -c "from osmium.tsm.vocos_mlx import vocos_mlx_variable_rate; print('ok')"`
Expected: `ok` (or import error for mlx if not on Apple Silicon — that's fine, the fallback path exists)

- [ ] **Step 4: Commit**

```bash
git add src/osmium/tsm/vocos_mlx.py src/osmium/tsm/vocos_engine.py
git commit -m "feat: integrate adaptive mel smoothing into vocos engines"
```

---

### Task 5: Spectral gating denoiser

Implement the DSP-based spectral gate (Tier 1 denoising).

**Files:**
- Create: `src/osmium/analyzer/denoise.py`
- Create: `tests/test_denoise.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_denoise.py
import numpy as np
from osmium.analyzer.denoise import spectral_gate


def _make_noisy_speech(sr=24000, duration=2.0, noise_level=0.05):
    """Create a test signal: sine tone (simulated speech) + white noise."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    speech = 0.5 * np.sin(2 * np.pi * 300 * t)  # 300Hz tone
    # Add envelope to simulate speech pauses
    envelope = np.ones_like(t)
    envelope[int(0.8 * sr):int(1.2 * sr)] = 0  # silence gap
    speech *= envelope
    noise = noise_level * np.random.RandomState(42).randn(len(t)).astype(np.float32)
    return speech + noise, speech, sr


def test_spectral_gate_reduces_noise():
    """Output should have lower noise in silent regions than input."""
    noisy, clean, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)

    # Measure RMS in the silent gap (0.8s-1.2s)
    gap = slice(int(0.8 * sr), int(1.2 * sr))
    input_noise = np.sqrt(np.mean(noisy[gap] ** 2))
    output_noise = np.sqrt(np.mean(denoised[gap] ** 2))
    assert output_noise < input_noise * 0.5


def test_spectral_gate_preserves_speech():
    """Output should preserve the speech signal (not attenuate it significantly)."""
    noisy, clean, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)

    # Speech region (0-0.8s) should retain most energy
    speech_region = slice(0, int(0.8 * sr))
    input_rms = np.sqrt(np.mean(noisy[speech_region] ** 2))
    output_rms = np.sqrt(np.mean(denoised[speech_region] ** 2))
    assert output_rms > input_rms * 0.7


def test_spectral_gate_returns_same_length():
    """Output length must match input length."""
    noisy, _, sr = _make_noisy_speech()
    denoised = spectral_gate(noisy, sr)
    assert len(denoised) == len(noisy)


def test_spectral_gate_handles_silence():
    """Pure silence should not crash."""
    silence = np.zeros(24000, dtype=np.float32)
    result = spectral_gate(silence, 24000)
    assert len(result) == 24000
    assert np.max(np.abs(result)) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_denoise.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement spectral gating**

```python
# src/osmium/analyzer/denoise.py
import numpy as np


def spectral_gate(
    samples: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 2048,
    hop: int = 512,
    noise_percentile: float = 0.12,
    threshold_db: float = 6.0,
    smoothing_frames: int = 3,
    chunk_duration: float = 300.0,
    overlap_duration: float = 0.5,
) -> np.ndarray:
    total_samples = len(samples)
    chunk_samples = int(chunk_duration * sample_rate)

    if total_samples <= chunk_samples:
        return _spectral_gate_chunk(
            samples, sample_rate, n_fft, hop,
            noise_percentile, threshold_db, smoothing_frames,
        )

    overlap_samples = int(overlap_duration * sample_rate)
    parts = []
    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_samples + overlap_samples, total_samples)
        start = max(0, pos - overlap_samples) if pos > 0 else 0
        chunk = samples[start:end]
        processed = _spectral_gate_chunk(
            chunk, sample_rate, n_fft, hop,
            noise_percentile, threshold_db, smoothing_frames,
        )
        ob = pos - start if pos > 0 else 0
        oa = end - (pos + chunk_samples) if end < total_samples else 0
        if pos == 0:
            parts.append(processed[:len(processed) - oa] if oa > 0 else processed)
        else:
            if ob > 0 and parts:
                xf = min(ob, len(parts[-1]), len(processed))
                if xf > 0:
                    fade = np.linspace(0, 1, xf, dtype=np.float32)
                    blended = parts[-1][-xf:] * (1 - fade) + processed[:xf] * fade
                    parts[-1] = parts[-1][:-xf]
                    parts.append(blended)
                trimmed = processed[xf:len(processed) - oa] if oa > 0 else processed[xf:]
            else:
                trimmed = processed[:len(processed) - oa] if oa > 0 else processed
            parts.append(trimmed)
        pos += chunk_samples

    return np.concatenate(parts).astype(np.float32)


def _spectral_gate_chunk(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop: int,
    noise_percentile: float,
    threshold_db: float,
    smoothing_frames: int,
) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    original_len = len(samples)
    padded = np.pad(samples, (n_fft // 2, n_fft // 2), mode="reflect")
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)

    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
    ).copy()
    frames *= window

    spec = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(spec)
    phase = np.angle(spec)

    frame_rms = np.sqrt(np.mean(mag ** 2, axis=1))
    n_noise = max(1, int(n_frames * noise_percentile))
    noise_frame_indices = np.argsort(frame_rms)[:n_noise]
    noise_floor = np.mean(mag[noise_frame_indices], axis=0)

    threshold_linear = 10 ** (threshold_db / 20.0)
    snr = mag / (noise_floor[None, :] + 1e-10)

    gain = np.clip((snr - 1.0) / (threshold_linear - 1.0 + 1e-10), 0.0, 1.0)
    gain = gain ** 2

    if smoothing_frames > 1:
        gain = uniform_filter1d(gain, size=smoothing_frames, axis=0)

    cleaned_mag = mag * gain
    cleaned_spec = cleaned_mag * np.exp(1j * phase)
    cleaned_frames = np.fft.irfft(cleaned_spec, n=n_fft)
    cleaned_frames *= window

    output = np.zeros(len(padded), dtype=np.float32)
    window_sum = np.zeros(len(padded), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        output[start:start + n_fft] += cleaned_frames[i]
        window_sum[start:start + n_fft] += window ** 2

    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum

    trim = n_fft // 2
    output = output[trim:trim + original_len]
    return output.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_denoise.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/osmium/analyzer/denoise.py tests/test_denoise.py
git commit -m "feat: spectral gating denoiser (tier 1, DSP-only)"
```

---

### Task 6: DeepFilterNet wrapper

Wrap DeepFilterNet as Tier 2 denoiser with 24k↔48k resampling.

**Files:**
- Create: `src/osmium/analyzer/denoise_deep.py`
- Modify: `pyproject.toml` (add `denoise` optional dep group)

- [ ] **Step 1: Implement wrapper**

```python
# src/osmium/analyzer/denoise_deep.py
import numpy as np


def deep_filter(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    from scipy.signal import resample_poly

    try:
        from df import enhance, init_df
    except ImportError:
        raise ImportError(
            "DeepFilterNet not installed. Install with: uv pip install -e '.[denoise]'"
        )

    model, df_state, _ = init_df()

    if sample_rate == 24000:
        upsampled = resample_poly(samples, 2, 1).astype(np.float32)
    elif sample_rate == 48000:
        upsampled = samples
    else:
        ratio = 48000 / sample_rate
        up = int(ratio * 1000)
        down = 1000
        from math import gcd
        g = gcd(up, down)
        upsampled = resample_poly(samples, up // g, down // g).astype(np.float32)

    import torch
    audio_tensor = torch.from_numpy(upsampled).unsqueeze(0)
    enhanced = enhance(model, df_state, audio_tensor)
    enhanced_np = enhanced.squeeze().numpy()

    if sample_rate == 48000:
        return enhanced_np.astype(np.float32)
    elif sample_rate == 24000:
        return resample_poly(enhanced_np, 1, 2).astype(np.float32)
    else:
        ratio = sample_rate / 48000
        up = int(ratio * 1000)
        down = 1000
        from math import gcd
        g = gcd(up, down)
        return resample_poly(enhanced_np, up // g, down // g).astype(np.float32)
```

- [ ] **Step 2: Add `denoise` dependency group to pyproject.toml**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
denoise = [
    "deepfilternet>=0.5,<1.0",
]
```

- [ ] **Step 3: Verify import error message**

Run: `uv run python -c "import numpy as np; from osmium.analyzer.denoise_deep import deep_filter; deep_filter(np.zeros(100, dtype=np.float32))"`
Expected: `ImportError: DeepFilterNet not installed...`

- [ ] **Step 4: Commit**

```bash
git add src/osmium/analyzer/denoise_deep.py pyproject.toml
git commit -m "feat: DeepFilterNet denoiser wrapper (tier 2)"
```

---

### Task 7: Demucs wrapper

Wrap Demucs HTDemucs as Tier 3 source separation.

**Files:**
- Create: `src/osmium/analyzer/denoise_demucs.py`
- Modify: `pyproject.toml` (add `demucs` optional dep group)

- [ ] **Step 1: Implement wrapper**

```python
# src/osmium/analyzer/denoise_demucs.py
import numpy as np


def demucs_separate(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    try:
        import demucs.api
    except ImportError:
        raise ImportError(
            "Demucs not installed. Install with: uv pip install -e '.[demucs]'"
        )
    import torch

    import sys
    sys.stderr.write("Loading htdemucs model (may download ~1GB on first run)...\n")
    separator = demucs.api.Separator(model="htdemucs", segment=12)

    if samples.ndim == 1:
        audio_tensor = torch.from_numpy(samples.copy()).unsqueeze(0).float()
    else:
        audio_tensor = torch.from_numpy(samples.copy()).float()

    if sample_rate != separator.samplerate:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, sample_rate, separator.samplerate,
        )

    _, separated = separator.separate_tensor(audio_tensor)
    vocals = separated["vocals"]

    if sample_rate != separator.samplerate:
        import torchaudio
        vocals = torchaudio.functional.resample(
            vocals, separator.samplerate, sample_rate,
        )

    result = vocals.squeeze().numpy().astype(np.float32)
    if result.ndim > 1:
        result = result.mean(axis=0)

    return result
```

- [ ] **Step 2: Add `demucs` dependency group to pyproject.toml**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
demucs = [
    "demucs>=4.0",
]
```

- [ ] **Step 3: Verify import error message**

Run: `uv run python -c "from osmium.analyzer.denoise_demucs import demucs_separate; demucs_separate(None)"`
Expected: `ImportError: Demucs not installed...`

- [ ] **Step 4: Commit**

```bash
git add src/osmium/analyzer/denoise_demucs.py pyproject.toml
git commit -m "feat: Demucs source separation wrapper (tier 3)"
```

---

### Task 8: CLI integration

Wire `--denoise` and `--rate-gamma` into the CLI and batch pipeline.

**Files:**
- Modify: `src/osmium/cli.py`

- [ ] **Step 1: Add CLI options**

Update `--smoothing` help text and add two new Click options to the `main` function:

Update existing option:
```python
@click.option("--smoothing", default=0.7, type=float, help="Mel smoothing sigma; adaptive in variable-rate mode (0=off)")
```

New options:

```python
@click.option("--denoise", type=click.Choice(["gate", "deep", "demucs"]), default=None,
              help="Voice cleanup: gate (DSP), deep (DeepFilterNet), demucs (source separation)")
@click.option("--rate-gamma", "rate_gamma", type=float, default=1.5,
              help="Rate contrast compression (1.0=linear/off, >1=smoother rhythm)")
```

Update `main` function signature to include `denoise` and `rate_gamma`.

Add validation in `main`:

```python
if stream and denoise:
    raise click.UsageError("--denoise is not supported with --stream")
```

Update the `_batch_mode` call site in `main` (currently line 35):

```python
_batch_mode(input_file, speed, output_file, uniform, mimi, resolution,
            smoothing, analyze_only, chunk_duration, use_prosody=not no_prosody,
            denoise=denoise, rate_gamma=rate_gamma)
```

- [ ] **Step 2: Wire denoise into `_batch_mode`**

Add denoise step after decode, before importance analysis:

```python
def _batch_mode(input_file, speed, output_file, uniform, use_mimi, resolution,
                smoothing, analyze_only, chunk_duration, use_prosody=False,
                denoise=None, rate_gamma=1.5):
    # ... after decode, before importance ...

    if denoise:
        denoise_task = progress.add_task("Denoising", total=None, status=denoise)
        audio_samples = _apply_denoise(audio.samples, audio.sample_rate, denoise, console)
        audio = type(audio)(samples=audio_samples, sample_rate=audio.sample_rate)
        progress.remove_task(denoise_task)
```

Add the denoise dispatch function:

```python
def _apply_denoise(samples, sample_rate, method, console):
    if method == "gate":
        from osmium.analyzer.denoise import spectral_gate
        return spectral_gate(samples, sample_rate)
    elif method == "deep":
        try:
            from osmium.analyzer.denoise_deep import deep_filter
            return deep_filter(samples, sample_rate)
        except ImportError:
            console.print("[red]DeepFilterNet requires:[/red] uv pip install -e '.[denoise]'")
            raise SystemExit(1)
    elif method == "demucs":
        try:
            from osmium.analyzer.denoise_demucs import demucs_separate
            return demucs_separate(samples, sample_rate)
        except ImportError:
            console.print("[red]Demucs requires:[/red] uv pip install -e '.[demucs]'")
            raise SystemExit(1)
```

- [ ] **Step 3: Pass `rate_gamma` to rate schedule**

In the rate schedule call:

```python
rate_curve, rate_times = importance_to_rate_schedule(
    imp.scores, imp.times, target_speed=speed,
    gamma=rate_gamma,
)
```

- [ ] **Step 4: Verify CLI help**

Run: `uv run osmium --help`
Expected: should show `--denoise` and `--rate-gamma` options

- [ ] **Step 5: Verify error on --denoise + --stream**

Run: `uv run osmium --stream --denoise gate -s 2.0 samples/clips/clip_01_opening.wav 2>&1 || true`
Expected: `UsageError: --denoise is not supported with --stream`

- [ ] **Step 6: Commit**

```bash
git add src/osmium/cli.py
git commit -m "feat: wire --denoise and --rate-gamma CLI options"
```

---

### Task 9: Update eval harness

Update the WER evaluation harness to use new defaults and support `rate_gamma` sweeps.

**Files:**
- Modify: `scripts/eval_wer.py`

- [ ] **Step 1: Update defaults and add gamma support**

In `eval_osmium`, update the defaults in the `compute_mel_importance` call:

```python
imp = compute_mel_importance(
    mel, duration,
    weight_flux=cfg.get("flux_w", 0.65),
    weight_energy=cfg.get("energy_w", 0.35),
    hf_boost=cfg.get("hf_boost", 2.5),
)
```

Pass gamma to the rate schedule call:

```python
rate_curve, rate_times = importance_to_rate_schedule(
    imp.scores, imp.times, target_speed=speed,
    gamma=cfg.get("rate_gamma", 1.5),
)
```

Add `rate_gamma` to `build_sweep_configs`:

```python
elif param == "rate_gamma":
    configs[label] = {"rate_gamma": vf}
```

- [ ] **Step 2: Verify script runs**

Run: `uv run python scripts/eval_wer.py --help`
Expected: shows help without errors

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_wer.py
git commit -m "feat: update eval harness for new defaults and rate_gamma sweeps"
```
