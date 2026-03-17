import numpy as np
import mlx.core as mx
import mlx.nn as nn

_model = None
_mel_basis = None
_window = None


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = mx.ones((dim,)) * layer_scale_init_value

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x


class VocosMLX(nn.Module):
    def __init__(
        self,
        input_channels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        layer_scale = 1.0 / num_layers
        self.convnext = [
            ConvNeXtBlock(dim, intermediate_dim, layer_scale)
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head_out = nn.Linear(dim, n_fft + 2)

    def __call__(self, features: mx.array) -> mx.array:
        x = mx.transpose(features, axes=(0, 2, 1))
        x = self.embed(x)
        x = self.norm(x)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x)

        x = self.head_out(x)
        n_freq = self.n_fft // 2 + 1
        mag = x[:, :, :n_freq]
        phase = x[:, :, n_freq:]

        mag = mx.exp(mag)
        mag = mx.clip(mag, a_min=None, a_max=1e2)
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)

        return self._istft(real, imag)

    def _istft(self, real: mx.array, imag: mx.array) -> mx.array:
        B, T, F = real.shape
        n_fft = self.n_fft
        hop = self.hop_length

        window = mx.array(_get_window(n_fft))
        out_length = (T - 1) * hop + n_fft

        spec = real + 1j * imag
        frames = mx.fft.irfft(spec, n=n_fft)
        frames = frames * window[None, None, :]
        mx.eval(frames)

        frames_np = np.array(frames)
        window_sq = np.array(window) ** 2

        offsets = np.arange(T) * hop
        indices = offsets[:, None] + np.arange(n_fft)[None, :]
        flat_indices = indices.ravel()

        output_np = np.zeros((B, out_length), dtype=np.float32)
        norm_np = np.zeros(out_length, dtype=np.float32)

        for b in range(B):
            np.add.at(output_np[b], flat_indices, frames_np[b].ravel())
        np.add.at(norm_np, flat_indices, np.tile(window_sq, T))

        norm_np = np.maximum(norm_np, 1e-8)
        output_np /= norm_np[None, :]

        trim = n_fft // 2
        output_np = output_np[:, trim:out_length - trim]

        return mx.array(output_np)


def _get_window(n_fft: int) -> np.ndarray:
    global _window
    if _window is not None and len(_window) == n_fft:
        return _window
    _window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    return _window


def _get_mel_basis(sr: int = 24000, n_fft: int = 1024, n_mels: int = 100) -> np.ndarray:
    global _mel_basis
    if _mel_basis is not None:
        return _mel_basis

    n_freq = n_fft // 2 + 1
    fmin, fmax = 0.0, sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)

    fft_freqs = np.linspace(0, sr / 2, n_freq)

    basis = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]
        for j in range(n_freq):
            if lower <= fft_freqs[j] <= center:
                basis[i, j] = (fft_freqs[j] - lower) / (center - lower + 1e-10)
            elif center < fft_freqs[j] <= upper:
                basis[i, j] = (upper - fft_freqs[j]) / (upper - center + 1e-10)

    _mel_basis = basis
    return basis


def extract_mel(samples: np.ndarray, sr: int = 24000, n_fft: int = 1024, hop: int = 256, n_mels: int = 100) -> np.ndarray:
    padded = np.pad(samples, (n_fft // 2, n_fft // 2), mode="reflect")
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    mel_basis = _get_mel_basis(sr, n_fft, n_mels)

    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
    ).copy()
    frames *= window

    spec = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(spec)

    mel = mel_basis @ mag.T
    mel = np.log(np.maximum(mel, 1e-5))
    return mel


def _load_model() -> VocosMLX:
    global _model
    if _model is not None:
        return _model

    from vocos import Vocos
    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    model = VocosMLX()
    weights = _convert_weights(pt_model)
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())

    _model = model
    return model


def _convert_weights(pt_model) -> list[tuple[str, mx.array]]:
    import torch
    weights = []

    def _c(name: str, tensor: torch.Tensor):
        arr = tensor.detach().cpu().numpy()
        weights.append((name, mx.array(arr)))

    sd = pt_model.state_dict()

    _c("embed.weight", sd["backbone.embed.weight"].transpose(1, 2))
    _c("embed.bias", sd["backbone.embed.bias"])

    _c("norm.weight", sd["backbone.norm.weight"])
    _c("norm.bias", sd["backbone.norm.bias"])

    for i in range(8):
        pt_pre = f"backbone.convnext.{i}"
        mlx_pre = f"convnext.{i}"

        dw_w = sd[f"{pt_pre}.dwconv.weight"]
        _c(f"{mlx_pre}.dwconv.weight", dw_w.transpose(1, 2))
        _c(f"{mlx_pre}.dwconv.bias", sd[f"{pt_pre}.dwconv.bias"])

        _c(f"{mlx_pre}.norm.weight", sd[f"{pt_pre}.norm.weight"])
        _c(f"{mlx_pre}.norm.bias", sd[f"{pt_pre}.norm.bias"])

        _c(f"{mlx_pre}.pwconv1.weight", sd[f"{pt_pre}.pwconv1.weight"])
        _c(f"{mlx_pre}.pwconv1.bias", sd[f"{pt_pre}.pwconv1.bias"])

        _c(f"{mlx_pre}.pwconv2.weight", sd[f"{pt_pre}.pwconv2.weight"])
        _c(f"{mlx_pre}.pwconv2.bias", sd[f"{pt_pre}.pwconv2.bias"])

        _c(f"{mlx_pre}.gamma", sd[f"{pt_pre}.gamma"])

    _c("final_layer_norm.weight", sd["backbone.final_layer_norm.weight"])
    _c("final_layer_norm.bias", sd["backbone.final_layer_norm.bias"])

    _c("head_out.weight", sd["head.out.weight"])
    _c("head_out.bias", sd["head.out.bias"])

    return weights


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


def vocos_mlx_variable_rate(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
    smoothing_sigma: float = 0.7,
) -> np.ndarray:
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d

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
        resampled = gaussian_filter1d(resampled, sigma=smoothing_sigma, axis=1)

    features = mx.array(resampled.astype(np.float32)[np.newaxis])
    audio = model(features)
    mx.eval(audio)

    return np.array(audio).squeeze().astype(np.float32)
