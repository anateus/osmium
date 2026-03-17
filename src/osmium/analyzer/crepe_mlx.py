import numpy as np
from dataclasses import dataclass

PITCH_BINS = 360
SAMPLE_RATE = 16000
WINDOW_SIZE = 1024
CENTS_PER_BIN = 20
HOP_SIZE = 160

_model = None
_model_capacity = None


@dataclass
class PitchResult:
    pitch: np.ndarray
    confidence: np.ndarray
    times: np.ndarray
    sample_rate: int


def _build_model(capacity: str = "tiny"):
    import mlx.core as mx
    import mlx.nn as nn

    if capacity == "full":
        in_ch = [1, 1024, 128, 128, 128, 256]
        out_ch = [1024, 128, 128, 128, 256, 512]
        classifier_in = 2048
    elif capacity == "tiny":
        in_ch = [1, 128, 16, 16, 16, 32]
        out_ch = [128, 16, 16, 16, 32, 64]
        classifier_in = 256
    else:
        raise ValueError(f"Unsupported capacity: {capacity}")

    kernel_sizes = [512] + [64] * 5
    strides = [4] + [1] * 5
    paddings_first = [254, 31, 31, 31, 31, 31]
    paddings_second = [254, 32, 32, 32, 32, 32]

    class CrepeMlx(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = []
            self.bns = []
            for i in range(6):
                self.convs.append(nn.Conv1d(
                    in_channels=in_ch[i],
                    out_channels=out_ch[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=0,
                ))
                self.bns.append(nn.BatchNorm(
                    out_ch[i],
                    eps=0.0010000000474974513,
                    momentum=1.0,
                ))
            self.classifier = nn.Linear(classifier_in, PITCH_BINS)
            self._paddings_left = paddings_first
            self._paddings_right = paddings_second

        def __call__(self, x):
            # x: (batch, channels, length) -> (batch, length, channels) for MLX
            x = mx.transpose(x, (0, 2, 1))
            for i in range(6):
                pl = self._paddings_left[i]
                pr = self._paddings_right[i]
                x = mx.pad(x, [(0, 0), (pl, pr), (0, 0)])
                x = self.convs[i](x)
                x = nn.relu(x)
                # BatchNorm expects (batch, ..., channels)
                x = self.bns[i](x)
                # Max pool by 2 along time axis
                n = x.shape[1]
                x = x[:, :n - n % 2, :]
                x = x.reshape(x.shape[0], n // 2, 2, x.shape[2])
                x = mx.max(x, axis=2)

            x = x.reshape(x.shape[0], -1)
            return mx.sigmoid(self.classifier(x))

    return CrepeMlx(), classifier_in


def _load_model(capacity: str = "tiny"):
    global _model, _model_capacity
    if _model is not None and _model_capacity == capacity:
        return _model

    import mlx.core as mx
    import torch
    import os

    model, _ = _build_model(capacity)

    weights_path = os.path.join(
        os.path.dirname(__import__("torchcrepe").__file__),
        "assets", f"{capacity}.pth",
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)

    weights = []
    for i in range(6):
        # PyTorch Conv2d weight: (out_ch, in_ch, kH, kW) -> squeeze kW -> (out_ch, in_ch, k)
        # MLX Conv1d weight: (out_ch, k, in_ch)
        w = state[f"conv{i+1}.weight"].numpy().squeeze(-1)
        w = np.swapaxes(w, 1, 2)
        weights.append((f"convs.{i}.weight", mx.array(w)))

        b = state[f"conv{i+1}.bias"].numpy()
        weights.append((f"convs.{i}.bias", mx.array(b)))

        bn_w = state[f"conv{i+1}_BN.weight"].numpy()
        bn_b = state[f"conv{i+1}_BN.bias"].numpy()
        bn_rm = state[f"conv{i+1}_BN.running_mean"].numpy()
        bn_rv = state[f"conv{i+1}_BN.running_var"].numpy()
        weights.append((f"bns.{i}.weight", mx.array(bn_w)))
        weights.append((f"bns.{i}.bias", mx.array(bn_b)))
        weights.append((f"bns.{i}.running_mean", mx.array(bn_rm)))
        weights.append((f"bns.{i}.running_var", mx.array(bn_rv)))

    clf_w = state["classifier.weight"].numpy()
    clf_b = state["classifier.bias"].numpy()
    weights.append(("classifier.weight", mx.array(clf_w)))
    weights.append(("classifier.bias", mx.array(clf_b)))

    model.load_weights(weights)
    mx.eval(model.parameters())

    _model = model
    _model_capacity = capacity
    return model


def predict(
    samples: np.ndarray,
    sample_rate: int = 24000,
    capacity: str = "tiny",
    batch_size: int = 512,
) -> PitchResult:
    import mlx.core as mx

    model = _load_model(capacity)

    if sample_rate != SAMPLE_RATE:
        from scipy.signal import resample
        n_out = int(len(samples) * SAMPLE_RATE / sample_rate)
        resampled = resample(samples, n_out).astype(np.float32)
    else:
        resampled = samples.astype(np.float32)

    resampled = resampled / (np.max(np.abs(resampled)) + 1e-10)

    padded = np.pad(resampled, WINDOW_SIZE // 2, mode="constant")
    n_frames = 1 + (len(padded) - WINDOW_SIZE) // HOP_SIZE
    frames = np.zeros((n_frames, WINDOW_SIZE), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_SIZE
        frames[i] = padded[start:start + WINDOW_SIZE]

    frames -= np.mean(frames, axis=1, keepdims=True)
    frame_std = np.std(frames, axis=1, keepdims=True)
    frame_std = np.maximum(frame_std, 1e-10)
    frames /= frame_std

    all_probs = []
    for batch_start in range(0, n_frames, batch_size):
        batch = frames[batch_start:batch_start + batch_size]
        x = mx.array(batch[:, np.newaxis, :])
        probs = model(x)
        mx.eval(probs)
        all_probs.append(np.array(probs))

    probs = np.concatenate(all_probs, axis=0)

    confidence = np.max(probs, axis=1)

    bins = np.argmax(probs, axis=1)
    cents = bins * CENTS_PER_BIN + 1997.3794084376191
    pitch = 10.0 * 2.0 ** (cents / 1200.0)

    duration = len(samples) / sample_rate
    times = np.linspace(0, duration, n_frames)

    return PitchResult(
        pitch=pitch.astype(np.float32),
        confidence=confidence.astype(np.float32),
        times=times,
        sample_rate=sample_rate,
    )
