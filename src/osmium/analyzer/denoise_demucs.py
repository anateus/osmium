import numpy as np

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError(
            "Demucs not installed. Install with: uv pip install -e '.[demucs]'"
        )
    import sys
    sys.stderr.write("Loading htdemucs model (may download ~80MB on first run)...\n")
    _model = get_model("htdemucs")
    _model.eval()
    return _model


def demucs_separate(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    from demucs.apply import apply_model
    import torch
    import torchaudio

    model = _load_model()

    mono = torch.from_numpy(samples.copy()).float()
    if mono.ndim == 1:
        mono = mono.unsqueeze(0)
    audio = mono.expand(model.audio_channels, -1)

    if sample_rate != model.samplerate:
        audio = torchaudio.functional.resample(audio, sample_rate, model.samplerate)

    ref = audio.mean(0)
    audio = (audio - ref.mean()) / ref.std()

    sources = apply_model(model, audio.unsqueeze(0), segment=7)[0]
    sources = sources * ref.std() + ref.mean()

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]

    if sample_rate != model.samplerate:
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sample_rate)

    result = vocals.mean(0).numpy().astype(np.float32)
    return result
