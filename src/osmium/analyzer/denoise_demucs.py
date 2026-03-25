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
    import sys
    import torch

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
