import numpy as np
from dataclasses import dataclass


@dataclass
class PhonemeImportance:
    scores: np.ndarray
    times: np.ndarray
    frame_rate: float
    duration: float


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    _model = bundle.get_model()
    _model.eval()
    _tokenizer = bundle.get_tokenizer()
    return _model, _tokenizer


def compute_phoneme_importance(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> PhonemeImportance:
    import torch
    import torchaudio

    model, tokenizer = _load_model()
    target_sr = 16000

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    with torch.no_grad():
        emission, _ = model(waveform)

    probs = torch.softmax(emission[0], dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    entropy = entropy.numpy()

    blank_prob = probs[:, 0].numpy()
    max_prob = probs.max(dim=-1).values.numpy()

    is_silence = blank_prob > 0.8
    is_transition = entropy > np.percentile(entropy, 70)

    plosive_like = (max_prob > 0.5) & (entropy > np.median(entropy))
    fricative_like = (blank_prob < 0.3) & (max_prob < 0.5)

    importance = np.zeros(len(entropy))

    importance[is_silence] = 0.1
    importance[~is_silence] = 0.5

    importance[is_transition] = np.maximum(importance[is_transition], 0.8)
    importance[plosive_like] = np.maximum(importance[plosive_like], 0.9)
    importance[fricative_like] = np.maximum(importance[fricative_like], 0.7)

    n_frames = len(entropy)
    duration = len(samples) / sample_rate
    times = np.linspace(0, duration, n_frames)

    return PhonemeImportance(
        scores=importance,
        times=times,
        frame_rate=n_frames / duration,
        duration=duration,
    )
