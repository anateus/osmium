import numpy as np
from osmium.analyzer.importance import ImportanceMap

MMS_FA_LABELS = list("-aienuotsrmkldghybpwcvjzf'qx*")

LABEL_TO_CLASS = {
    "t": "plosive", "d": "plosive", "k": "plosive",
    "g": "plosive", "b": "plosive", "p": "plosive",
    "c": "plosive", "'": "plosive",
    "s": "fricative", "z": "fricative", "f": "fricative",
    "v": "fricative", "h": "fricative", "x": "fricative",
    "m": "nasal", "n": "nasal",
    "l": "liquid_glide", "r": "liquid_glide", "w": "liquid_glide",
    "y": "liquid_glide", "j": "liquid_glide",
    "a": "vowel", "e": "vowel", "i": "vowel",
    "o": "vowel", "u": "vowel",
    "q": "plosive", "*": "plosive",
}

PHONEME_CLASS_FLOORS = {
    "plosive": 0.50,
    "fricative": 0.42,
    "nasal": 0.35,
    "liquid_glide": 0.25,
    "vowel": 0.10,
    "silence": 0.02,
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
