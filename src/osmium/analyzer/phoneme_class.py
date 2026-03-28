import numpy as np

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
