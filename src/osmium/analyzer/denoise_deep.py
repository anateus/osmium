import numpy as np


def deep_filter(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(
        y=samples, sr=sample_rate,
        stationary=False,
        prop_decrease=0.95,
    ).astype(np.float32)
