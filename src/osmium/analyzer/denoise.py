import numpy as np


def spectral_gate(
    samples: np.ndarray,
    sample_rate: int = 24000,
    stationary: bool = True,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(
        y=samples, sr=sample_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
    ).astype(np.float32)
