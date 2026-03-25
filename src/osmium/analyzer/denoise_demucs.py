import numpy as np


def demucs_separate(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    try:
        import demucs.separate
    except ImportError:
        raise ImportError(
            "Demucs not installed. Install with: uv pip install -e '.[demucs]'"
        )
    import sys
    import tempfile
    import soundfile as sf

    sys.stderr.write("Loading htdemucs model (may download ~1GB on first run)...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.wav"
        sf.write(input_path, samples, sample_rate)

        demucs.separate.main([
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", tmpdir,
            "--segment", "7",
            input_path,
        ])

        vocals_path = f"{tmpdir}/htdemucs/input/vocals.wav"
        vocals, _ = sf.read(vocals_path, dtype="float32")

    if vocals.ndim > 1:
        vocals = vocals.mean(axis=1)

    if sample_rate != 44100:
        from scipy.signal import resample_poly
        from math import gcd
        up = sample_rate
        down = 44100
        g = gcd(up, down)
        vocals = resample_poly(vocals, up // g, down // g).astype(np.float32)

    return vocals.astype(np.float32)
