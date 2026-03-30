import numpy as np


def apply_room(audio: np.ndarray, sample_rate: int = 24000, rt60_ms: float = 80.0) -> np.ndarray:
    from scipy.signal import fftconvolve

    ir = _make_room_ir(sample_rate, rt60_ms)
    result = fftconvolve(audio, ir, mode="full")[: len(audio)].astype(np.float32)
    peak_orig = np.max(np.abs(audio)) + 1e-10
    peak_result = np.max(np.abs(result)) + 1e-10
    result *= peak_orig / peak_result
    return result


def apply_warm_dither(audio: np.ndarray, sample_rate: int = 24000, level_db: float = -60.0) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    rng = np.random.RandomState(hash(len(audio)) & 0xFFFFFFFF)
    noise = rng.randn(len(audio)).astype(np.float32)
    sos = butter(3, 1500, btype="lowpass", fs=sample_rate, output="sos")
    noise = sosfilt(sos, noise).astype(np.float32)
    level = 10 ** (level_db / 20)
    noise *= level / (np.std(noise) + 1e-10)
    return audio + noise


def post_process(
    audio: np.ndarray,
    sample_rate: int = 24000,
    declick: bool = True,
    declick_threshold: float = 5.0,
    room: bool = True,
    room_rt60_ms: float = 80.0,
    warm_dither: bool = True,
    warm_dither_db: float = -60.0,
) -> np.ndarray:
    if declick:
        from osmium.tsm.declick import declick as _declick
        audio = _declick(audio, sample_rate=sample_rate, threshold=declick_threshold)
    if room:
        audio = apply_room(audio, sample_rate, rt60_ms=room_rt60_ms)
    if warm_dither:
        audio = apply_warm_dither(audio, sample_rate, level_db=warm_dither_db)
    return audio


_room_ir_cache = {}


def _make_room_ir(sr: int, rt60_ms: float) -> np.ndarray:
    key = (sr, rt60_ms)
    if key in _room_ir_cache:
        return _room_ir_cache[key]

    length = int(sr * rt60_ms / 1000)
    ir = np.zeros(length, dtype=np.float32)
    ir[0] = 1.0

    rng = np.random.RandomState(42)
    for i in range(6):
        delay = int(sr * (1.0 + rng.uniform(1, 15)) / 1000)
        if delay < length:
            ir[delay] += 0.15 * (0.6 ** i) * rng.choice([-1, 1])

    t = np.arange(length, dtype=np.float32) / sr
    decay = 0.02 * np.exp(-6.9 * t / (rt60_ms / 1000))
    ir += rng.randn(length).astype(np.float32) * decay
    ir /= np.max(np.abs(ir))

    _room_ir_cache[key] = ir
    return ir
