import numpy as np
from osmium.analyzer.mimi import MimiCodes, MIMI_CHUNK_SIZE
from dataclasses import dataclass


@dataclass
class ImportanceMap:
    scores: np.ndarray
    times: np.ndarray
    frame_rate: float
    duration: float


def compute_importance(
    codes: MimiCodes,
    samples: np.ndarray,
    sample_rate: int = 24000,
    weight_transition: float = 0.35,
    weight_energy: float = 0.30,
    weight_multi_cb: float = 0.35,
) -> ImportanceMap:
    cb = codes.codes
    n_frames = len(cb)

    transitions = np.zeros(n_frames)
    if n_frames > 1:
        transitions[1:] = (cb[1:, 0] != cb[:-1, 0]).astype(float)

    frame_energy = np.array([
        np.sqrt(np.mean(samples[i * MIMI_CHUNK_SIZE:(i + 1) * MIMI_CHUNK_SIZE] ** 2))
        for i in range(n_frames)
    ])
    energy_max = frame_energy.max()
    if energy_max > 0:
        frame_energy = frame_energy / energy_max

    multi_cb_change = np.zeros(n_frames)
    if n_frames > 1 and cb.shape[1] > 1:
        for j in range(1, n_frames):
            multi_cb_change[j] = np.mean(cb[j] != cb[j - 1])

    importance = (
        weight_transition * transitions
        + weight_energy * frame_energy
        + weight_multi_cb * multi_cb_change
    )

    importance = np.clip(importance, 0.0, 1.0)

    times = np.arange(n_frames) / codes.frame_rate

    return ImportanceMap(
        scores=importance,
        times=times,
        frame_rate=codes.frame_rate,
        duration=codes.duration,
    )


def resample_importance(imp: ImportanceMap, resolution_s: float) -> ImportanceMap:
    n_out = max(1, int(imp.duration / resolution_s))
    out_times = np.linspace(0, imp.duration, n_out)
    out_scores = np.interp(out_times, imp.times, imp.scores)
    return ImportanceMap(
        scores=out_scores,
        times=out_times,
        frame_rate=1.0 / resolution_s,
        duration=imp.duration,
    )
