import numpy as np
from osmium.analyzer.crepe_mlx import predict as crepe_predict, PitchResult


def detect_gci(
    samples: np.ndarray,
    sample_rate: int = 24000,
    lpc_order: int = 12,
    confidence_threshold: float = 0.5,
) -> tuple[np.ndarray, PitchResult]:
    pitch_result = crepe_predict(samples, sample_rate, capacity="tiny")

    residual = _lpc_residual(samples, lpc_order, sample_rate)

    gcis = _find_gcis(
        residual, samples,
        pitch_result.pitch, pitch_result.confidence,
        sample_rate, confidence_threshold,
    )

    return gcis, pitch_result


def _lpc_residual(samples: np.ndarray, order: int, sample_rate: int) -> np.ndarray:
    frame_size = int(0.025 * sample_rate)
    hop = int(0.005 * sample_rate)
    residual = np.zeros_like(samples)

    for start in range(0, len(samples) - frame_size, hop):
        frame = samples[start:start + frame_size].copy()
        frame *= np.hanning(len(frame))

        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame) - 1:]
        r = r[:order + 1]

        if r[0] < 1e-10:
            continue

        a = _levinson_durbin(r, order)

        end = min(start + frame_size, len(samples))
        seg = samples[start:end]
        res = np.zeros(len(seg))
        for i in range(len(seg)):
            res[i] = seg[i]
            for j in range(1, min(order + 1, i + 1)):
                res[i] += a[j] * seg[i - j]

        win_start = start
        win_end = start + len(res)
        overlap = min(win_end, len(residual)) - win_start
        if overlap > 0:
            residual[win_start:win_start + overlap] = res[:overlap]

    return residual


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -(r[i] + acc) / (e + 1e-10)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e = e * (1 - k * k)
        if e < 1e-10:
            break

    return a


def _find_gcis(
    residual: np.ndarray,
    samples: np.ndarray,
    pitch: np.ndarray,
    confidence: np.ndarray,
    sample_rate: int,
    confidence_threshold: float,
) -> np.ndarray:
    crepe_hop = int(sample_rate * 0.01)
    gcis = []
    pos = 0

    for i in range(len(pitch)):
        f0 = pitch[i]
        conf = confidence[i]
        frame_center = i * crepe_hop

        if conf < confidence_threshold or f0 < 50:
            target = frame_center
            if not gcis or target - gcis[-1] > crepe_hop:
                gcis.append(min(target, len(samples) - 1))
            continue

        period = int(sample_rate / f0)
        half_period = period // 2

        if not gcis:
            pos = frame_center

        while pos < min(frame_center + crepe_hop, len(samples)):
            search_lo = max(0, pos - half_period)
            search_hi = min(len(residual), pos + half_period)

            if search_hi <= search_lo:
                pos += period
                continue

            region = residual[search_lo:search_hi]
            peak_offset = np.argmax(np.abs(region))
            gci_pos = search_lo + peak_offset

            if samples is not None and gci_pos > 0 and gci_pos < len(samples) - 1:
                search_r = min(period // 4, 30)
                lo = max(1, gci_pos - search_r)
                hi = min(len(samples) - 1, gci_pos + search_r)
                for j in range(lo, hi):
                    if samples[j - 1] <= 0 < samples[j]:
                        gci_pos = j
                        break

            if not gcis or gci_pos > gcis[-1]:
                gcis.append(gci_pos)

            pos = gci_pos + period

    return np.array(gcis, dtype=np.int64)
