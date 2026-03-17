import numpy as np
from osmium.analyzer.crepe_mlx import predict as crepe_predict, PitchResult


def detect_gci(
    samples: np.ndarray,
    sample_rate: int = 24000,
    confidence_threshold: float = 0.5,
    method: str = "zfr",
) -> tuple[np.ndarray, PitchResult]:
    pitch_result = crepe_predict(samples, sample_rate, capacity="tiny")

    if method == "zfr":
        raw_crossings = detect_gci_zfr(samples, sample_rate)
        gcis = _select_gcis_with_f0(
            raw_crossings, pitch_result, sample_rate, confidence_threshold,
        )
    else:
        residual = _lpc_residual(samples, 12, sample_rate)
        gcis = _find_gcis_lp(
            residual, samples,
            pitch_result.pitch, pitch_result.confidence,
            sample_rate, confidence_threshold,
        )

    return gcis, pitch_result


def _select_gcis_with_f0(
    crossings: np.ndarray,
    pitch_result: PitchResult,
    sample_rate: int,
    confidence_threshold: float,
) -> np.ndarray:
    if len(crossings) == 0:
        return crossings

    crepe_hop = int(sample_rate * 0.01)
    n_samples = len(pitch_result.pitch) * crepe_hop
    gcis = []
    pos = 0

    def _f0_at(sample_pos: int) -> tuple[float, float]:
        idx = min(sample_pos // crepe_hop, len(pitch_result.pitch) - 1)
        idx = max(0, idx)
        return float(pitch_result.pitch[idx]), float(pitch_result.confidence[idx])

    def _find_nearest_crossing(target: int, radius: int) -> int | None:
        lo = np.searchsorted(crossings, target - radius)
        hi = np.searchsorted(crossings, target + radius, side='right')
        if lo >= hi:
            return None
        region = crossings[lo:hi]
        best_idx = np.argmin(np.abs(region - target))
        return int(region[best_idx])

    while pos < n_samples:
        f0, conf = _f0_at(pos)

        if conf < confidence_threshold or f0 < 50:
            if not gcis or pos - gcis[-1] > crepe_hop:
                gcis.append(pos)
            pos += crepe_hop
            continue

        period = int(sample_rate / f0)
        search_radius = int(period * 0.4)

        if not gcis:
            c = _find_nearest_crossing(pos, search_radius)
            if c is not None:
                gcis.append(c)
                pos = c + period
            else:
                gcis.append(pos)
                pos += period
            continue

        expected = gcis[-1] + period
        c = _find_nearest_crossing(expected, search_radius)
        if c is not None:
            gcis.append(c)
            pos = c + period
        else:
            gcis.append(expected)
            pos = expected + period

    return np.array(gcis, dtype=np.int64)


def detect_gci_zfr(
    samples: np.ndarray,
    sample_rate: int = 24000,
    trend_window_ms: float = 10.0,
) -> np.ndarray:
    from scipy.signal import lfilter

    x = np.diff(samples.astype(np.float64), prepend=0.0)

    y = np.asarray(lfilter([1.0], [1.0, -2.0, 1.0], x))

    half_win = int(trend_window_ms * sample_rate / 1000.0)
    if half_win < 1:
        half_win = 1
    win = 2 * half_win + 1

    y = _remove_trend(y, win, half_win)
    zfr_signal = _remove_trend(y, win, half_win)

    crossings = np.where(
        (zfr_signal[:-1] <= 0) & (zfr_signal[1:] > 0)
    )[0] + 1

    min_spacing = int(sample_rate / 500)
    if len(crossings) > 1:
        filtered = [crossings[0]]
        for c in crossings[1:]:
            if c - filtered[-1] >= min_spacing:
                filtered.append(c)
        crossings = np.array(filtered, dtype=np.int64)

    return crossings


def _remove_trend(y: np.ndarray, win: int, half_win: int) -> np.ndarray:
    cumsum = np.cumsum(np.concatenate(([0.0], y)))
    mean = (cumsum[win:] - cumsum[:-win]) / win
    trend = np.concatenate((
        np.full(half_win, mean[0]),
        mean,
        np.full(len(y) - half_win - len(mean), mean[-1]),
    ))
    return y - trend


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


def _find_gcis_lp(
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
