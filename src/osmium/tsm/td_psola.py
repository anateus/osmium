import numpy as np


def _snap_to_zero_crossing(samples: np.ndarray, pos: int, search_range: int = 50) -> int:
    lo = max(1, pos - search_range)
    hi = min(len(samples) - 1, pos + search_range)
    for j in range(lo, hi):
        if samples[j - 1] <= 0 < samples[j]:
            return j
    return pos


def pitch_to_marks(
    pitch: np.ndarray,
    confidence: np.ndarray,
    sample_rate: int = 24000,
    hop_size: int = 240,
    confidence_threshold: float = 0.5,
    samples: np.ndarray | None = None,
) -> np.ndarray:
    marks = []
    pos = 0.0

    for i in range(len(pitch)):
        f0 = pitch[i]
        conf = confidence[i]

        if conf < confidence_threshold or f0 < 50.0:
            frame_center = i * hop_size
            if not marks or frame_center - marks[-1] > hop_size:
                marks.append(frame_center)
            continue

        period = sample_rate / f0

        frame_start = i * hop_size
        frame_end = frame_start + hop_size

        if not marks:
            pos = float(frame_start)
        elif pos < frame_start:
            pos = float(frame_start)

        while pos < frame_end:
            mark = int(round(pos))
            if samples is not None:
                search = int(period * 0.25)
                mark = _snap_to_zero_crossing(samples, mark, search)
            if mark >= 0 and (not marks or mark > marks[-1]):
                marks.append(mark)
            pos += period

    return np.array(marks, dtype=np.int64)


def td_psola_stretch(
    samples: np.ndarray,
    pitch_marks: np.ndarray,
    speed: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    if speed == 1.0:
        return samples.copy()
    if len(pitch_marks) < 2:
        return samples[:max(1, int(len(samples) / speed))].astype(np.float32)

    output_length = int(len(samples) / speed)
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)

    synthesis_marks = (pitch_marks.astype(np.float64) / speed).astype(np.int64)
    synthesis_marks = synthesis_marks[synthesis_marks < output_length]
    n_use = min(len(pitch_marks), len(synthesis_marks))

    periods = np.zeros(len(pitch_marks))
    for i in range(len(pitch_marks)):
        if i > 0 and i < len(pitch_marks) - 1:
            periods[i] = (pitch_marks[i + 1] - pitch_marks[i - 1]) / 2.0
        elif i > 0:
            periods[i] = pitch_marks[i] - pitch_marks[i - 1]
        elif len(pitch_marks) > 1:
            periods[i] = pitch_marks[1] - pitch_marks[0]
        else:
            periods[i] = 480

    for i in range(n_use):
        am = int(pitch_marks[i])
        sm = int(synthesis_marks[i])

        win_half = int(periods[i])
        if win_half < 2:
            continue

        src_start = am - win_half
        src_end = am + win_half
        if src_start < 0 or src_end > len(samples):
            continue

        win_size = 2 * win_half
        window = np.hanning(win_size).astype(np.float64)
        frame = samples[src_start:src_end] * window

        dst_start = sm - win_half
        dst_end = sm + win_half
        if dst_start < 0:
            trim = -dst_start
            frame = frame[trim:]
            window = window[trim:]
            dst_start = 0
        if dst_end > output_length:
            trim = dst_end - output_length
            frame = frame[:len(frame) - trim]
            window = window[:len(window) - trim]
            dst_end = output_length

        if len(frame) == 0:
            continue

        output[dst_start:dst_end] += frame
        window_sum[dst_start:dst_end] += window

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    low_energy = ~nonzero
    if low_energy.any():
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(nonzero, iterations=48)
        fill_mask = dilated & low_energy
        if fill_mask.any():
            indices = np.arange(output_length)
            valid_idx = indices[nonzero]
            valid_vals = output[nonzero]
            if len(valid_idx) > 0:
                output[fill_mask] = np.interp(indices[fill_mask], valid_idx, valid_vals)

    return output.astype(np.float32)


def td_psola_variable_rate(
    samples: np.ndarray,
    pitch_marks: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    if len(pitch_marks) < 2:
        avg_rate = float(np.mean(rate_curve))
        return samples[:max(1, int(len(samples) / avg_rate))].astype(np.float32)

    synthesis_positions = []
    out_pos = 0.0
    for i in range(len(pitch_marks)):
        am = pitch_marks[i]
        t = am / sample_rate
        local_rate = float(np.interp(t, rate_times, rate_curve))
        synthesis_positions.append(int(round(out_pos)))

        if i < len(pitch_marks) - 1:
            input_gap = pitch_marks[i + 1] - am
            out_gap = input_gap / local_rate
            out_pos += out_gap
        else:
            out_pos += 1

    synthesis_marks = np.array(synthesis_positions, dtype=np.int64)
    output_length = int(synthesis_marks[-1]) + int(sample_rate * 0.1)
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)

    n_use = len(pitch_marks)
    periods = np.zeros(n_use)
    for i in range(n_use):
        if i > 0 and i < n_use - 1:
            periods[i] = (pitch_marks[i + 1] - pitch_marks[i - 1]) / 2.0
        elif i > 0:
            periods[i] = pitch_marks[i] - pitch_marks[i - 1]
        elif n_use > 1:
            periods[i] = pitch_marks[1] - pitch_marks[0]
        else:
            periods[i] = 480

    for i in range(n_use):
        am = int(pitch_marks[i])
        sm = int(synthesis_marks[i])

        win_half = int(periods[i])
        if win_half < 2:
            continue

        src_start = am - win_half
        src_end = am + win_half
        if src_start < 0 or src_end > len(samples):
            continue

        win_size = 2 * win_half
        window = np.hanning(win_size).astype(np.float64)
        frame = samples[src_start:src_end] * window

        dst_start = sm - win_half
        dst_end = sm + win_half
        if dst_start < 0:
            trim = -dst_start
            frame = frame[trim:]
            window = window[trim:]
            dst_start = 0
        if dst_end > output_length:
            trim = dst_end - output_length
            frame = frame[:len(frame) - trim]
            window = window[:len(window) - trim]
            dst_end = output_length

        if len(frame) == 0:
            continue

        output[dst_start:dst_end] += frame
        window_sum[dst_start:dst_end] += window

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    low_energy = ~nonzero
    if low_energy.any():
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(nonzero, iterations=48)
        fill_mask = dilated & low_energy
        if fill_mask.any():
            indices = np.arange(output_length)
            valid_idx = indices[nonzero]
            valid_vals = output[nonzero]
            if len(valid_idx) > 0:
                output[fill_mask] = np.interp(indices[fill_mask], valid_idx, valid_vals)

    used = np.max(np.where(nonzero)[0]) + 1 if nonzero.any() else 0
    return output[:used].astype(np.float32)
