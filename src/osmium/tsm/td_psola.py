import numpy as np


def pitch_to_marks(
    pitch: np.ndarray,
    confidence: np.ndarray,
    sample_rate: int = 24000,
    hop_size: int = 240,
    confidence_threshold: float = 0.5,
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

    synthesis_marks = (pitch_marks / speed).astype(np.int64)
    synthesis_marks = synthesis_marks[synthesis_marks < output_length]

    n_use = min(len(pitch_marks), len(synthesis_marks))

    for i in range(n_use):
        am = pitch_marks[i]
        sm = synthesis_marks[i]

        if i > 0:
            half_left = am - pitch_marks[i - 1]
        else:
            half_left = pitch_marks[1] - pitch_marks[0] if len(pitch_marks) > 1 else 512
        if i < n_use - 1:
            half_right = pitch_marks[i + 1] - am
        else:
            half_right = half_left

        win_size = half_left + half_right
        if win_size < 4:
            continue

        src_start = am - half_left
        src_end = am + half_right
        if src_start < 0 or src_end > len(samples):
            continue

        window = np.hanning(win_size).astype(np.float64)
        frame = samples[src_start:src_end] * window

        dst_start = sm - half_left
        dst_end = sm + half_right
        if dst_start < 0 or dst_end > output_length:
            continue

        output[dst_start:dst_end] += frame
        window_sum[dst_start:dst_end] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

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

    for i in range(len(pitch_marks)):
        am = pitch_marks[i]
        sm = synthesis_marks[i]

        if i > 0:
            half_left = am - pitch_marks[i - 1]
        else:
            half_left = pitch_marks[1] - pitch_marks[0] if len(pitch_marks) > 1 else 512
        if i < len(pitch_marks) - 1:
            half_right = pitch_marks[i + 1] - am
        else:
            half_right = half_left

        win_size = half_left + half_right
        if win_size < 4:
            continue

        src_start = am - half_left
        src_end = am + half_right
        if src_start < 0 or src_end > len(samples):
            continue

        window = np.hanning(win_size).astype(np.float64)
        frame = samples[src_start:src_end] * window

        dst_start = sm - half_left
        dst_end = sm + half_right
        if dst_start < 0 or dst_end > output_length:
            continue

        output[dst_start:dst_end] += frame
        window_sum[dst_start:dst_end] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    used = np.max(np.where(nonzero)[0]) + 1 if nonzero.any() else 0
    return output[:used].astype(np.float32)
