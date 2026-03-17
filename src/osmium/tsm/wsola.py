import numpy as np


def wsola_stretch(
    samples: np.ndarray,
    speed: float,
    window_size: int = 1024,
    tolerance: int = 512,
) -> np.ndarray:
    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    if speed == 1.0:
        return samples.copy()

    hop_analysis = window_size // 2
    hop_synthesis = int(hop_analysis / speed)
    window = np.hanning(window_size).astype(np.float64)

    output_length = int(len(samples) / speed)
    output = np.zeros(output_length + window_size, dtype=np.float64)
    window_sum = np.zeros(output_length + window_size, dtype=np.float64)

    analysis_pos = 0
    synthesis_pos = 0
    offset = 0

    while analysis_pos + offset + window_size <= len(samples) and synthesis_pos + window_size <= len(output):
        actual_start = analysis_pos + offset
        frame = samples[actual_start:actual_start + window_size] * window

        output[synthesis_pos:synthesis_pos + window_size] += frame
        window_sum[synthesis_pos:synthesis_pos + window_size] += window ** 2

        analysis_pos += hop_analysis
        synthesis_pos += hop_synthesis

        if analysis_pos + window_size + tolerance > len(samples):
            break

        natural_start = analysis_pos
        best_offset = 0
        best_corr = -1.0

        search_lo = max(0, natural_start - tolerance)
        search_hi = min(len(samples) - window_size, natural_start + tolerance)

        if synthesis_pos >= hop_synthesis and synthesis_pos + window_size <= len(output):
            ref_start = synthesis_pos - hop_synthesis
            ref_end = min(ref_start + window_size, len(output))
            ref_len = ref_end - ref_start
            if ref_len > 0:
                ref = output[ref_start:ref_end]
                ref_norm = np.sqrt(np.sum(ref ** 2)) + 1e-10

                for test_pos in range(search_lo, search_hi + 1, 4):
                    candidate = samples[test_pos:test_pos + ref_len]
                    if len(candidate) < ref_len:
                        continue
                    corr = np.sum(ref * candidate * window[:ref_len]) / (
                        ref_norm * (np.sqrt(np.sum((candidate * window[:ref_len]) ** 2)) + 1e-10)
                    )
                    if corr > best_corr:
                        best_corr = corr
                        best_offset = test_pos - natural_start

        offset = best_offset

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    return output[:output_length].astype(np.float32)


def variable_rate_wsola(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    window_size: int = 1024,
    tolerance: int = 512,
    sample_rate: int = 24000,
) -> np.ndarray:
    hop_analysis = window_size // 2
    window = np.hanning(window_size).astype(np.float64)

    safe_min = max(0.1, float(np.min(rate_curve)))
    max_output = int(len(samples) / safe_min * 1.2)
    output = np.zeros(max_output, dtype=np.float64)
    window_sum = np.zeros(max_output, dtype=np.float64)

    analysis_pos = 0
    synthesis_pos = 0
    offset = 0

    while analysis_pos + offset + window_size <= len(samples) and synthesis_pos + window_size <= max_output:
        actual_start = analysis_pos + offset
        frame = samples[actual_start:actual_start + window_size] * window

        output[synthesis_pos:synthesis_pos + window_size] += frame
        window_sum[synthesis_pos:synthesis_pos + window_size] += window ** 2

        frame_time = analysis_pos / sample_rate
        local_rate = float(np.interp(frame_time, rate_times, rate_curve))
        hop_synthesis = max(1, int(hop_analysis / local_rate))

        analysis_pos += hop_analysis
        synthesis_pos += hop_synthesis

        if analysis_pos + window_size + tolerance > len(samples):
            break

        natural_start = analysis_pos
        best_offset = 0
        best_corr = -1.0

        search_lo = max(0, natural_start - tolerance)
        search_hi = min(len(samples) - window_size, natural_start + tolerance)

        if synthesis_pos >= hop_synthesis and synthesis_pos + window_size <= max_output:
            ref_start = synthesis_pos - hop_synthesis
            ref_end = min(ref_start + window_size, max_output)
            ref_len = ref_end - ref_start
            if ref_len > 0:
                ref = output[ref_start:ref_end]
                ref_norm = np.sqrt(np.sum(ref ** 2)) + 1e-10

                for test_pos in range(search_lo, search_hi + 1, 4):
                    candidate = samples[test_pos:test_pos + ref_len]
                    if len(candidate) < ref_len:
                        continue
                    corr = np.sum(ref * candidate * window[:ref_len]) / (
                        ref_norm * (np.sqrt(np.sum((candidate * window[:ref_len]) ** 2)) + 1e-10)
                    )
                    if corr > best_corr:
                        best_corr = corr
                        best_offset = test_pos - natural_start

        offset = best_offset

    final_len = synthesis_pos
    output = output[:final_len]
    window_sum = window_sum[:final_len]

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    return output.astype(np.float32)
