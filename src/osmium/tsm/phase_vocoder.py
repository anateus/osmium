import numpy as np


def _hann_window(size: int) -> np.ndarray:
    return np.hanning(size).astype(np.float64)


def _peak_regions(magnitude: np.ndarray) -> np.ndarray:
    n = len(magnitude)
    peaks = np.zeros(n, dtype=bool)
    if n <= 1:
        peaks[:] = True
        return np.arange(n, dtype=np.intp)

    peaks[1:-1] = (magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])
    peaks[0] = magnitude[0] >= magnitude[1]
    peaks[-1] = magnitude[-1] >= magnitude[-2]

    peak_indices = np.where(peaks)[0]
    if len(peak_indices) == 0:
        return np.zeros(n, dtype=np.intp)

    all_bins = np.arange(n)
    insert_points = np.searchsorted(peak_indices, all_bins)
    insert_points = np.clip(insert_points, 0, len(peak_indices) - 1)

    regions = peak_indices[insert_points]
    left = np.clip(insert_points - 1, 0, len(peak_indices) - 1)
    left_peaks = peak_indices[left]
    use_left = np.abs(all_bins - left_peaks) < np.abs(all_bins - regions)
    regions[use_left] = left_peaks[use_left]

    return regions


def _identity_phase_lock(
    magnitude: np.ndarray,
    phase: np.ndarray,
    prev_phase: np.ndarray,
    prev_synth_phase: np.ndarray,
    expected_advance: np.ndarray,
    hop_ratio: float,
) -> np.ndarray:
    phase_diff = phase - prev_phase - expected_advance
    phase_diff -= 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
    inst_freq = expected_advance + phase_diff
    peak_phase = prev_synth_phase + inst_freq * hop_ratio

    regions = _peak_regions(magnitude)
    rotation = peak_phase[regions] - phase[regions]
    return phase + rotation


def _detect_transients(
    samples: np.ndarray,
    window_size: int,
    hop: int,
    threshold: float = 1.5,
) -> np.ndarray:
    n_frames = 1 + (len(samples) - window_size) // hop
    flux = np.zeros(n_frames)

    prev_mag = None
    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + window_size]
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)

        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = np.sum(np.maximum(0, diff)) / (np.sum(mag) + 1e-10)

        prev_mag = mag

    if flux.std() > 0:
        flux_norm = (flux - flux.mean()) / (flux.std() + 1e-10)
    else:
        flux_norm = flux

    return flux_norm > threshold


def phase_vocoder_stretch(
    samples: np.ndarray,
    speed: float,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    if speed == 1.0:
        return samples.copy()

    hop_analysis = window_size // 4
    hop_synthesis = max(1, int(hop_analysis / speed))
    hop_ratio = hop_synthesis / hop_analysis

    window = _hann_window(window_size)
    half_win = window_size // 2

    samples_padded = np.pad(samples, (half_win, half_win), mode="constant")

    n_frames = 1 + (len(samples_padded) - window_size) // hop_analysis
    if n_frames < 2:
        return samples[:max(1, int(len(samples) / speed))].astype(np.float32)

    transients = _detect_transients(samples_padded, window_size, hop_analysis)

    freq_bins = window_size // 2 + 1
    expected_advance = 2.0 * np.pi * hop_analysis * np.arange(freq_bins) / window_size

    output_length = (n_frames - 1) * hop_synthesis + window_size
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)

    prev_phase = None
    prev_synth_phase = None

    for i in range(n_frames):
        start = i * hop_analysis
        frame = samples_padded[start:start + window_size] * window

        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if transients[i]:
            synth_phase = phase.copy()
            prev_phase = None
            prev_synth_phase = None
        elif prev_phase is None:
            synth_phase = phase.copy()
        else:
            synth_phase = _identity_phase_lock(
                magnitude, phase, prev_phase, prev_synth_phase,
                expected_advance, hop_ratio,
            )

        prev_phase = phase
        prev_synth_phase = synth_phase

        synth_spectrum = magnitude * np.exp(1j * synth_phase)
        synth_frame = np.fft.irfft(synth_spectrum, n=window_size) * window

        out_start = i * hop_synthesis
        out_end = out_start + window_size
        output[out_start:out_end] += synth_frame
        window_sum[out_start:out_end] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    output = output[half_win:]
    expected_len = int(len(samples) / speed)
    output = output[:expected_len]

    return output.astype(np.float32)


def variable_rate_phase_vocoder(
    samples: np.ndarray,
    rate_curve: np.ndarray,
    rate_times: np.ndarray,
    window_size: int = 2048,
    sample_rate: int = 24000,
) -> np.ndarray:
    hop_analysis = window_size // 4
    window = _hann_window(window_size)
    half_win = window_size // 2

    samples_padded = np.pad(samples, (half_win, half_win), mode="constant")

    n_frames = 1 + (len(samples_padded) - window_size) // hop_analysis
    if n_frames < 2:
        avg_rate = float(np.mean(rate_curve))
        return samples[:max(1, int(len(samples) / avg_rate))].astype(np.float32)

    transients = _detect_transients(samples_padded, window_size, hop_analysis)

    freq_bins = window_size // 2 + 1
    expected_advance = 2.0 * np.pi * hop_analysis * np.arange(freq_bins) / window_size

    frame_rates = np.array([
        float(np.interp(i * hop_analysis / sample_rate, rate_times, rate_curve))
        for i in range(n_frames)
    ])
    hop_synth_float = hop_analysis / frame_rates
    hop_synth_arr = np.maximum(1, np.round(hop_synth_float)).astype(int)

    prev_phase = None
    prev_synth_phase = None
    output_pos = 0

    safe_min = max(0.1, float(np.min(rate_curve)))
    max_output = int(len(samples_padded) / safe_min * 1.2)
    output = np.zeros(max_output, dtype=np.float64)
    window_sum = np.zeros(max_output, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_analysis
        hop_synthesis = int(hop_synth_arr[i])
        hop_ratio = hop_synthesis / hop_analysis

        frame = samples_padded[start:start + window_size] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if transients[i]:
            synth_phase = phase.copy()
            prev_phase = None
            prev_synth_phase = None
        elif prev_phase is None:
            synth_phase = phase.copy()
        else:
            synth_phase = _identity_phase_lock(
                magnitude, phase, prev_phase, prev_synth_phase,
                expected_advance, hop_ratio,
            )

        prev_phase = phase
        prev_synth_phase = synth_phase

        synth_spectrum = magnitude * np.exp(1j * synth_phase)
        synth_frame = np.fft.irfft(synth_spectrum, n=window_size) * window

        out_start = output_pos
        out_end = out_start + window_size
        if out_end > max_output:
            break
        output[out_start:out_end] += synth_frame
        window_sum[out_start:out_end] += window ** 2
        output_pos += hop_synthesis

    final_len = output_pos + half_win
    output = output[:final_len]
    window_sum = window_sum[:final_len]

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    output = output[half_win:]
    return output.astype(np.float32)
