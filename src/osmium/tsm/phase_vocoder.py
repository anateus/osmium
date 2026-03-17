import numpy as np


def _hann_window(size: int) -> np.ndarray:
    return np.hanning(size).astype(np.float64)


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
    hop_synthesis = int(hop_analysis / speed)

    window = _hann_window(window_size)
    half_win = window_size // 2

    samples = np.pad(samples, (half_win, half_win), mode="constant")

    n_frames = 1 + (len(samples) - window_size) // hop_analysis
    if n_frames < 2:
        return samples[:int(len(samples) / speed)]

    freq_bins = window_size // 2 + 1
    expected_phase_advance = 2.0 * np.pi * hop_analysis * np.arange(freq_bins) / window_size

    output_length = int((n_frames - 1) * hop_synthesis + window_size)
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)

    prev_phase = None
    prev_synth_phase = None

    for i in range(n_frames):
        start = i * hop_analysis
        frame = samples[start:start + window_size] * window

        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if prev_phase is None:
            synth_phase = phase
        else:
            phase_diff = phase - prev_phase - expected_phase_advance
            phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
            true_freq = expected_phase_advance + phase_diff
            synth_phase = prev_synth_phase + true_freq * (hop_synthesis / hop_analysis)

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
    expected_len = int((len(samples) - 2 * half_win) / speed)
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

    samples = np.pad(samples, (half_win, half_win), mode="constant")

    n_frames = 1 + (len(samples) - window_size) // hop_analysis
    if n_frames < 2:
        avg_rate = np.mean(rate_curve)
        return samples[:int(len(samples) / avg_rate)]

    freq_bins = window_size // 2 + 1
    expected_phase_advance = 2.0 * np.pi * hop_analysis * np.arange(freq_bins) / window_size

    output_chunks = []
    prev_phase = None
    prev_synth_phase = None
    output_pos = 0

    max_output = int(len(samples) / np.min(rate_curve) * 1.1)
    output = np.zeros(max_output, dtype=np.float64)
    window_sum = np.zeros(max_output, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_analysis
        frame_time = start / sample_rate
        local_rate = np.interp(frame_time, rate_times, rate_curve)
        hop_synthesis = max(1, int(hop_analysis / local_rate))

        frame = samples[start:start + window_size] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if prev_phase is None:
            synth_phase = phase
        else:
            phase_diff = phase - prev_phase - expected_phase_advance
            phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
            true_freq = expected_phase_advance + phase_diff
            synth_phase = prev_synth_phase + true_freq * (hop_synthesis / hop_analysis)

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
