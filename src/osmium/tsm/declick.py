import numpy as np


def declick(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: float = 2.0,
    median_window_ms: float = 40.0,
    threshold: float = 5.0,
    attenuation: float = 0.5,
    crossfade_ms: float = 2.0,
) -> np.ndarray:
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(audio) // frame_samples
    if n_frames < 5:
        return audio

    crossfade_samples = max(1, int(sample_rate * crossfade_ms / 1000))

    trimmed_len = n_frames * frame_samples
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        energy[i] = np.mean(audio[start:end] ** 2)

    median_frames = max(3, int(median_window_ms / frame_ms))
    if median_frames % 2 == 0:
        median_frames += 1
    half = median_frames // 2

    click_frames = np.zeros(n_frames, dtype=bool)
    for i in range(half, n_frames - half):
        local_med = np.median(energy[i - half : i + half + 1])
        if local_med > 1e-10 and energy[i] > threshold * local_med:
            click_frames[i] = True

    if not click_frames.any():
        return audio

    result = audio.copy()
    gain = np.ones(len(audio), dtype=np.float32)

    for i in range(n_frames):
        if click_frames[i]:
            start = i * frame_samples
            end = min(start + frame_samples, len(audio))
            gain[start:end] = attenuation

    for i in range(n_frames):
        if click_frames[i]:
            start = i * frame_samples
            end = min(start + frame_samples, len(audio))

            fade_in_start = max(0, start - crossfade_samples)
            if fade_in_start < start:
                t = np.linspace(1.0, attenuation, start - fade_in_start)
                existing = gain[fade_in_start:start]
                gain[fade_in_start:start] = np.minimum(existing, t)

            fade_out_end = min(len(audio), end + crossfade_samples)
            if fade_out_end > end:
                t = np.linspace(attenuation, 1.0, fade_out_end - end)
                existing = gain[end:fade_out_end]
                gain[end:fade_out_end] = np.minimum(existing, t)

    result[:trimmed_len] = audio[:trimmed_len] * gain[:trimmed_len]
    return result
