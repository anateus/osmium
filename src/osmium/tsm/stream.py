import numpy as np
from osmium.tsm.phase_vocoder import phase_vocoder_stretch


def process_streaming(
    audio_chunks,
    speed: float,
    window_size: int = 2048,
    sample_rate: int = 24000,
    overlap_seconds: float = 0.1,
):
    overlap_samples = int(overlap_seconds * sample_rate)
    crossfade = np.linspace(0, 1, overlap_samples, dtype=np.float32)

    prev_tail = None

    for chunk in audio_chunks:
        samples = chunk.samples

        stretched = phase_vocoder_stretch(
            samples, speed=speed, window_size=window_size, sample_rate=sample_rate,
        )

        out_overlap = int(overlap_samples / speed)

        if prev_tail is not None and len(prev_tail) > 0 and len(stretched) > 0:
            xfade_len = min(len(prev_tail), out_overlap, len(stretched))
            if xfade_len > 0:
                fade = np.linspace(0, 1, xfade_len, dtype=np.float32)
                stretched[:xfade_len] = (
                    prev_tail[-xfade_len:] * (1 - fade) + stretched[:xfade_len] * fade
                )
                output = stretched
            else:
                output = stretched
        else:
            output = stretched

        if len(output) > out_overlap:
            prev_tail = output[-out_overlap:].copy()
            yield output[:-out_overlap]
        else:
            prev_tail = output.copy()

    if prev_tail is not None and len(prev_tail) > 0:
        yield prev_tail
