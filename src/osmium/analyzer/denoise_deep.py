import numpy as np


def deep_filter(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> np.ndarray:
    from scipy.signal import resample_poly

    try:
        from df import enhance, init_df
    except ImportError:
        raise ImportError(
            "DeepFilterNet not installed. Install with: uv pip install -e '.[denoise]'"
        )

    model, df_state, _ = init_df()

    if sample_rate == 24000:
        upsampled = resample_poly(samples, 2, 1).astype(np.float32)
    elif sample_rate == 48000:
        upsampled = samples
    else:
        ratio = 48000 / sample_rate
        up = int(ratio * 1000)
        down = 1000
        from math import gcd
        g = gcd(up, down)
        upsampled = resample_poly(samples, up // g, down // g).astype(np.float32)

    import torch
    audio_tensor = torch.from_numpy(upsampled).unsqueeze(0)
    enhanced = enhance(model, df_state, audio_tensor)
    enhanced_np = enhanced.squeeze().numpy()

    if sample_rate == 48000:
        return enhanced_np.astype(np.float32)
    elif sample_rate == 24000:
        return resample_poly(enhanced_np, 1, 2).astype(np.float32)
    else:
        ratio = sample_rate / 48000
        up = int(ratio * 1000)
        down = 1000
        from math import gcd
        g = gcd(up, down)
        return resample_poly(enhanced_np, up // g, down // g).astype(np.float32)
