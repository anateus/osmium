import subprocess
import numpy as np
from pathlib import Path

FORMAT_MAP = {
    ".mp3": ("libmp3lame", ["-q:a", "2"]),
    ".m4a": ("aac", ["-b:a", "128k"]),
    ".wav": ("pcm_s16le", []),
    ".flac": ("flac", []),
}


def encode(samples: np.ndarray, sample_rate: int, output_path: str) -> None:
    ext = Path(output_path).suffix.lower()
    if ext not in FORMAT_MAP:
        raise ValueError(f"Unsupported output format: {ext}. Supported: {list(FORMAT_MAP.keys())}")

    codec, extra_args = FORMAT_MAP[ext]
    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-i", "-",
        "-acodec", codec,
        *extra_args,
        output_path,
    ]
    subprocess.run(cmd, input=samples.astype(np.float32).tobytes(), check=True)


def encode_pcm_stdout(samples: np.ndarray) -> bytes:
    return samples.astype(np.float32).tobytes()
