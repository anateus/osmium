import subprocess
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int


def decode(path: str, target_sr: int = 24000) -> AudioData:
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=target_sr)


def decode_streaming(path: str, chunk_seconds: float = 5.0, target_sr: int = 24000):
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    chunk_samples = int(chunk_seconds * target_sr)
    chunk_bytes = chunk_samples * 4

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.float32)
            yield AudioData(samples=samples, sample_rate=target_sr)
    finally:
        proc.terminate()
        proc.wait()
