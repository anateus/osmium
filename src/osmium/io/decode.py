import subprocess
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int


def probe_duration(path: str) -> float | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def decode(path: str, target_sr: int = 24000, progress_callback=None) -> AudioData:
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]

    if progress_callback is None:
        result = subprocess.run(cmd, capture_output=True, check=True)
        samples = np.frombuffer(result.stdout, dtype=np.float32)
        return AudioData(samples=samples, sample_rate=target_sr)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    chunks = []
    bytes_per_second = target_sr * 4
    try:
        while True:
            data = proc.stdout.read(bytes_per_second)
            if not data:
                break
            chunks.append(data)
            total_bytes = sum(len(c) for c in chunks)
            decoded_seconds = total_bytes / bytes_per_second
            progress_callback(decoded_seconds)
    finally:
        proc.wait()

    if not chunks:
        return AudioData(samples=np.array([], dtype=np.float32), sample_rate=target_sr)

    all_bytes = b"".join(chunks)
    samples = np.frombuffer(all_bytes, dtype=np.float32)
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
