import os
import subprocess
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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


def _decode_range(path: str, start: float, duration: float, target_sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", f"{start:.3f}",
        "-i", path,
        "-t", f"{duration:.3f}",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def decode(path: str, target_sr: int = 24000, progress_callback=None) -> AudioData:
    total_duration = probe_duration(path)

    if total_duration and total_duration > 120:
        return _decode_parallel(path, total_duration, target_sr, progress_callback)

    return _decode_single(path, target_sr, progress_callback)


def _decode_single(path: str, target_sr: int, progress_callback=None) -> AudioData:
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
            progress_callback(total_bytes / bytes_per_second)
    finally:
        proc.wait()

    if not chunks:
        return AudioData(samples=np.array([], dtype=np.float32), sample_rate=target_sr)

    samples = np.frombuffer(b"".join(chunks), dtype=np.float32)
    return AudioData(samples=samples, sample_rate=target_sr)


def _decode_parallel(
    path: str,
    total_duration: float,
    target_sr: int,
    progress_callback=None,
) -> AudioData:
    n_workers = min(os.cpu_count() or 4, 8)
    overlap = 0.05
    chunk_dur = total_duration / n_workers

    ranges = []
    for i in range(n_workers):
        start = max(0, i * chunk_dur - overlap)
        end = min(total_duration, (i + 1) * chunk_dur + overlap)
        ranges.append((start, end - start))

    results = [None] * n_workers
    completed = [0]

    def _decode_one(idx_start_dur):
        idx, start, dur = idx_start_dur
        samples = _decode_range(path, start, dur, target_sr)
        results[idx] = samples
        completed[0] += 1
        if progress_callback:
            progress_callback(completed[0] * chunk_dur)
        return idx

    args = [(i, s, d) for i, (s, d) in enumerate(ranges)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(_decode_one, args))

    overlap_samples = int(overlap * target_sr)
    parts = []
    for i, chunk in enumerate(results):
        if i == 0:
            trim_end = overlap_samples if i < n_workers - 1 else 0
            parts.append(chunk[:len(chunk) - trim_end] if trim_end > 0 else chunk)
        elif i == n_workers - 1:
            trim_start = overlap_samples
            if trim_start > 0 and trim_start < len(chunk):
                prev = parts[-1]
                xf = min(overlap_samples, len(prev), len(chunk))
                if xf > 1:
                    fade = np.linspace(0, 1, xf, dtype=np.float32)
                    blended = prev[-xf:] * (1 - fade) + chunk[:xf] * fade
                    parts[-1] = prev[:-xf]
                    parts.append(blended)
                parts.append(chunk[xf:])
            else:
                parts.append(chunk)
        else:
            trim_start = overlap_samples
            trim_end = overlap_samples
            if trim_start > 0 and trim_start < len(chunk):
                prev = parts[-1]
                xf = min(overlap_samples, len(prev), len(chunk))
                if xf > 1:
                    fade = np.linspace(0, 1, xf, dtype=np.float32)
                    blended = prev[-xf:] * (1 - fade) + chunk[:xf] * fade
                    parts[-1] = prev[:-xf]
                    parts.append(blended)
                end = len(chunk) - trim_end if trim_end > 0 else len(chunk)
                parts.append(chunk[xf:end])
            else:
                end = len(chunk) - trim_end if trim_end > 0 else len(chunk)
                parts.append(chunk[:end])

    samples = np.concatenate(parts)
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
