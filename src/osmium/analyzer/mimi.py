import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path

MIMI_CHUNK_SIZE = 1920
MIMI_FRAME_RATE = 12.5
MIMI_SAMPLE_RATE = 24000
HF_REPO = "kyutai/moshiko-mlx-q4"
HF_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"


@dataclass
class MimiCodes:
    codes: np.ndarray
    frame_rate: float
    sample_rate: int
    duration: float


def get_model_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, HF_FILENAME)


def encode(samples: np.ndarray, sample_rate: int = 24000, model_path: str | None = None) -> MimiCodes:
    import rustymimi

    if model_path is None:
        model_path = get_model_path()

    tokenizer = rustymimi.StreamTokenizer(model_path)
    pcm = samples.astype(np.float32)

    all_codes = []
    for i in range(0, len(pcm), MIMI_CHUNK_SIZE):
        chunk = pcm[i:i + MIMI_CHUNK_SIZE]
        if len(chunk) < MIMI_CHUNK_SIZE:
            chunk = np.pad(chunk, (0, MIMI_CHUNK_SIZE - len(chunk)))
        tokenizer.encode(chunk)
        for _ in range(500):
            codes = tokenizer.get_encoded()
            if codes is not None:
                all_codes.append(np.array(codes).flatten()[:8])
                break
            import time
            time.sleep(0.001)

    codes_arr = np.array(all_codes)
    duration = len(samples) / sample_rate
    return MimiCodes(
        codes=codes_arr,
        frame_rate=MIMI_FRAME_RATE,
        sample_rate=sample_rate,
        duration=duration,
    )
