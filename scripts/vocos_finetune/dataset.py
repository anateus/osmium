import numpy as np
import soundfile as sf
import torch
import torchaudio.functional
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, filelist_path: str, num_samples: int = 24000, sample_rate: int = 24000, train: bool = True):
        with open(filelist_path) as f:
            self.filelist = [line.strip() for line in f if line.strip()]
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = torch.from_numpy(data.T)
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)

        gain_db = np.random.uniform(-6, -1) if self.train else -3.0
        peak = y.abs().max()
        if peak > 0:
            y = y / peak
            y = y * (10 ** (gain_db / 20))

        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            repeats = 1 + pad_length // y.size(-1)
            y = y.repeat(1, repeats + 1)[:, : self.num_samples]
        elif self.train:
            start = np.random.randint(0, y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]

        return y[0]
