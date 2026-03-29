import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def tmp_audio_files(tmp_path):
    paths = []
    for i in range(5):
        samples = np.random.randn(48000).astype(np.float32) * 0.5
        path = tmp_path / f"test_{i}.wav"
        sf.write(str(path), samples, 24000)
        paths.append(path)

    filelist = tmp_path / "filelist.txt"
    filelist.write_text("\n".join(str(p) for p in paths) + "\n")
    return filelist


def test_dataset_returns_correct_shape(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset
    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=True)
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (24000,)
    assert sample.dtype == torch.float32


def test_dataset_length(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset
    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=True)
    assert len(ds) == 5


def test_dataset_gain_normalization(tmp_audio_files):
    from scripts.vocos_finetune.dataset import AudioDataset
    ds = AudioDataset(filelist_path=str(tmp_audio_files), num_samples=24000, sample_rate=24000, train=False)
    sample = ds[0]
    assert sample.abs().max() > 0
    assert sample.abs().max() <= 1.5


def test_dataset_pads_short_audio(tmp_path):
    from scripts.vocos_finetune.dataset import AudioDataset
    samples = np.random.randn(6000).astype(np.float32) * 0.5
    path = tmp_path / "short.wav"
    sf.write(str(path), samples, 24000)
    filelist = tmp_path / "filelist.txt"
    filelist.write_text(str(path) + "\n")
    ds = AudioDataset(filelist_path=str(filelist), num_samples=24000, sample_rate=24000, train=False)
    sample = ds[0]
    assert sample.shape == (24000,)
