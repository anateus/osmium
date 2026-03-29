import argparse
import hashlib
import random
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/60/train-clean-100.tar.gz"
MD5 = "2c05cecece06364326d57678c8791e82"
ARCHIVE_NAME = "train-clean-100.tar.gz"
EXTRACT_SUBDIR = "LibriTTS/train-clean-100"
VAL_SIZE = 200
SEED = 42


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dest: Path) -> None:
    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = count * block_size
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        print(f"\r  {pct}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("training/data"))
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    archive_path = data_dir / ARCHIVE_NAME
    extract_dir = data_dir / EXTRACT_SUBDIR
    filelists_dir = data_dir / "filelists"
    train_txt = filelists_dir / "train.txt"
    val_txt = filelists_dir / "val.txt"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        download_with_progress(URL, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    print("Verifying MD5 checksum...")
    actual = md5_file(archive_path)
    if actual != MD5:
        raise ValueError(f"MD5 mismatch: expected {MD5}, got {actual}")
    print("  Checksum OK")

    if not extract_dir.exists():
        print(f"Extracting to {data_dir / 'LibriTTS'} ...")
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(path=data_dir, filter="data")
        print("  Extraction complete")
    else:
        print(f"Already extracted: {extract_dir}")

    if train_txt.exists() and val_txt.exists():
        print("Filelists already exist, skipping generation")
        return

    print("Scanning for .wav files...")
    wav_files = sorted(extract_dir.rglob("*.wav"))
    print(f"  Found {len(wav_files)} utterances")

    rng = random.Random(SEED)
    indices = list(range(len(wav_files)))
    rng.shuffle(indices)
    val_indices = set(indices[:VAL_SIZE])

    train_paths = [wav_files[i] for i in range(len(wav_files)) if i not in val_indices]
    val_paths = [wav_files[i] for i in val_indices]

    filelists_dir.mkdir(parents=True, exist_ok=True)
    train_txt.write_text("\n".join(str(p) for p in train_paths) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_paths) + "\n")

    print(f"  train.txt: {len(train_paths)} utterances")
    print(f"  val.txt:   {len(val_paths)} utterances")
    print("Done.")


if __name__ == "__main__":
    main()
