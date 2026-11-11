"""
ad/train.py — Train the Deep AutoEncoder on ToyADMOS / DCASE2020 (ToyCar)

─────────────────────────────────────────────────────────────────────────────
HOW TO OBTAIN THE DATASET
─────────────────────────────────────────────────────────────────────────────
1. Download DCASE 2020 Task 2 "Development dataset":
       https://zenodo.org/record/3678171  (dev_data_ToyCar.zip, ~1.1 GB)
   OR use the MLCommons script:
       tiny/benchmark/training/anomaly_detection/download_dataset.sh

2. Extract to a folder, e.g.  ./data/DCASE2020/dev_data/ToyCar/
   Expected structure:
       ToyCar/
         train/
           normal_id_01_00000000.wav
           ...
         test/
           normal_id_01_00000000.wav
           anomaly_id_01_00000000.wav
           ...

Usage:
  python ad/train.py --data-dir ./data/DCASE2020/dev_data/ToyCar
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.io import wavfile as _wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ad.model import build_autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Feature extraction config (MLPerf Tiny spec) ────────────────────────────
SAMPLE_RATE = 16_000
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 128
N_FRAMES    = 5          # 5 consecutive mel frames → 640 dims
INPUT_DIM   = N_MELS * N_FRAMES   # 640


def _load_wav(wav_path: str):
    """Load WAV file → (samples, sample_rate), mono float32."""
    if HAS_SCIPY:
        sr, data = _wavfile.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.dtype == np.int16 or data.max() > 1.0:
            data = data / 32768.0
        return data, sr
    else:
        wf, sr = torchaudio.load(wav_path)
        return wf.mean(dim=0).numpy(), sr


def _mel_spectrogram(samples: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram → shape (time, n_mels)."""
    # Resample if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        samples = resample_poly(samples, SAMPLE_RATE // g, sr // g).astype(np.float32)

    # STFT → power spectrogram
    n = len(samples)
    frames = []
    for start in range(0, n - N_FFT + 1, HOP_LENGTH):
        frame = samples[start: start + N_FFT] * np.hanning(N_FFT)
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    power = np.array(frames, dtype=np.float32)  # (T, n_fft//2+1)

    # Mel filterbank (triangular, Hz scale)
    f_max  = SAMPLE_RATE / 2.0
    freqs  = np.linspace(0, f_max, N_FFT // 2 + 1)
    mel_lo = 2595 * np.log10(1 + 20 / 700)
    mel_hi = 2595 * np.log10(1 + f_max / 700)
    mels   = np.linspace(mel_lo, mel_hi, N_MELS + 2)
    hz     = 700 * (10 ** (mels / 2595) - 1)
    fb     = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        lo, cen, hi = hz[m-1], hz[m], hz[m+1]
        for k, f in enumerate(freqs):
            if lo <= f <= cen:
                fb[m-1, k] = (f - lo) / (cen - lo + 1e-8)
            elif cen < f <= hi:
                fb[m-1, k] = (hi - f) / (hi - cen + 1e-8)
    mel_spec = power @ fb.T          # (T, N_MELS)
    return np.log(mel_spec + 1e-8)


def _extract_features(wav_path: str) -> np.ndarray:
    """Extract overlapping 640-dim log-mel feature vectors from a WAV file."""
    samples, sr = _load_wav(wav_path)
    log_mel = _mel_spectrogram(samples, sr)  # (T, N_MELS)

    # Sliding window: collect 5-frame windows
    vectors = []
    for i in range(len(log_mel) - N_FRAMES + 1):
        vectors.append(log_mel[i: i + N_FRAMES].flatten())  # (640,)

    if len(vectors) == 0:
        vectors = [np.zeros(N_MELS * N_FRAMES, dtype=np.float32)]
    return np.stack(vectors, axis=0).astype(np.float32)


class ToyCarDataset(Dataset):
    """
    Dataset of 640-dim log-mel feature vectors from ToyCar WAV files.
    Training uses only 'normal' clips (unsupervised anomaly detection).
    """

    def __init__(self, data_dir: str, split: str = "train"):
        folder = Path(data_dir) / split
        if not folder.exists():
            raise FileNotFoundError(
                f"ToyCar {split} folder not found: {folder}\n"
                "Please download the DCASE2020 Task 2 dataset.")

        # For training: use only normal files
        if split == "train":
            wav_files = sorted(folder.glob("normal_*.wav"))
        else:
            wav_files = sorted(folder.glob("*.wav"))

        if len(wav_files) == 0:
            raise RuntimeError(f"No WAV files found in {folder}")

        print(f"Extracting features from {len(wav_files)} WAV files [{split}] ...")
        cache_path = str(folder) + f"_features_{split}.npy"

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            print(f"  Loaded from cache: {data.shape}")
        else:
            chunks = [_extract_features(str(f)) for f in wav_files]
            data   = np.concatenate(chunks, axis=0)
            np.save(cache_path, data)
            print(f"  Extracted + saved: {data.shape}")

        # Normalise (z-score over training set)
        self.mean_ = data.mean(axis=0)
        self.std_  = data.std(axis=0) + 1e-8
        data = (data - self.mean_) / self.std_

        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True,
                        help="Path to ToyCar dataset folder")
    parser.add_argument("--ckpt-dir",    default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--num-workers", type=int,   default=2)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device : {DEVICE}")

    train_ds = ToyCarDataset(args.data_dir, split="train")
    loader   = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers)

    model     = build_autoencoder(INPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_path = os.path.join(args.ckpt_dir, "ad_best.pth")

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss  = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n          += x.size(0)
        scheduler.step()
        avg = total_loss / n
        print(f"Epoch {epoch+1:4d}/{args.epochs}  MSE loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"model":     model.state_dict(),
                        "train_loss": best_loss,
                        "epoch":     epoch,
                        "mean":      train_ds.mean_.tolist(),
                        "std":       train_ds.std_.tolist()}, best_path)
            print(f"  -> saved best (loss={best_loss:.6f})")

    print(f"\nTraining done. Best MSE: {best_loss:.6f}")


if __name__ == "__main__":
    main()
