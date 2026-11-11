"""
ad/evaluate.py — Evaluate the Deep AutoEncoder (anomaly detection)

Metrics:
  - Reconstruction MSE on normal vs anomaly test clips
  - ROC-AUC score (higher = better; MLPerf Tiny target >= 0.85)

Also reports:
  - INT4-quantized model AUC (weight-only fake-quant)

Usage:
  python ad/evaluate.py --ckpt checkpoints/ad_best.pth
                         --data-dir ./data/DCASE2020/dev_data/ToyCar

Note: The autoencoder uses dense (Linear) layers only.
      Systolic arrays give poor utilisation on batch-size-1 dense layers,
      so Timeloop analysis is not included in this script.
      See timeloop/problem_ad_*.yaml for the layer geometries.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.io import wavfile as _wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ad.model import build_autoencoder

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16_000
N_FFT      = 1024
HOP_LENGTH = 512
N_MELS     = 128
N_FRAMES   = 5
INPUT_DIM  = N_MELS * N_FRAMES   # 640

# ─── Feature extraction (same as train.py) ───────────────────────────────────

def _load_wav(wav_path: str):
    if HAS_SCIPY:
        sr, data = _wavfile.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / 32768.0
        return data, sr
    else:
        wf, sr = torchaudio.load(wav_path)
        return wf.mean(dim=0).numpy(), sr


def _mel_spectrogram(samples, sr):
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        samples = resample_poly(samples, SAMPLE_RATE // g, sr // g).astype(np.float32)
    frames = []
    for start in range(0, len(samples) - N_FFT + 1, HOP_LENGTH):
        frame = samples[start: start + N_FFT] * np.hanning(N_FFT)
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    power = np.array(frames, dtype=np.float32)
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
    return np.log(power @ fb.T + 1e-8)


def _extract_features(wav_path: str) -> np.ndarray:
    samples, sr = _load_wav(wav_path)
    log_mel = _mel_spectrogram(samples, sr)
    vectors = []
    for i in range(len(log_mel) - N_FRAMES + 1):
        vectors.append(log_mel[i: i + N_FRAMES].flatten())
    if not vectors:
        vectors = [np.zeros(N_MELS * N_FRAMES, dtype=np.float32)]
    return np.stack(vectors).astype(np.float32)


# ─── Fake-quant helper ────────────────────────────────────────────────────────

def fake_quant(x: torch.Tensor, bits: int = 4,
               per_channel: bool = False, ch_dim: int = 0) -> torch.Tensor:
    n = 2 ** (bits - 1)
    if per_channel:
        scale = x.abs().amax(
            dim=[i for i in range(x.dim()) if i != ch_dim], keepdim=True)
    else:
        scale = x.abs().max()
    scale = scale.clamp(1e-8) / n
    return (x / scale).round().clamp(-n, n - 1) * scale


# ─── Compute anomaly scores ───────────────────────────────────────────────────

@torch.no_grad()
def _score_model(model: nn.Module, x_tensor: torch.Tensor,
                 batch: int = 512) -> np.ndarray:
    """Return per-window MSE reconstruction error for all windows."""
    model.eval()
    scores = []
    for i in range(0, len(x_tensor), batch):
        chunk = x_tensor[i: i + batch].to(DEVICE)
        recon = model(chunk)
        mse   = ((chunk - recon) ** 2).mean(dim=-1).cpu().numpy()
        scores.append(mse)
    return np.concatenate(scores)


def clip_score(per_window_scores: np.ndarray) -> float:
    """Aggregate per-window scores to a single clip score (mean)."""
    return float(per_window_scores.mean())


# ─── INT4 weight-only model ───────────────────────────────────────────────────

class INT4WeightAutoEncoder(nn.Module):
    """AutoEncoder with per-channel INT4-quantized weights (Mode 4 analogue)."""

    def __init__(self, base: nn.Module, bits: int = 4):
        super().__init__()
        self.m    = deepcopy(base)
        self.bits = bits

    def _fq_w(self, w):
        return fake_quant(w, self.bits, per_channel=True, ch_dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.m.net:
            if isinstance(layer, nn.Linear):
                x = F.linear(x, self._fq_w(layer.weight), layer.bias)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=False)
        return x


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--data-dir",     required=True,
                        help="Path to ToyCar folder (contains test/)")
    parser.add_argument("--batch-size",   type=int, default=512)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model = build_autoencoder(INPUT_DIM).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Normalisation stats from training
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)

    # ── Load test clips ──────────────────────────────────────────────────────
    test_folder = Path(args.data_dir) / "test"
    wav_files   = sorted(test_folder.glob("*.wav"))
    if len(wav_files) == 0:
        raise RuntimeError(f"No WAV files found in {test_folder}")

    print(f"Found {len(wav_files)} test clips")

    labels: List[int] = []
    scores_fp32: List[float] = []

    for wav_path in wav_files:
        label = 1 if "anomaly" in wav_path.name else 0   # 1 = anomaly
        feats = _extract_features(str(wav_path))
        feats = (feats - mean) / std

        x = torch.tensor(feats, dtype=torch.float32)
        s = _score_model(model, x, args.batch_size)
        scores_fp32.append(clip_score(s))
        labels.append(label)

    labels      = np.array(labels)
    scores_fp32 = np.array(scores_fp32)

    # ── Compute AUC (FP32) ───────────────────────────────────────────────────
    auc_fp32 = roc_auc_score(labels, scores_fp32)

    # ── INT4 weight-only model ───────────────────────────────────────────────
    print("Computing INT4 weight-only AUC ...")
    int4_model  = INT4WeightAutoEncoder(model).to(DEVICE)
    scores_int4 = []
    for wav_path in wav_files:
        feats = _extract_features(str(wav_path))
        feats = (feats - mean) / std
        x = torch.tensor(feats, dtype=torch.float32)
        s = _score_model(int4_model, x, args.batch_size)
        scores_int4.append(clip_score(s))
    auc_int4 = roc_auc_score(labels, np.array(scores_int4))

    # ── Summary ──────────────────────────────────────────────────────────────
    n_normal  = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()
    target    = 0.85

    print()
    print("=" * 52)
    print("AD (AutoEncoder) AUC Summary")
    print("=" * 52)
    print(f"Test clips  : {len(labels):3d}  "
          f"(normal={n_normal}, anomaly={n_anomaly})")
    print("-" * 52)
    print(f"{'Mode':<28}  {'AUC':>8}  {'>=0.85':>7}")
    print("-" * 52)
    print(f"{'FP32':<28}  {auc_fp32:>7.4f}  "
          f"{'PASS' if auc_fp32 >= target else 'FAIL':>7}")
    print(f"{'INT4 weight-only':<28}  {auc_int4:>7.4f}  "
          f"{'PASS' if auc_int4 >= target else 'FAIL':>7}")
    print("=" * 52)

    out_path = os.path.join(os.path.dirname(args.ckpt), "ad_results.json")
    results  = {"FP32 AUC": auc_fp32, "INT4 weight-only AUC": auc_int4}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
