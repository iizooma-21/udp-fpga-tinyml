"""
ad/create_synthetic_dataset.py — Create a synthetic ToyCar-like dataset

If the real DCASE2020 dataset cannot be downloaded (Zenodo requires
terms acceptance), this script generates synthetic WAV files with:
  - Normal: stationary sine wave + low-amplitude noise
  - Anomaly: same but with pulsed distortion injected

The model trained on this will validate the pipeline correctness.
For the real benchmark result, download the DCASE dataset manually from:
  https://zenodo.org/record/3678171

Usage:
  python ad/create_synthetic_dataset.py --output-dir ./data/DCASE2020/dev_data/ToyCar
"""

import argparse
import os
import struct
import wave
import numpy as np

SR    = 16_000
SECS  = 10
N     = SR * SECS   # 160,000 samples per clip


def _write_wav(path: str, data: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.clip(data, -1.0, 1.0)
    pcm  = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _normal_audio(seed: int) -> np.ndarray:
    """Stationary machine noise: harmonic + Gaussian noise."""
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, SECS, N)
    freq = 120.0 + rng.uniform(-5, 5)   # machine fundamental freq ~120 Hz
    sig  = (0.5 * np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.15 * np.sin(2 * np.pi * 3 * freq * t))
    sig += rng.normal(0, 0.02, N)
    return sig / np.abs(sig).max()


def _anomaly_audio(seed: int) -> np.ndarray:
    """Normal audio + pulsed distortion (impact/grinding noise)."""
    sig = _normal_audio(seed)
    rng = np.random.default_rng(seed + 10_000)
    # Random impact spikes
    for _ in range(rng.integers(3, 8)):
        pos = rng.integers(SR, N - SR)
        dur = rng.integers(SR // 20, SR // 5)
        decay = np.exp(-np.linspace(0, 10, dur))
        spike = rng.normal(0, 0.8, dur) * decay
        sig[pos:pos + dur] += spike
    return np.clip(sig / np.abs(sig).max(), -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data/DCASE2020/dev_data/ToyCar")
    parser.add_argument("--n-train",    type=int, default=200,
                        help="Normal train clips (real dataset has ~1000)")
    parser.add_argument("--n-test-normal",  type=int, default=100)
    parser.add_argument("--n-test-anomaly", type=int, default=100)
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    test_dir  = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print(f"Creating synthetic ToyCar dataset at: {args.output_dir}")
    print("  (Use the real DCASE2020 dataset for publication-quality results)")

    # Training: normal only
    for i in range(args.n_train):
        path = os.path.join(train_dir, f"normal_id_01_{i:08d}.wav")
        if not os.path.exists(path):
            _write_wav(path, _normal_audio(seed=i))
        if (i + 1) % 50 == 0:
            print(f"  train: {i+1}/{args.n_train}")

    # Test: normal
    for i in range(args.n_test_normal):
        path = os.path.join(test_dir, f"normal_id_01_{i:08d}.wav")
        if not os.path.exists(path):
            _write_wav(path, _normal_audio(seed=10_000 + i))

    # Test: anomaly
    for i in range(args.n_test_anomaly):
        path = os.path.join(test_dir, f"anomaly_id_01_{i:08d}.wav")
        if not os.path.exists(path):
            _write_wav(path, _anomaly_audio(seed=20_000 + i))

    n_tr = len([f for f in os.listdir(train_dir) if f.endswith(".wav")])
    n_te = len([f for f in os.listdir(test_dir)  if f.endswith(".wav")])
    print(f"\nDone: {n_tr} train clips, {n_te} test clips")
    print(f"Dataset: {args.output_dir}")


if __name__ == "__main__":
    main()
