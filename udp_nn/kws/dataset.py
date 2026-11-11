"""
kws/dataset.py — Google Speech Commands dataset for MLPerf Tiny KWS

Classes (12): 10 target keywords + silence + unknown
MFCC features: 49 frames × 10 coefficients → shape (1, 49, 10)

Requires: torchaudio (for download), scipy (for wav loading)
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# MLPerf Tiny KWS target keywords
KEYWORDS = ["yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go"]
SILENCE  = "silence"
UNKNOWN  = "unknown"
CLASSES  = KEYWORDS + [SILENCE, UNKNOWN]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Audio / MFCC config matching MLPerf Tiny reference
SAMPLE_RATE  = 16_000
CLIP_LEN     = 16_000        # 1 second
N_MFCC       = 10
N_FFT        = 512
HOP_LENGTH   = 320           # 20 ms hop
WIN_LENGTH   = 480           # 30 ms window  (must be <= n_fft=512)
N_MELS       = 40


def _make_mfcc_transform():
    return T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft":      N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "n_mels":     N_MELS,
            "f_min":      20.0,
            "f_max":      4000.0,
        },
    )


def _pad_or_trim(wav: torch.Tensor) -> torch.Tensor:
    """Pad or trim waveform to exactly CLIP_LEN samples."""
    if wav.shape[-1] < CLIP_LEN:
        wav = torch.nn.functional.pad(wav, (0, CLIP_LEN - wav.shape[-1]))
    else:
        wav = wav[..., :CLIP_LEN]
    return wav


def _load_wav_scipy(path: str) -> torch.Tensor:
    """Load a WAV file using scipy (no torchcodec required)."""
    sr, data = wavfile.read(path)
    # Normalise to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    # Handle stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data).unsqueeze(0)  # (1, samples)
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    return wav


def _find_data_root(root: str) -> Path:
    """Find the speech_commands_v0.02 folder under root."""
    candidates = [
        Path(root) / "SpeechCommands" / "speech_commands_v0.02",
        Path(root) / "speech_commands_v0.02",
        Path(root),
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"Could not find Speech Commands data under {root}. "
        "Please ensure torchaudio downloaded the dataset first."
    )


def _get_split_files(data_root: Path, split: str) -> List[Path]:
    """Return list of WAV file paths for the requested split."""
    val_file  = data_root / "validation_list.txt"
    test_file = data_root / "testing_list.txt"

    val_set  = set()
    test_set = set()
    if val_file.exists():
        val_set  = {l.strip() for l in val_file.read_text().splitlines() if l.strip()}
    if test_file.exists():
        test_set = {l.strip() for l in test_file.read_text().splitlines() if l.strip()}

    all_wavs = list(data_root.rglob("*.wav"))
    result   = []
    for p in all_wavs:
        rel = str(p.relative_to(data_root)).replace("\\", "/")
        if split == "validation":
            if rel in val_set:
                result.append(p)
        elif split == "testing":
            if rel in test_set:
                result.append(p)
        else:  # training
            if rel not in val_set and rel not in test_set:
                result.append(p)
    return result


class SpeechCommandsDataset(Dataset):
    """
    Google Speech Commands v2 dataset filtered to 12 MLPerf Tiny classes.

    Args:
        root     : parent directory; dataset downloaded into root/SpeechCommands/
        split    : "training" | "validation" | "testing"
        download : if True, downloads the dataset automatically
    """

    # ------------------------------------------------------------------
    # Class-level helpers (called before __init__ scans files)
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_downloaded(root: str, download: bool) -> Path:
        """Return path to speech_commands_v0.02, downloading if needed."""
        candidates = [
            Path(root) / "SpeechCommands" / "speech_commands_v0.02",
            Path(root) / "speech_commands_v0.02",
        ]
        for c in candidates:
            # Use validation_list.txt as a lightweight existence check
            # (avoids iterdir() on 100k-file directories which is slow on Windows)
            if (c / "validation_list.txt").exists():
                return c
        # Not present — fall back to torchaudio download (slow but correct)
        if download:
            print("[dataset] Dataset not found locally — downloading via torchaudio …")
            torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)
            for c in candidates:
                if c.is_dir() and any(c.iterdir()):
                    return c
        raise FileNotFoundError(
            f"Speech Commands data not found under {root}. "
            "Re-run with download=True."
        )

    @staticmethod
    def _read_split_lists(data_root: Path):
        """Return (val_set, test_set) as sets of 'keyword/filename.wav' strings."""
        def _load(fname):
            p = data_root / fname
            if not p.exists():
                return set()
            return {l.strip().replace("\\", "/")
                    for l in p.read_text().splitlines() if l.strip()}
        return _load("validation_list.txt"), _load("testing_list.txt")

    @staticmethod
    def _load_or_build_cache(data_root: Path, cache_file: Path) -> dict:
        """
        Return dict {"training": [(path, label_idx), ...], "validation": ..., "testing": ...}.
        Scans the filesystem only on the first call; subsequent calls load from cache JSON.
        """
        if cache_file.exists():
            print("[dataset] Loading file list from cache …")
            data = json.loads(cache_file.read_text())
            # Convert lists back to list-of-tuples
            return {k: [tuple(v) for v in lst] for k, lst in data.items()}

        print("[dataset] Building file list cache (one-time scan, may take a minute) …")
        val_set, test_set = SpeechCommandsDataset._read_split_lists(data_root)

        splits: dict = {"training": [], "validation": [], "testing": []}
        n = 0
        for kw_dir in sorted(os.scandir(str(data_root)), key=lambda e: e.name):
            if not kw_dir.is_dir() or kw_dir.name.startswith("_"):
                continue
            label_idx = CLASS_TO_IDX.get(kw_dir.name, CLASS_TO_IDX[UNKNOWN])
            for wav_entry in os.scandir(kw_dir.path):
                if not wav_entry.name.endswith(".wav"):
                    continue
                rel = f"{kw_dir.name}/{wav_entry.name}"
                if rel in val_set:
                    splits["validation"].append((wav_entry.path, label_idx))
                elif rel in test_set:
                    splits["testing"].append((wav_entry.path, label_idx))
                else:
                    splits["training"].append((wav_entry.path, label_idx))
                n += 1
                if n % 10000 == 0:
                    print(f"[dataset]   … {n} files scanned", flush=True)

        print(f"[dataset] Scan complete: {n} total files. Saving cache …")
        cache_file.write_text(json.dumps(splits))
        print("[dataset] Cache saved.")
        return splits

    def __init__(self, root: str, split: str = "training",
                 download: bool = True):
        self.split     = split
        self.transform = _make_mfcc_transform()
        self._samples: List[Tuple[str, int]] = []

        # ------------------------------------------------------------------
        # 1. Download if needed (only triggers network if not yet present)
        # ------------------------------------------------------------------
        data_root = self._ensure_downloaded(root, download)
        print(f"[dataset] data root: {data_root}")

        # ------------------------------------------------------------------
        # 2. Load (or build) the split→samples cache (JSON, built once)
        # ------------------------------------------------------------------
        cache_file = data_root / "_cursor_cache.json"
        all_splits = self._load_or_build_cache(data_root, cache_file)
        for path, label_idx in all_splits[split]:
            self._samples.append((path, label_idx))
        print(f"[dataset] {len(self._samples)} samples for split='{split}'")

        # ------------------------------------------------------------------
        # 4. Silence samples (~10 % of dataset)
        # ------------------------------------------------------------------
        if split == "training":
            self._n_silence = len(self._samples) // 10
        else:
            self._n_silence = max(100, len(self._samples) // 10)

        self._silence_label = CLASS_TO_IDX[SILENCE]

    def __len__(self):
        return len(self._samples) + self._n_silence

    def _get_silence(self) -> torch.Tensor:
        """Return a near-zero waveform representing silence."""
        return torch.randn(1, CLIP_LEN) * 1e-4

    def __getitem__(self, idx: int):
        if idx >= len(self._samples):
            # Silence sample
            wav   = self._get_silence()
            label = self._silence_label
        else:
            path, label = self._samples[idx]
            wav = _load_wav_scipy(path)
            wav = _pad_or_trim(wav)

        mfcc = self.transform(wav)         # (1, n_mfcc, time)
        # Transpose to (1, time, n_mfcc) → (B, 1, 49, 10)
        mfcc = mfcc.permute(0, 2, 1)
        # Normalise per sample
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return mfcc, label
