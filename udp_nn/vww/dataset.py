"""
vww/dataset.py — Visual Wake Words (VWW) dataset loader

The VWW dataset is a subset of MS-COCO with binary person/no-person labels.
Images are resized to 96 × 96 RGB.

─────────────────────────────────────────────────────────────────────────────
HOW TO OBTAIN THE VWW DATASET
─────────────────────────────────────────────────────────────────────────────
Option A — Official MLPerf Tiny script (recommended):
  1. Clone the MLCommons tiny repo:
       git clone https://github.com/mlcommons/tiny
  2. Follow tiny/benchmark/training/visual_wake_words/README.md:
       cd tiny/benchmark/training/visual_wake_words
       bash download_and_prepare_dataset.sh
  This produces:
       vw_coco2014_96/
         train/  (person/ + background/)
         val/    (person/ + background/)

Option B — Use TensorFlow Datasets + convert (advanced):
       tfds build visual_wake_words
       python convert_tfds_to_folders.py   # custom script

Option C — Direct download (unofficial mirror):
  https://www.tensorflow.org/datasets/catalog/visual_wake_words
  Then convert to the folder layout expected below.

─────────────────────────────────────────────────────────────────────────────
Expected folder layout:
  <root>/
    train/
      person/       ← images containing a person
      background/   ← images without a person  (OR label 0)
    val/            (same structure)
─────────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES   = ["background", "person"]  # 0 = no-person, 1 = person
IMG_SIZE  = 96

# Normalisation matching the MLPerf Tiny reference (ImageNet stats)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class VWWDataset(Dataset):
    """
    VWW dataset from a folder layout:
        root/
          train/person/   root/train/background/
          val/person/     root/val/background/

    Args:
        root  : path to the dataset root (contains train/ and val/)
        split : "train" or "val"
        augment : apply random flips / colour jitter (train only)
    """

    def __init__(self, root: str, split: str = "train",
                 augment: Optional[bool] = None):
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"VWW dataset not found at '{split_dir}'.\n"
                "Please follow the instructions in vww/dataset.py to download.")

        _aug = augment if augment is not None else (split == "train")
        self.transform = _train_transform() if _aug else _eval_transform()

        self.samples: list = []
        for label_idx, label_name in enumerate(CLASSES):
            folder = split_dir / label_name
            if not folder.exists():
                # Also try numeric folder names as fallback
                for ext in ["jpg", "jpeg", "png", "JPEG"]:
                    pass
                raise FileNotFoundError(
                    f"Expected folder: {folder}\n"
                    "Make sure the dataset has 'person/' and 'background/' "
                    "subdirectories.")
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        print(f"VWW {split}: {len(self.samples):,} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
