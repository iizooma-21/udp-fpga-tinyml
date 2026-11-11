"""
vww/download_vww.py — Download and prepare the VWW dataset

Strategy (tried in order):
  1. tensorflow-datasets (TFDS) — downloads ~1.5 GB of COCO-VWW images
     from Google Storage and converts to our folder layout.
  2. Direct COCO download fallback — downloads only the COCO 2014 validation
     images (~6 GB) and generates VWW splits from annotations.

Output layout:
  <output_dir>/
    train/person/   (≈81 K images)
    train/background/
    val/person/     (≈7 K images)
    val/background/

Usage:
  python vww/download_vww.py --output-dir ./data/vww_96
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

# ─── Approach 1: tensorflow-datasets ─────────────────────────────────────────

def _install_pkg(pkg):
    import subprocess
    print(f"Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _try_tfds(output_dir: str) -> bool:
    """Download VWW via tensorflow-datasets. Returns True if successful."""
    try:
        import tensorflow_datasets as tfds   # noqa: F401
        print("tensorflow-datasets found.")
    except ImportError:
        print("tensorflow-datasets not installed; trying COCO fallback.")
        return False

    import tensorflow_datasets as tfds
    from PIL import Image

    print("Downloading VWW via tensorflow-datasets (~1.5 GB) ...")
    try:
        for split_tf, split_dir in [("train", "train"), ("validation", "val")]:
            ds = tfds.load("visual_wake_words", split=split_tf,
                           as_supervised=True, shuffle_files=False)
            for label_name in ["person", "background"]:
                os.makedirs(os.path.join(output_dir, split_dir, label_name), exist_ok=True)

            print(f"  Converting {split_tf} split ...")
            for idx, (img_tensor, label_tensor) in enumerate(ds):
                img_np    = img_tensor.numpy()
                label_int = int(label_tensor.numpy())
                class_name = "person" if label_int == 1 else "background"
                save_path  = os.path.join(output_dir, split_dir, class_name,
                                          f"{idx:06d}.jpg")
                if not os.path.exists(save_path):
                    Image.fromarray(img_np).save(save_path, quality=95)
                if idx % 5000 == 0:
                    print(f"    {idx} images processed ...")
    except Exception as e:
        print(f"TFDS download failed: {e}")
        return False

    print(f"VWW dataset ready at: {output_dir}")
    return True


# ─── Approach 2: Direct COCO image download (annotation-guided) ──────────────

def _download_single(url: str, dest: str, session) -> bool:
    """Download one image; returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def _try_coco_fallback(output_dir: str) -> bool:
    """
    Download full VWW dataset from COCO CDN using parallel HTTP requests.
    Uses 16 worker threads for ~16x speedup over sequential downloads.
    Downloads ALL person/background images (no per-class cap).
    """
    import zipfile
    import urllib.request
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image

    try:
        import requests
        USE_REQUESTS = True
    except ImportError:
        USE_REQUESTS = False

    ANNOTATIONS_URL = (
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    )
    VAL_BASE   = "http://images.cocodataset.org/val2014/"
    TRAIN_BASE = "http://images.cocodataset.org/train2014/"
    N_WORKERS  = 16   # parallel download threads

    cache_dir = os.path.join(output_dir, "_cache")
    ann_zip   = os.path.join(cache_dir, "annotations_trainval2014.zip")
    ann_dir   = os.path.join(cache_dir, "annotations")

    os.makedirs(cache_dir, exist_ok=True)

    # Download annotations if missing
    if not os.path.exists(ann_zip) or os.path.getsize(ann_zip) < 50_000_000:
        print("Downloading COCO annotations (~240 MB) ...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, ann_zip,
            reporthook=lambda n, bs, t:
                print(f"  {n*bs/1e6:.0f}/{t/1e6:.0f} MB", end="\r") if t > 0 else None)
        print()

    if not os.path.exists(ann_dir):
        print("Extracting annotations ...")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(cache_dir)

    # Thread-local requests sessions (one per thread for connection reuse)
    _local = threading.local()

    def _get_session():
        if not hasattr(_local, "session"):
            _local.session = requests.Session() if USE_REQUESTS else None
        return _local.session

    def _dl_and_resize(args):
        """Download + resize one image. Returns (dst, ok)."""
        url, dst = args
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return dst, True
        ok = False
        session = _get_session()
        if session:
            try:
                r = session.get(url, timeout=20)
                if r.status_code == 200:
                    with open(dst, "wb") as f:
                        f.write(r.content)
                    ok = True
            except Exception:
                pass
        if not ok:
            try:
                urllib.request.urlretrieve(url, dst)
                ok = True
            except Exception:
                pass
        if ok:
            try:
                img = Image.open(dst).convert("RGB").resize((96, 96))
                img.save(dst, quality=92)
            except Exception:
                pass
        return dst, ok

    def _build_split(ann_json: str, base_url: str, split_name: str,
                     max_per_class: int = 0) -> bool:
        ann_path = os.path.join(ann_dir, ann_json)
        if not os.path.exists(ann_path):
            print(f"Annotation file not found: {ann_path}")
            return False

        print(f"Reading {ann_path} ...")
        with open(ann_path) as f:
            coco = json.load(f)

        img_meta  = {img["id"]: img for img in coco["images"]}
        person_ids: set = set()
        for ann in coco["annotations"]:
            img      = img_meta.get(ann["image_id"], {})
            img_area = img.get("width", 1) * img.get("height", 1)
            if ann.get("area", 0) / max(img_area, 1) >= 0.01:
                person_ids.add(ann["image_id"])

        all_ids        = list(img_meta.keys())
        background_ids = [i for i in all_ids if i not in person_ids]
        person_list    = list(person_ids)
        if max_per_class > 0:
            person_list    = person_list[:max_per_class]
            background_ids = random.sample(background_ids,
                                           min(max_per_class, len(background_ids)))

        for cls_name in ["person", "background"]:
            os.makedirs(os.path.join(output_dir, split_name, cls_name), exist_ok=True)

        # Build task list
        tasks = []
        for cls_name, id_list in [("person", person_list),
                                  ("background", background_ids)]:
            for img_id in id_list:
                meta  = img_meta[img_id]
                fname = meta["file_name"]
                url   = base_url + fname
                dst   = os.path.join(output_dir, split_name, cls_name,
                                     fname.split("_")[-1])
                tasks.append((url, dst))

        total, done, ok_count = len(tasks), 0, 0
        print(f"  {split_name}: {total} images to download with {N_WORKERS} threads …")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(_dl_and_resize, t) for t in tasks]
            for future in as_completed(futures):
                _, ok = future.result()
                done += 1
                ok_count += int(ok)
                if done % 500 == 0:
                    print(f"  {split_name}: {done}/{total} done "
                          f"({ok_count} ok)", end="\r", flush=True)
        print(f"  {split_name}: {total} images done ({ok_count} ok)        ")
        return True

    print("\nBuilding VWW val split from COCO val2014 (up to 2000/class) ...")
    _build_split("person_keypoints_val2014.json",   VAL_BASE,   "val",   1_000)
    print("\nBuilding VWW train split from COCO train2014 (FULL dataset) ...")
    _build_split("person_keypoints_train2014.json", TRAIN_BASE, "train", 0)  # 0 = no limit

    return True


# ─── Entry point ─────────────────────────────────────────────────────────────

def download_vww(output_dir: str):
    # Check if already downloaded (need at least 500 train images)
    train_person = Path(output_dir) / "train" / "person"
    if train_person.exists():
        n = len(list(train_person.glob("*.jpg")))
        if n > 50_000:   # full dataset has ~64k person images
            print(f"VWW full dataset already present ({n} train/person images).")
            return
        print(f"VWW only has {n} train/person images — downloading full dataset ...")

    os.makedirs(output_dir, exist_ok=True)
    if _try_tfds(output_dir):
        return
    if _try_coco_fallback(output_dir):
        return
    raise RuntimeError("All VWW download methods failed. "
                       "Please download manually: see vww/dataset.py for instructions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data/vww_96")
    args = parser.parse_args()
    download_vww(args.output_dir)


if __name__ == "__main__":
    main()
