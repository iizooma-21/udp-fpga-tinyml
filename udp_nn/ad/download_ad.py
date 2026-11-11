"""
ad/download_ad.py — Download DCASE2020 Task 2 ToyCar dataset from Zenodo

Record : https://zenodo.org/record/3678171   (~1.1 GB zip)
Output : <output_dir>/dev_data/ToyCar/{train,test}/

Usage:
  python ad/download_ad.py --output-dir ./data/DCASE2020
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Zenodo record for DCASE2020 Task 2 development dataset (ToyCar)
# Multiple candidate URLs — tried in order until one succeeds
ZENODO_URLS = [
    # DCASE2020 Task2 dev data (ToyCar) — try direct Zenodo CDN first
    "https://zenodo.org/records/3678171/files/dev_data_ToyCar.zip",
    "https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip",
    "https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1",
    # Alternative: DCASE2020 Task2 from backup
    "https://zenodo.org/records/3960822/files/dev_data_ToyCar.zip",
    "https://zenodo.org/record/3960822/files/dev_data_ToyCar.zip",
]
FILENAME    = "dev_data_ToyCar.zip"
EXPECTED_KB = 500_000   # at least 500 MB

def _reporthook(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        print(f"  {pct:5.1f}%  {mb:.0f}/{tot:.0f} MB", end="\r", flush=True)


def download_ad(output_dir: str):
    toy_car_dir = Path(output_dir) / "dev_data" / "ToyCar"
    train_dir   = toy_car_dir / "train"
    test_dir    = toy_car_dir / "test"

    # Already downloaded?
    if train_dir.exists() and len(list(train_dir.glob("*.wav"))) > 100:
        print(f"DCASE ToyCar dataset already present at {toy_car_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, FILENAME)

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < EXPECTED_KB * 500:
        print(f"Downloading DCASE2020 Task 2 ToyCar (~1.1 GB) ...")
        downloaded = False
        for url in ZENODO_URLS:
            print(f"  Trying: {url}")
            try:
                # Use requests if available for better redirect handling
                try:
                    import requests
                    r = requests.get(url, stream=True, timeout=30,
                                    headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code == 200:
                        total = int(r.headers.get("content-length", 0))
                        done  = 0
                        with open(zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                                done += len(chunk)
                                if total > 0:
                                    print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB",
                                          end="\r", flush=True)
                        print()
                        downloaded = True
                        break
                except ImportError:
                    urllib.request.urlretrieve(url, zip_path, _reporthook)
                    print()
                    downloaded = True
                    break
            except Exception as e:
                print(f"    Failed: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                continue

        if not downloaded:
            print("\nAll download URLs failed.")
            print("Please manually download dev_data_ToyCar.zip from:")
            print("  https://zenodo.org/record/3678171")
            print(f"And place it at: {zip_path}")
            sys.exit(1)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    # Verify
    n_train = len(list(train_dir.glob("*.wav"))) if train_dir.exists() else 0
    n_test  = len(list(test_dir.glob("*.wav")))  if test_dir.exists() else 0
    print(f"Extracted: {n_train} train WAVs, {n_test} test WAVs")
    print(f"Dataset ready at: {toy_car_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data/DCASE2020")
    args = parser.parse_args()
    download_ad(args.output_dir)


if __name__ == "__main__":
    main()
