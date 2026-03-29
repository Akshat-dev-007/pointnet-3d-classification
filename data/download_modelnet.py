"""Download and extract the ModelNet10 dataset."""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path


MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
DATA_DIR = Path(__file__).resolve().parent


def download_modelnet10(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract ModelNet10 dataset.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to the extracted ModelNet10 directory.
    """
    dataset_dir = data_dir / "ModelNet10"
    zip_path = data_dir / "ModelNet10.zip"

    if dataset_dir.exists():
        print(f"ModelNet10 already exists at {dataset_dir}")
        return dataset_dir

    # Download
    print(f"Downloading ModelNet10 from {MODELNET10_URL}...")
    print("This may take a few minutes (~500MB)...")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(MODELNET10_URL, str(zip_path), reporthook=_progress)
    print("\nDownload complete.")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(data_dir))
    print(f"Extracted to {dataset_dir}")

    # Clean up zip
    zip_path.unlink()
    print("Removed zip file.")

    return dataset_dir


def print_dataset_stats(dataset_dir: Path) -> None:
    """Print number of samples per class for train and test splits."""
    classes = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"\n{'Class':<15} {'Train':>6} {'Test':>6} {'Total':>6}")
    print("-" * 37)

    total_train, total_test = 0, 0
    for cls in classes:
        train_dir = dataset_dir / cls / "train"
        test_dir = dataset_dir / cls / "test"
        n_train = len(list(train_dir.glob("*.off"))) if train_dir.exists() else 0
        n_test = len(list(test_dir.glob("*.off"))) if test_dir.exists() else 0
        total_train += n_train
        total_test += n_test
        print(f"{cls:<15} {n_train:>6} {n_test:>6} {n_train + n_test:>6}")

    print("-" * 37)
    print(f"{'TOTAL':<15} {total_train:>6} {total_test:>6} {total_train + total_test:>6}")


if __name__ == "__main__":
    dataset_path = download_modelnet10()
    print_dataset_stats(dataset_path)
