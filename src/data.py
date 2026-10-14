"""LOL dataset discovery, image I/O and the train/validation split."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    name: str
    low: Path
    high: Path


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1] with shape (H, W, 3).

    Grayscale inputs are expanded to three channels and alpha is dropped.
    """
    # cv2.imread cannot open non-ASCII paths on Windows; decoding from bytes avoids that.
    buffer = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 with rounding and clipping."""
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write a float RGB image in [0, 1] to disk; the format follows the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise ValueError(f"Could not encode image as {path.suffix}: {path}")
    encoded.tofile(str(path))


def _image_files(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def list_pairs(root: str | Path, subset: str) -> list[ImagePair]:
    """List paired low/high images in ``root/subset/{low,high}``, matched by filename."""
    subset_dir = Path(root) / subset
    low_dir, high_dir = subset_dir / "low", subset_dir / "high"
    for directory in (low_dir, high_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Expected directory {directory}. Place the LOL dataset as "
                f"<data_root>/{subset}/low and <data_root>/{subset}/high."
            )

    low_files, high_files = _image_files(low_dir), _image_files(high_dir)
    unmatched = sorted(set(low_files) ^ set(high_files))
    if unmatched:
        raise ValueError(f"{len(unmatched)} unpaired files in {subset_dir}: {unmatched[:10]}")
    if not low_files:
        raise ValueError(f"No images found in {low_dir}")

    return [ImagePair(name, low_files[name], high_files[name]) for name in sorted(low_files)]


def load_pair(pair: ImagePair) -> tuple[np.ndarray, np.ndarray]:
    """Load a pair as float32 RGB arrays and check that their shapes agree."""
    low, high = read_image(pair.low), read_image(pair.high)
    if low.shape != high.shape:
        raise ValueError(f"Shape mismatch for {pair.name}: {low.shape} vs {high.shape}")
    return low, high


def make_val_names(names: list[str], val_size: int, seed: int) -> list[str]:
    """Pick a reproducible random subset of names for validation."""
    if not 0 < val_size < len(names):
        raise ValueError(f"val_size must be in (0, {len(names)}), got {val_size}")
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(names), val_size))


def split_train_val(
    pairs: list[ImagePair], split_file: str | Path, val_size: int, seed: int
) -> tuple[list[ImagePair], list[ImagePair]]:
    """Split pairs into train and validation sets.

    The validation names are written to ``split_file`` on first use and read back
    afterwards, so the split stays fixed even if the sampling code or seed changes.
    """
    split_file = Path(split_file)
    by_name = {p.name: p for p in pairs}

    if split_file.exists():
        val_names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        missing = [n for n in val_names if n not in by_name]
        if missing:
            raise ValueError(f"Names in {split_file} not found in dataset: {missing[:10]}")
    else:
        val_names = make_val_names(list(by_name), val_size, seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text("\n".join(val_names) + "\n")

    val_set = set(val_names)
    train = [p for p in pairs if p.name not in val_set]
    val = [p for p in pairs if p.name in val_set]
    return train, val
