"""LOL dataset discovery, image I/O and the train/validation split."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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
    try:
        return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())
    except ValueError as error:
        raise ValueError(f"Could not decode image: {path}") from error


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (PNG, JPEG, ...) to float32 RGB in [0, 1].

    Grayscale inputs are expanded to three channels and alpha is dropped.
    """
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image data")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 with rounding and clipping."""
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """Encode a float RGB image in [0, 1] as bytes in the format of ``extension``."""
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(extension, bgr)
    if not ok:
        raise ValueError(f"Could not encode image as {extension}")
    return encoded.tobytes()


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write a float RGB image in [0, 1] to disk; the format follows the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(image, path.suffix))


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


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an (H, W, 3) array in [0, 1] or uint8 to a float (3, H, W) tensor in [0, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.float().div(255.0) if tensor.dtype == torch.uint8 else tensor.float()


class LowLightCrops(Dataset):
    """Random crops of the low-light images only; Zero-DCE trains without references.

    Images are decoded once and kept in memory as uint8 (about 0.7 MB each for LOL).
    Randomness comes from torch's RNG, which the training loop seeds per epoch.
    """

    def __init__(self, pairs: list[ImagePair], crop_size: int, augment: bool = True) -> None:
        self.images = [to_uint8(read_image(p.low)) for p in pairs]
        self.crop_size = crop_size
        self.augment = augment
        for pair, image in zip(pairs, self.images, strict=True):
            if min(image.shape[:2]) < crop_size:
                raise ValueError(f"{pair.name} is smaller than crop size {crop_size}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index]
        h, w = image.shape[:2]
        top = int(torch.randint(0, h - self.crop_size + 1, (1,)))
        left = int(torch.randint(0, w - self.crop_size + 1, (1,)))
        crop = to_tensor(image[top : top + self.crop_size, left : left + self.crop_size])
        if self.augment:
            if torch.rand(1) < 0.5:
                crop = crop.flip(-1)
            if torch.rand(1) < 0.5:
                crop = crop.flip(-2)
        return crop
