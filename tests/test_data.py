from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import list_pairs, load_pair, read_image, split_train_val, write_image


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


@pytest.fixture
def fake_lol(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for i in range(10):
        for kind in ("low", "high"):
            _write_png(
                tmp_path / "train" / kind / f"{i}.png",
                rng.integers(0, 256, (8, 12, 3), dtype=np.uint8),
            )
    return tmp_path


def test_list_pairs_matches_by_name(fake_lol: Path) -> None:
    pairs = list_pairs(fake_lol, "train")
    assert len(pairs) == 10
    assert all(p.low.name == p.high.name == p.name for p in pairs)


def test_list_pairs_rejects_unpaired_files(fake_lol: Path) -> None:
    (fake_lol / "train" / "high" / "3.png").unlink()
    with pytest.raises(ValueError, match="unpaired"):
        list_pairs(fake_lol, "train")


def test_list_pairs_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_pairs(tmp_path, "eval15")


def test_read_image_is_rgb_float(tmp_path: Path) -> None:
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    _write_png(tmp_path / "red.png", bgr)
    image = read_image(tmp_path / "red.png")
    assert image.dtype == np.float32 and image.shape == (4, 5, 3)
    assert np.allclose(image[..., 0], 1.0) and np.allclose(image[..., 1:], 0.0)


def test_read_image_expands_grayscale(tmp_path: Path) -> None:
    _write_png(tmp_path / "gray.png", np.full((6, 7), 128, dtype=np.uint8))
    assert read_image(tmp_path / "gray.png").shape == (6, 7, 3)


def test_write_read_roundtrip(tmp_path: Path) -> None:
    image = np.random.default_rng(1).random((5, 9, 3)).astype(np.float32)
    write_image(tmp_path / "out" / "img.png", image)
    assert np.abs(read_image(tmp_path / "out" / "img.png") - image).max() <= 0.5 / 255 + 1e-6


def test_load_pair_rejects_shape_mismatch(fake_lol: Path) -> None:
    _write_png(fake_lol / "train" / "high" / "0.png", np.zeros((3, 3, 3), dtype=np.uint8))
    pair = list_pairs(fake_lol, "train")[0]
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_pair(pair)


def test_split_is_disjoint_and_persisted(fake_lol: Path, tmp_path: Path) -> None:
    pairs = list_pairs(fake_lol, "train")
    split_file = tmp_path / "splits" / "val.txt"
    train, val = split_train_val(pairs, split_file, val_size=3, seed=7)
    assert len(train) == 7 and len(val) == 3
    assert not {p.name for p in train} & {p.name for p in val}
    assert split_file.exists()

    # A different seed must not change an existing split: the file is the source of truth.
    _, val_again = split_train_val(pairs, split_file, val_size=3, seed=999)
    assert [p.name for p in val_again] == [p.name for p in val]


def test_split_is_reproducible_for_same_seed(fake_lol: Path, tmp_path: Path) -> None:
    pairs = list_pairs(fake_lol, "train")
    _, a = split_train_val(pairs, tmp_path / "a.txt", val_size=4, seed=3)
    _, b = split_train_val(pairs, tmp_path / "b.txt", val_size=4, seed=3)
    assert [p.name for p in a] == [p.name for p in b]
