from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from src.enhance import METHODS, enhance, get_enhancer, limit_size
from src.zerodce import DCENet, init_weights


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    model = DCENet()
    init_weights(model)
    weights = tmp_path / "zerodce.pth"
    torch.save(model.state_dict(), weights)

    tuned = tmp_path / "tuned_params.yaml"
    tuned.write_text(
        yaml.safe_dump({"gamma": {"gamma": 0.4}, "clahe": {"clip_limit": 2.0, "tile_size": 8}})
    )
    return {
        "paths": {"weights": str(weights), "tuned_params": str(tuned)},
        "zerodce": {"features": 32, "iterations": 8},
        "inference": {"max_side": 64},
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(37, 53, 3), (1, 1, 3), (17, 9, 3)])
def test_odd_sizes(config: dict[str, Any], method: str, shape: tuple[int, int, int]) -> None:
    image = np.random.default_rng(0).random(shape).astype(np.float32) * 0.2
    out = enhance(image, method, config)
    assert out.shape == shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("method", METHODS)
def test_grayscale_input(config: dict[str, Any], method: str) -> None:
    gray = (np.random.default_rng(1).random((20, 30)) * 50).astype(np.uint8)
    out = enhance(gray, method, config)
    assert out.shape == (20, 30, 3)


def test_rgba_and_single_channel_inputs(config: dict[str, Any]) -> None:
    rgba = np.random.default_rng(2).random((10, 12, 4)).astype(np.float32)
    single = np.random.default_rng(3).random((10, 12, 1)).astype(np.float32)
    assert enhance(rgba, "zerodce", config).shape == (10, 12, 3)
    assert enhance(single, "gamma", config).shape == (10, 12, 3)


def test_large_images_are_capped(config: dict[str, Any]) -> None:
    image = np.zeros((100, 300, 3), dtype=np.float32)
    out = enhance(image, "zerodce", config)
    assert max(out.shape[:2]) == 64
    assert out.shape[:2] == (21, 64)


def test_limit_size_leaves_small_images() -> None:
    image = np.zeros((10, 20, 3), dtype=np.float32)
    assert limit_size(image, 64) is image
    assert limit_size(image, None) is image


def test_unknown_method(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        get_enhancer("retinex", config)


def test_missing_weights(config: dict[str, Any]) -> None:
    config["paths"]["weights"] = "does/not/exist.pth"
    with pytest.raises(FileNotFoundError):
        get_enhancer("zerodce", config)
