"""Single inference entry point shared by the CLI, evaluation, the API and the demo.

Usage:
    python -m src.enhance --input dark.jpg --output out.png --method zerodce
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.classical import Enhancer, build_classical_methods, load_tuned_params
from src.config import load_config
from src.data import read_image, write_image
from src.zerodce import DCENet, enhance_array, load_model

METHODS = ("gamma", "hist_eq", "clahe", "zerodce")


@lru_cache(maxsize=4)
def _cached_model(weights: str, features: int, iterations: int) -> DCENet:
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"Zero-DCE weights not found: {path}")
    return load_model(path, "cpu", features, iterations)


def get_enhancer(method: str, config: dict[str, Any], weights: str | None = None) -> Enhancer:
    """Return a function mapping a float RGB image in [0, 1] to its enhanced version.

    Zero-DCE weights are loaded once per path and reused across calls.
    """
    if method == "zerodce":
        zcfg = config["zerodce"]
        model = _cached_model(
            weights or config["paths"]["weights"], zcfg["features"], zcfg["iterations"]
        )
        return lambda image: enhance_array(model, image)
    if method in METHODS:
        return build_classical_methods(load_tuned_params(config["paths"]["tuned_params"]))[method]
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def limit_size(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longer side is at most ``max_side``, keeping the aspect ratio."""
    if not max_side or max(image.shape[:2]) <= max_side:
        return image
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def enhance(image: np.ndarray, method: str, config: dict[str, Any]) -> np.ndarray:
    """Enhance an RGB or grayscale image (float in [0, 1] or uint8); returns float RGB in [0, 1]."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[..., :3]
    image = limit_size(
        np.ascontiguousarray(image, dtype=np.float32), config["inference"]["max_side"]
    )
    with torch.inference_mode():
        return np.clip(get_enhancer(method, config)(image), 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance a low-light image.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path (e.g. out.png)")
    parser.add_argument("--method", choices=METHODS, default="zerodce")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    result = enhance(read_image(args.input), args.method, config)
    write_image(args.output, result)
    print(f"Saved {args.output} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
