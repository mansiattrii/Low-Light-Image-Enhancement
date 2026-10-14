"""Classical low-light baselines and their hyperparameter search on the validation set.

Usage:
    python -m src.classical --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.config import load_config
from src.data import list_pairs, load_pair, split_train_val, to_uint8
from src.metrics import psnr_ssim

Enhancer = Callable[[np.ndarray], np.ndarray]


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply ``out = in ** gamma`` per channel; gamma < 1 brightens."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return np.power(np.clip(image, 0.0, 1.0), gamma).astype(np.float32)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Equalize the histogram of the luma channel (YCrCb) and leave chroma unchanged.

    Equalizing R, G and B independently would change their ratios and shift hues.
    """
    ycrcb = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0


def clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space with a ``tile_size x tile_size`` grid."""
    lab = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def build_classical_methods(params: dict[str, Any]) -> dict[str, Enhancer]:
    """Create the classical enhancers from tuned parameters (see ``tuned_params.yaml``)."""
    gamma = params["gamma"]["gamma"]
    clip, tile = params["clahe"]["clip_limit"], params["clahe"]["tile_size"]
    return {
        "gamma": lambda x: gamma_correction(x, gamma),
        "hist_eq": histogram_equalization,
        "clahe": lambda x: clahe(x, clip, tile),
    }


def load_tuned_params(path: str | Path) -> dict[str, Any]:
    """Load tuned classical parameters written by ``python -m src.classical``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.classical` first.")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _score(enhancer: Enhancer, pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    scores = np.array([psnr_ssim(enhancer(low), high) for low, high in pairs])
    return float(scores[:, 0].mean()), float(scores[:, 1].mean())


def tune(config: dict[str, Any]) -> dict[str, Any]:
    """Grid-search gamma and CLAHE parameters on the validation split by mean PSNR.

    Writes the chosen parameters and the full grid of validation scores to disk.
    """
    paths, grid = config["paths"], config["classical"]
    pairs = list_pairs(paths["data_root"], paths["train_subset"])
    _, val = split_train_val(
        pairs, paths["val_split_file"], config["data"]["val_size"], config["seed"]
    )
    val_images = [load_pair(p) for p in tqdm(val, desc="Loading validation")]

    candidates: list[tuple[str, dict[str, Any], Enhancer]] = [
        ("gamma", {"gamma": g}, lambda x, g=g: gamma_correction(x, g)) for g in grid["gamma_grid"]
    ] + [
        ("clahe", {"clip_limit": c, "tile_size": t}, lambda x, c=c, t=t: clahe(x, c, t))
        for c, t in itertools.product(grid["clahe_clip_grid"], grid["clahe_tile_grid"])
    ]

    rows = []
    for method, params, enhancer in tqdm(candidates, desc="Grid search"):
        psnr, ssim = _score(enhancer, val_images)
        rows.append((method, params, psnr, ssim))

    tuned: dict[str, Any] = {
        "selection": {
            "split": f"validation ({len(val)} pairs from {paths['train_subset']})",
            "criterion": "mean PSNR",
        }
    }
    for method in ("gamma", "clahe"):
        _, params, psnr, ssim = max((r for r in rows if r[0] == method), key=lambda r: r[2])
        tuned[method] = {**params, "val_psnr": round(psnr, 4), "val_ssim": round(ssim, 4)}

    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["tuned_params"], "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False)
    with open(results_dir / "tuning_classical_val.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "params", "psnr", "ssim"])
        for method, params, psnr, ssim in rows:
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            writer.writerow([method, label, f"{psnr:.4f}", f"{ssim:.4f}"])
    return tuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune classical baselines on the validation set.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    tuned = tune(load_config(args.config))
    print(yaml.safe_dump(tuned, sort_keys=False))


if __name__ == "__main__":
    main()
