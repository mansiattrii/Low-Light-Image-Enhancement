"""Evaluate enhancement methods on a LOL split and write the results table.

Usage:
    python -m src.evaluate --config config.yaml              # eval15 (final results)
    python -m src.evaluate --config config.yaml --split val  # validation split
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import skimage
from tqdm import tqdm

from src.classical import Enhancer, build_classical_methods, load_tuned_params
from src.config import load_config
from src.data import ImagePair, list_pairs, load_pair, split_train_val
from src.metrics import psnr_ssim


@dataclass
class Method:
    name: str
    enhance: Enhancer
    settings: str
    learned_params: int


@dataclass
class MethodResult:
    method: Method
    psnr: np.ndarray
    ssim: np.ndarray
    cpu_ms: np.ndarray


def build_methods(config: dict[str, Any]) -> list[Method]:
    """Assemble every method to evaluate, including the unprocessed input as a reference."""
    tuned = load_tuned_params(config["paths"]["tuned_params"])
    classical = build_classical_methods(tuned)
    clahe_params = tuned["clahe"]
    return [
        Method("input (no enhancement)", lambda x: x, "-", 0),
        Method("gamma", classical["gamma"], f"gamma={tuned['gamma']['gamma']}", 0),
        Method("hist_eq", classical["hist_eq"], "luma (YCrCb)", 0),
        Method(
            "clahe",
            classical["clahe"],
            f"clip={clahe_params['clip_limit']}, tile={clahe_params['tile_size']}",
            0,
        ),
    ]


def select_pairs(config: dict[str, Any], split: str) -> list[ImagePair]:
    """Return the pairs for ``split``: 'eval' (eval15) or 'val' (held-out validation)."""
    paths = config["paths"]
    if split == "eval":
        return list_pairs(paths["data_root"], paths["eval_subset"])
    pairs = list_pairs(paths["data_root"], paths["train_subset"])
    _, val = split_train_val(
        pairs, paths["val_split_file"], config["data"]["val_size"], config["seed"]
    )
    return val


def evaluate_method(
    method: Method, images: list[tuple[np.ndarray, np.ndarray]], warmup: int
) -> MethodResult:
    """Score one method on preloaded image pairs and time it per image (excluding I/O)."""
    for low, _ in images[:warmup]:
        method.enhance(low)

    psnr, ssim, times = [], [], []
    for low, high in images:
        start = time.perf_counter()
        output = method.enhance(low)
        times.append((time.perf_counter() - start) * 1000.0)
        p, s = psnr_ssim(output, high)
        psnr.append(p)
        ssim.append(s)
    return MethodResult(method, np.array(psnr), np.array(ssim), np.array(times))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    # Sample standard deviation: the images are a sample of possible low-light scenes.
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def write_results(
    results: list[MethodResult],
    pairs: list[ImagePair],
    split_label: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Write per-image scores, the summary CSV and the Markdown table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{stem}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "image", "psnr", "ssim", "cpu_ms"])
        for r in results:
            for i, pair in enumerate(pairs):
                writer.writerow(
                    [
                        r.method.name,
                        pair.name,
                        f"{r.psnr[i]:.4f}",
                        f"{r.ssim[i]:.4f}",
                        f"{r.cpu_ms[i]:.2f}",
                    ]
                )

    summary = []
    for r in results:
        psnr_m, psnr_s = _mean_std(r.psnr)
        ssim_m, ssim_s = _mean_std(r.ssim)
        summary.append(
            {
                "method": r.method.name,
                "settings": r.method.settings,
                "psnr_mean": psnr_m,
                "psnr_std": psnr_s,
                "ssim_mean": ssim_m,
                "ssim_std": ssim_s,
                "cpu_ms_mean": float(r.cpu_ms.mean()),
                "learned_params": r.method.learned_params,
            }
        )

    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

    height, width = load_pair(pairs[0])[0].shape[:2]
    lines = [
        f"# Results on {split_label}",
        "",
        f"- Images: {len(pairs)} (first image {width}x{height})",
        "- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian "
        "window (sigma 1.5), averaged over channels. Mean ± sample std over images.",
        "- CPU time: mean wall-clock time per image for the enhancement call only (no file I/O).",
        f"- Machine: {platform.processor() or platform.machine()}, {platform.system()}; "
        f"Python {platform.python_version()}, OpenCV {cv2.__version__}, "
        f"scikit-image {skimage.__version__}",
        f"- Generated: {date.today().isoformat()} by `python -m src.evaluate`",
        "",
        "| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | CPU ms / image | Learned params |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['settings']} "
            f"| {row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} "
            f"| {row['ssim_mean']:.3f} ± {row['ssim_std']:.3f} "
            f"| {row['cpu_ms_mean']:.1f} | {row['learned_params']:,} |"
        )
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate enhancement methods with PSNR/SSIM.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--split",
        choices=["eval", "val"],
        default="eval",
        help="'eval' = eval15 test set (final results), 'val' = validation split",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = select_pairs(config, args.split)
    images = [load_pair(p) for p in tqdm(pairs, desc="Loading images")]
    methods = build_methods(config)

    results = [
        evaluate_method(m, images, config["evaluation"]["timing_warmup"])
        for m in tqdm(methods, desc="Evaluating")
    ]

    if args.split == "eval":
        label, stem = f"{config['paths']['eval_subset']} (test set)", "results"
    else:
        label, stem = "the validation split", "val_results"
    write_results(results, pairs, label, Path(config["paths"]["results_dir"]), stem)

    for r in results:
        print(
            f"{r.method.name:<24} PSNR {r.psnr.mean():6.2f}  SSIM {r.ssim.mean():.3f}  "
            f"{r.cpu_ms.mean():7.1f} ms"
        )


if __name__ == "__main__":
    main()
