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
import torch
from tqdm import tqdm

from src.classical import Enhancer, build_classical_methods, load_tuned_params
from src.config import load_config
from src.data import ImagePair, list_pairs, load_pair, split_train_val
from src.metrics import psnr_ssim
from src.zerodce import count_parameters, enhance_array, load_model


@dataclass
class Method:
    name: str
    enhance: Enhancer
    settings: str
    learned_params: int
    gpu_enhance: Enhancer | None = None


@dataclass
class MethodResult:
    method: Method
    psnr: np.ndarray
    ssim: np.ndarray
    brightness: np.ndarray
    cpu_ms: np.ndarray
    gpu_ms: np.ndarray | None


def zerodce_variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The main Zero-DCE weights plus any extra variants listed in the config."""
    return {
        "zerodce": {
            "weights": config["paths"]["weights"],
            "settings": f"E={config['zerodce']['exposure_level']} (paper)",
            "run_dir": config["evaluation"].get("zerodce_run_dir", ""),
        },
        **config["evaluation"].get("zerodce_variants", {}),
    }


def build_methods(config: dict[str, Any]) -> list[Method]:
    """Assemble every method to evaluate, including the unprocessed input as a reference.

    Zero-DCE variants whose weights file is missing are skipped with a message.
    """
    tuned = load_tuned_params(config["paths"]["tuned_params"])
    classical = build_classical_methods(tuned)
    clahe_params = tuned["clahe"]
    methods = [
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

    zcfg = config["zerodce"]
    for name, variant in zerodce_variants(config).items():
        weights = variant["weights"]
        if not Path(weights).exists():
            print(f"Skipping {name}: {weights} not found")
            continue
        model = load_model(weights, "cpu", zcfg["features"], zcfg["iterations"])
        gpu_enhance = None
        if torch.cuda.is_available():
            gpu_model = load_model(weights, "cuda", zcfg["features"], zcfg["iterations"])

            def gpu_enhance(x: np.ndarray, m=gpu_model) -> np.ndarray:
                return enhance_array(m, x, "cuda")

        methods.append(
            Method(
                name,
                lambda x, m=model: enhance_array(m, x),
                variant["settings"],
                count_parameters(model),
                gpu_enhance,
            )
        )
    return methods


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


def _time_ms(enhance: Enhancer, image: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    output = enhance(image)
    return output, (time.perf_counter() - start) * 1000.0


def evaluate_method(
    method: Method, images: list[tuple[np.ndarray, np.ndarray]], warmup: int
) -> MethodResult:
    """Score one method on preloaded image pairs and time it per image (excluding I/O)."""
    for low, _ in images[:warmup]:
        method.enhance(low)

    psnr, ssim, brightness, cpu_ms = [], [], [], []
    for low, high in images:
        output, ms = _time_ms(method.enhance, low)
        p, s = psnr_ssim(output, high)
        psnr.append(p)
        ssim.append(s)
        brightness.append(float(np.clip(output, 0.0, 1.0).mean()))
        cpu_ms.append(ms)

    gpu_ms = None
    if method.gpu_enhance is not None:
        for low, _ in images[:warmup]:
            method.gpu_enhance(low)
        # enhance_array copies the result back to the CPU, so each call includes the full GPU work.
        gpu_ms = np.array([_time_ms(method.gpu_enhance, low)[1] for low, _ in images])

    return MethodResult(
        method, np.array(psnr), np.array(ssim), np.array(brightness), np.array(cpu_ms), gpu_ms
    )


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    # Sample standard deviation: the images are a sample of possible low-light scenes.
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def write_results(
    results: list[MethodResult],
    images: list[tuple[np.ndarray, np.ndarray]],
    pairs: list[ImagePair],
    split_label: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Write per-image scores, the summary CSV and the Markdown table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{stem}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "image", "psnr", "ssim", "brightness", "cpu_ms"])
        for r in results:
            for i, pair in enumerate(pairs):
                writer.writerow(
                    [
                        r.method.name,
                        pair.name,
                        f"{r.psnr[i]:.4f}",
                        f"{r.ssim[i]:.4f}",
                        f"{r.brightness[i]:.4f}",
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
                "brightness_mean": float(r.brightness.mean()),
                "cpu_ms_mean": float(r.cpu_ms.mean()),
                "gpu_ms_mean": float(r.gpu_ms.mean()) if r.gpu_ms is not None else None,
                "learned_params": r.method.learned_params,
            }
        )

    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    k: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
                    for k, v in row.items()
                }
            )

    height, width = images[0][0].shape[:2]
    reference_brightness = float(np.mean([high.mean() for _, high in images]))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none available"
    lines = [
        f"# Results on {split_label}",
        "",
        f"- Images: {len(pairs)} (first image {width}x{height})",
        "- PSNR (dB) and SSIM on 8-bit RGB, data range [0, 1]; SSIM with an 11x11 Gaussian "
        "window (sigma 1.5), averaged over channels. Mean ± sample std over images.",
        f"- Brightness: mean output intensity in [0, 1]; the reference images average "
        f"{reference_brightness:.3f}.",
        "- Time: mean wall-clock time per image for the enhancement call only (no file I/O), "
        "after a warm-up call.",
        f"- Machine: {platform.processor() or platform.machine()}, {platform.system()}, "
        f"{torch.get_num_threads()} PyTorch threads; GPU: {gpu}. Python "
        f"{platform.python_version()}, PyTorch {torch.__version__}, OpenCV {cv2.__version__}, "
        f"scikit-image {skimage.__version__}",
        f"- Generated: {date.today().isoformat()} by `python -m src.evaluate`",
        "",
        "| Method | Settings | PSNR (dB) ↑ | SSIM ↑ | Brightness | CPU ms / image "
        "| GPU ms / image | Learned params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        gpu_ms = "–" if row["gpu_ms_mean"] is None else f"{row['gpu_ms_mean']:.1f}"
        lines.append(
            f"| {row['method']} | {row['settings']} "
            f"| {row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} "
            f"| {row['ssim_mean']:.3f} ± {row['ssim_std']:.3f} "
            f"| {row['brightness_mean']:.3f} "
            f"| {row['cpu_ms_mean']:.1f} | {gpu_ms} | {row['learned_params']:,} |"
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

    with torch.inference_mode():
        results = [
            evaluate_method(m, images, config["evaluation"]["timing_warmup"])
            for m in tqdm(methods, desc="Evaluating")
        ]

    if args.split == "eval":
        label, stem = f"{config['paths']['eval_subset']} (test set)", "results"
    else:
        label, stem = "the validation split", "val_results"
    write_results(results, images, pairs, label, Path(config["paths"]["results_dir"]), stem)

    for r in results:
        print(
            f"{r.method.name:<24} PSNR {r.psnr.mean():6.2f}  SSIM {r.ssim.mean():.3f}  "
            f"brightness {r.brightness.mean():.3f}  {r.cpu_ms.mean():7.1f} ms"
        )


if __name__ == "__main__":
    main()
