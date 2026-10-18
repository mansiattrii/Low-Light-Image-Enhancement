"""Qualitative figures: comparison grid, failure cases, training curves and own photos.

Usage:
    python -m src.figures --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure

from src.config import load_config
from src.data import IMAGE_EXTENSIONS, list_pairs, load_pair, read_image
from src.enhance import limit_size
from src.evaluate import Method, build_methods, zerodce_variants
from src.metrics import psnr_ssim
from src.zerodce import enhance_array, load_model

TEXT_MUTED = "#52514e"


def _method_columns(config: dict[str, Any]) -> list[Method]:
    return [m for m in build_methods(config) if m.name != "input (no enhancement)"]


def _show(
    ax: Any,
    image: np.ndarray,
    title: str | None = None,
    caption: str | None = None,
    interpolation: str | None = None,
) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0), interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if caption:
        ax.set_xlabel(caption, fontsize=8, color=TEXT_MUTED)


def comparison_grid(config: dict[str, Any], out_path: Path) -> None:
    """Input, every method and the reference for the configured eval images."""
    by_name = {p.name: p for p in list_pairs(config["paths"]["data_root"], "eval15")}
    names = config["evaluation"]["figure_images"]
    methods = _method_columns(config)
    cols = len(methods) + 2

    fig = Figure(figsize=(2.3 * cols, 1.75 * len(names) + 0.5), layout="constrained")
    axes = fig.subplots(len(names), cols, squeeze=False)
    for row, name in enumerate(names):
        low, high = load_pair(by_name[name])
        p, s = psnr_ssim(low, high)
        _show(axes[row, 0], low, "Input" if row == 0 else None, f"{name}  {p:.2f} dB / {s:.3f}")
        for col, method in enumerate(methods, start=1):
            output = method.enhance(low)
            p, s = psnr_ssim(output, high)
            _show(
                axes[row, col], output, method.name if row == 0 else None, f"{p:.2f} dB / {s:.3f}"
            )
        _show(axes[row, -1], high, "Reference" if row == 0 else None)
    fig.suptitle("eval15 comparison (caption: PSNR / SSIM against the reference)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)


def failure_cases(config: dict[str, Any], out_path: Path) -> list[dict[str, Any]]:
    """For each method, the eval15 image with its lowest PSNR: input, output and reference."""
    per_image = Path(config["paths"]["results_dir"]) / "results_per_image.csv"
    with open(per_image, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_name = {p.name: p for p in list_pairs(config["paths"]["data_root"], "eval15")}
    methods = _method_columns(config)

    cases = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method.name]
        if not method_rows:
            continue
        worst = min(method_rows, key=lambda r: float(r["psnr"]))
        cases.append({**worst, "method": method})

    fig = Figure(figsize=(9, 2.3 * len(cases) + 0.4), layout="constrained")
    axes = fig.subplots(len(cases), 3, squeeze=False)
    for row, case in enumerate(cases):
        low, high = load_pair(by_name[case["image"]])
        output = case["method"].enhance(low)
        label = f"{case['method'].name} on {case['image']}"
        _show(axes[row, 0], low, "Input" if row == 0 else None, case["image"])
        _show(
            axes[row, 1],
            output,
            "Method output" if row == 0 else None,
            f"{label}: {float(case['psnr']):.2f} dB / {float(case['ssim']):.3f}, "
            f"brightness {float(case['brightness']):.3f} (reference {high.mean():.3f})",
        )
        _show(axes[row, 2], high, "Reference" if row == 0 else None)
    fig.suptitle("Worst eval15 image (lowest PSNR) for each method", fontsize=11)
    fig.savefig(out_path, dpi=110)
    return [
        {k: v for k, v in c.items() if k != "method"} | {"method": c["method"].name} for c in cases
    ]


def own_photos(config: dict[str, Any], folder: Path, out_path: Path) -> bool:
    """Qualitative grid for photos without ground truth; returns False if there are none."""
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        return False
    methods = _method_columns(config)
    max_side = config["inference"]["max_side"]
    fig = Figure(figsize=(2.6 * (len(methods) + 1), 2.2 * len(files) + 0.4), layout="constrained")
    axes = fig.subplots(len(files), len(methods) + 1, squeeze=False)
    for row, path in enumerate(files):
        image = limit_size(read_image(path), min(max_side, 1024))
        _show(axes[row, 0], image, "Input" if row == 0 else None, path.name)
        for col, method in enumerate(methods, start=1):
            _show(axes[row, col], method.enhance(image), method.name if row == 0 else None)
    fig.suptitle("Own photos (no reference: qualitative only)", fontsize=11)
    fig.savefig(out_path, dpi=110)
    return True


def checkpoint_comparison(config: dict[str, Any], out_path: Path) -> bool:
    """Best vs last checkpoint of each Zero-DCE run; returns False if the runs are missing."""
    zcfg = config["zerodce"]
    columns = []
    for name, variant in zerodce_variants(config).items():
        run_dir = Path(variant.get("run_dir") or "")
        for checkpoint in ("best.pth", "last.pth"):
            path = run_dir / checkpoint
            if variant.get("run_dir") and path.exists():
                epoch = torch.load(path, map_location="cpu", weights_only=False)["epoch"]
                model = load_model(path, "cpu", zcfg["features"], zcfg["iterations"])
                columns.append((f"{name} epoch {epoch}", model))
    if not columns:
        return False

    by_name = {p.name: p for p in list_pairs(config["paths"]["data_root"], "eval15")}
    names = config["evaluation"]["figure_images"]
    cols = len(columns) + 2
    fig = Figure(figsize=(2.3 * cols, 1.75 * len(names) + 0.5), layout="constrained")
    axes = fig.subplots(len(names), cols, squeeze=False)
    for row, name in enumerate(names):
        low, high = load_pair(by_name[name])
        _show(axes[row, 0], low, "Input" if row == 0 else None, name)
        for col, (label, model) in enumerate(columns, start=1):
            output = enhance_array(model, low)
            p, _ = psnr_ssim(output, high)
            caption = f"{p:.2f} dB, brightness {output.mean():.2f}"
            _show(axes[row, col], output, label if row == 0 else None, caption)
        caption = f"brightness {high.mean():.2f}"
        _show(axes[row, -1], high, "Reference" if row == 0 else None, caption)
    fig.suptitle("Best (by validation PSNR) vs last checkpoint of each run", fontsize=11)
    fig.savefig(out_path, dpi=110)
    return True


def collect_training_logs(config: dict[str, Any], out_dir: Path) -> list[str]:
    """Copy each run's history and loss curves next to its weights into results/training."""
    copied = []
    for name, variant in zerodce_variants(config).items():
        if not variant.get("run_dir") or not Path(variant["run_dir"]).is_dir():
            continue
        run_dir = Path(variant["run_dir"])
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("history.csv", "loss_curves.png", "config_used.yaml"):
            if (run_dir / file).exists():
                shutil.copy2(run_dir / file, target / file)
        copied.append(name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative result figures.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    figures = Path(config["paths"]["results_dir"]) / "figures"

    comparison_grid(config, figures / "comparison_grid.png")
    for case in failure_cases(config, figures / "failure_cases.png"):
        print(f"Worst for {case['method']}: {case['image']} ({float(case['psnr']):.2f} dB)")
    if checkpoint_comparison(config, figures / "checkpoint_comparison.png"):
        print("Saved checkpoint comparison figure")
    if own_photos(config, Path("data/own"), figures / "own_photos.png"):
        print("Saved own photos figure")
    copied = collect_training_logs(config, Path(config["paths"]["results_dir"]) / "training")
    print(f"Figures saved to {figures}; training logs copied for: {copied or 'none'}")


if __name__ == "__main__":
    main()
