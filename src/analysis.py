"""Diagnose why validation PSNR falls during Zero-DCE training.

Compares each run's best and last checkpoints on the validation split. Besides plain
PSNR it reports PSNR after removing the global brightness error and after removing the
per-channel (colour) error, using the reference's means. Those corrections use the
references, so they are a diagnostic only and never a reported result.

Usage:
    python -m src.analysis --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.config import load_config
from src.data import list_pairs, load_pair, split_train_val
from src.evaluate import zerodce_variants
from src.metrics import psnr_ssim
from src.zerodce import enhance_array, load_model


def _luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def image_stats(output: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Plain and corrected PSNR plus brightness, colour-cast and contrast measures."""
    psnr, ssim = psnr_ssim(output, reference)
    gain = reference.mean() / max(output.mean(), 1e-6)
    psnr_brightness, _ = psnr_ssim(np.clip(output * gain, 0, 1), reference)
    channel_gain = reference.mean(axis=(0, 1)) / np.maximum(output.mean(axis=(0, 1)), 1e-6)
    psnr_colour, _ = psnr_ssim(np.clip(output * channel_gain, 0, 1), reference)

    def chroma_offset(image: np.ndarray) -> np.ndarray:
        means = image.mean(axis=(0, 1))
        return means - means.mean()

    return {
        "psnr": psnr,
        "ssim": ssim,
        "psnr_brightness_corrected": psnr_brightness,
        "psnr_colour_corrected": psnr_colour,
        "brightness": float(output.mean()),
        "reference_brightness": float(reference.mean()),
        # Distance between the output's and the reference's channel balance (0 = same tint).
        "colour_cast_error": float(
            np.linalg.norm(chroma_offset(output) - chroma_offset(reference))
        ),
        "contrast": float(_luma(output).std()),
        "reference_contrast": float(_luma(reference).std()),
    }


def analyse(config: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Evaluate best and last checkpoints of every Zero-DCE run on the validation split."""
    paths, zcfg = config["paths"], config["zerodce"]
    pairs = list_pairs(paths["data_root"], paths["train_subset"])
    _, val = split_train_val(
        pairs, paths["val_split_file"], config["data"]["val_size"], config["seed"]
    )
    images = [load_pair(p) for p in val]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for name, variant in zerodce_variants(config).items():
        run_dir = Path(variant.get("run_dir", ""))
        for checkpoint in ("best.pth", "last.pth"):
            path = run_dir / checkpoint
            if not variant.get("run_dir") or not path.exists():
                continue
            epoch = torch.load(path, map_location="cpu", weights_only=False)["epoch"]
            model = load_model(path, device, zcfg["features"], zcfg["iterations"])
            stats = [image_stats(enhance_array(model, low, device), high) for low, high in images]
            rows.append(
                {
                    "run": name,
                    "checkpoint": checkpoint.removesuffix(".pth"),
                    "epoch": epoch,
                    **{k: float(np.mean([s[k] for s in stats])) for k in stats[0]},
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "checkpoint_drift.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows
        )

    lines = [
        f"# Checkpoint drift on the validation split ({len(val)} images)",
        "",
        "Means over images. Corrected PSNR rescales the output to the reference's mean "
        "brightness (global) or per-channel means (colour); diagnostic only.",
        "",
        "| Run | Checkpoint | Epoch | PSNR | Brightness-corrected PSNR | Colour-corrected PSNR "
        "| SSIM | Brightness (ref) | Colour-cast error | Contrast (ref) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['checkpoint']} | {r['epoch']} | {r['psnr']:.2f} "
            f"| {r['psnr_brightness_corrected']:.2f} | {r['psnr_colour_corrected']:.2f} "
            f"| {r['ssim']:.3f} | {r['brightness']:.3f} ({r['reference_brightness']:.3f}) "
            f"| {r['colour_cast_error']:.3f} "
            f"| {r['contrast']:.3f} ({r['reference_contrast']:.3f}) |"
        )
    (out_dir / "checkpoint_drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best and last Zero-DCE checkpoints.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    with torch.inference_mode():
        rows = analyse(config, Path(config["paths"]["results_dir"]) / "analysis")
    print(yaml.safe_dump(rows, sort_keys=False))


if __name__ == "__main__":
    main()
