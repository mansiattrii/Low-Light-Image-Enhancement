"""Denoising study: where in the pipeline should denoising go, before or after Zero-DCE?

Compares enhance only, enhance -> denoise and denoise -> enhance with OpenCV's non-local
means and bilateral filters. Filter strengths are tuned on the validation split (each
filter and order separately, by mean PSNR), then every pipeline is scored on eval15.

Usage:
    python -m src.denoise --config config.yaml               # tune, evaluate, figure
    python -m src.denoise --config config.yaml --skip-tuning  # reuse tuned_denoise.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from tqdm import tqdm

from src.classical import Enhancer
from src.config import load_config
from src.data import list_pairs, load_pair, to_uint8
from src.evaluate import Method, evaluate_method, select_pairs, write_results
from src.figures import _show
from src.metrics import psnr_ssim
from src.zerodce import count_parameters, enhance_array, load_model

FILTERS = ("nlm", "bilateral")
ORDERS = ("after", "before")  # denoise after enhancement or before it


def nlm_denoise(image: np.ndarray, h: float, template: int = 7, search: int = 21) -> np.ndarray:
    """OpenCV non-local means on an RGB image in [0, 1]; ``h`` is in 8-bit intensity levels.

    The colour version works in LAB and expects BGR input, so the channels are swapped
    around the call. The same strength is used for luminance and colour.
    """
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, template, search)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def bilateral_denoise(image: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Bilateral filter on a float RGB image; ``sigma_color`` is in [0, 1] intensity units.

    The window diameter is derived by OpenCV from ``sigma_space``.
    """
    out = cv2.bilateralFilter(
        np.ascontiguousarray(image, dtype=np.float32), 0, sigma_color, sigma_space
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_denoiser(name: str, params: dict[str, Any], config: dict[str, Any]) -> Enhancer:
    """Build a denoiser from its name and tuned parameters."""
    if name == "nlm":
        dcfg = config["denoise"]
        return lambda x: nlm_denoise(
            x, params["h"], dcfg["nlm_template_window"], dcfg["nlm_search_window"]
        )
    if name == "bilateral":
        return lambda x: bilateral_denoise(x, params["sigma_color"], params["sigma_space"])
    raise ValueError(f"Unknown denoiser {name!r}; choose from {FILTERS}")


def compose(enhancer: Enhancer, denoiser: Enhancer, order: str) -> Enhancer:
    """Chain enhancement and denoising: ``after`` = enhance then denoise, ``before`` = reverse."""
    if order == "after":
        return lambda x: denoiser(enhancer(x))
    if order == "before":
        return lambda x: enhancer(denoiser(x))
    raise ValueError(f"Unknown order {order!r}; choose from {ORDERS}")


def pipeline_name(filter_name: str, order: str) -> str:
    return f"zerodce->{filter_name}" if order == "after" else f"{filter_name}->zerodce"


def noise_sigma(image: np.ndarray) -> float:
    """Estimated noise standard deviation of an image in [0, 1], in 8-bit levels.

    Donoho's wavelet estimate: the finest diagonal Haar detail of a smooth image is close to
    zero, so its spread is mostly noise, and ``median(|d|) / 0.6745`` estimates the noise
    standard deviation while ignoring the few large values at edges. Averaged over channels.
    It needs no reference, so it measures noise rather than closeness to the reference;
    strong texture also counts as "noise", so compare pipelines on the same images.
    Computed before 8-bit rounding: on very smooth outputs most rounded details are exactly
    zero, and the median would collapse to 0.
    """
    x = np.clip(image, 0.0, 1.0).astype(np.float64) * 255.0
    h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    x = x[:h, :w]
    # Orthonormal Haar diagonal detail of each 2x2 block: (a - b - c + d) / 2.
    diagonal = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    sigmas = np.median(np.abs(diagonal), axis=(0, 1)) / 0.6745
    return float(sigmas.mean())


def _candidates(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    dcfg = config["denoise"]
    return [("nlm", {"h": h}) for h in dcfg["nlm_h_grid"]] + [
        ("bilateral", {"sigma_color": c, "sigma_space": s})
        for c, s in itertools.product(
            dcfg["bilateral_sigma_color_grid"], dcfg["bilateral_sigma_space_grid"]
        )
    ]


def tune(config: dict[str, Any], enhancer: Enhancer) -> dict[str, Any]:
    """Grid-search each filter's strength separately for each order on the validation split.

    The two orders need different strengths: before enhancement the noise is a few 8-bit
    levels, after enhancement it has been amplified many times.
    """
    paths = config["paths"]
    val = select_pairs(config, "val")
    images = [load_pair(p) for p in tqdm(val, desc="Loading validation")]
    # Zero-DCE output is the same for every "after" candidate, so compute it once.
    enhanced = [enhancer(low) for low, _ in tqdm(images, desc="Enhancing validation")]

    rows = []
    for (filter_name, params), order in tqdm(
        list(itertools.product(_candidates(config), ORDERS)), desc="Grid search"
    ):
        denoiser = make_denoiser(filter_name, params, config)
        if order == "after":
            outputs = [denoiser(e) for e in enhanced]
        else:
            outputs = [enhancer(denoiser(low)) for low, _ in images]
        scores = np.array(
            [psnr_ssim(o, high) for o, (_, high) in zip(outputs, images, strict=True)]
        )
        rows.append((filter_name, order, params, scores[:, 0].mean(), scores[:, 1].mean()))

    tuned: dict[str, Any] = {
        "selection": {
            "split": f"validation ({len(val)} pairs from {paths['train_subset']})",
            "criterion": "mean PSNR",
            "enhancer": f"zerodce ({paths['weights']})",
        }
    }
    for filter_name, order in itertools.product(FILTERS, ORDERS):
        _, _, params, psnr, ssim = max(
            (r for r in rows if r[0] == filter_name and r[1] == order), key=lambda r: r[3]
        )
        tuned[pipeline_name(filter_name, order)] = {
            **params,
            "val_psnr": round(float(psnr), 4),
            "val_ssim": round(float(ssim), 4),
        }

    out_dir = Path(config["denoise"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuned_denoise.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False)
    with open(out_dir / "tuning_denoise_val.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "params", "psnr", "ssim"])
        for filter_name, order, params, psnr, ssim in rows:
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            writer.writerow(
                [pipeline_name(filter_name, order), label, f"{psnr:.4f}", f"{ssim:.4f}"]
            )
    return tuned


def build_pipelines(
    config: dict[str, Any], tuned: dict[str, Any], enhancer: Enhancer, params: int
) -> list[Method]:
    """Enhance only, plus every (filter, order) pipeline with its tuned parameters."""
    methods = [Method("zerodce (no denoising)", enhancer, "-", params)]
    for filter_name, order in itertools.product(FILTERS, ORDERS):
        name = pipeline_name(filter_name, order)
        chosen = {k: v for k, v in tuned[name].items() if not k.startswith("val_")}
        settings = ", ".join(f"{k}={v}" for k, v in chosen.items())
        denoiser = make_denoiser(filter_name, chosen, config)
        methods.append(Method(name, compose(enhancer, denoiser, order), settings, params))
    return methods


def write_noise_table(
    methods: list[Method],
    images: list[tuple[np.ndarray, np.ndarray]],
    names: list[str],
    out_dir: Path,
) -> dict[str, np.ndarray]:
    """Per-image estimated noise of each pipeline's output and of the references."""
    sigmas = {"reference": np.array([noise_sigma(high) for _, high in images])}
    for method in tqdm(methods, desc="Estimating noise"):
        sigmas[method.name] = np.array([noise_sigma(method.enhance(low)) for low, _ in images])

    with open(out_dir / "denoise_noise_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "image", "noise_sigma"])
        for name, values in sigmas.items():
            for image_name, value in zip(names, values, strict=True):
                writer.writerow([name, image_name, f"{value:.4f}"])

    lines = [
        "",
        "## Estimated noise in the output",
        "",
        "Noise standard deviation in 8-bit levels, estimated from each saved output without a "
        "reference (median absolute Haar diagonal detail / 0.6745), mean ± sample std over "
        "images. Lower means smoother; it cannot tell noise from fine texture, so an output "
        "well below the reference has lost real detail.",
        "",
        "| Output | Noise σ (8-bit levels) |",
        "|---|---|",
    ]
    for name, values in sigmas.items():
        lines.append(f"| {name} | {values.mean():.2f} ± {values.std(ddof=1):.2f} |")
    with open(out_dir / "denoise_results.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return sigmas


def noisiest_window(output: np.ndarray, reference: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the ``size`` x ``size`` window where the output's local luma std
    exceeds the reference's the most, i.e. where enhancement amplified noise the most."""

    def local_std(image: np.ndarray) -> np.ndarray:
        luma = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.float32), cv2.COLOR_RGB2GRAY)
        mean = cv2.blur(luma, (size, size), borderType=cv2.BORDER_REFLECT)
        sq = cv2.blur(luma * luma, (size, size), borderType=cv2.BORDER_REFLECT)
        return np.sqrt(np.maximum(sq - mean * mean, 0.0))

    excess = local_std(output) - local_std(reference)
    h, w = excess.shape
    half = size // 2
    # Only centres whose window lies fully inside the image.
    valid = excess[half : h - size + half + 1, half : w - size + half + 1]
    y, x = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return int(y), int(x)


def noise_figure(config: dict[str, Any], methods: list[Method], out_path: Path) -> None:
    """Full reference with the crop marked, then zoomed crops of every pipeline's output."""
    dcfg = config["denoise"]
    by_name = {p.name: p for p in list_pairs(config["paths"]["data_root"], "eval15")}
    names, size = dcfg["figure_images"], dcfg["figure_crop"]
    cols = len(methods) + 2

    fig = Figure(figsize=(2.1 * cols, 2.3 * len(names) + 0.5), layout="constrained")
    axes = fig.subplots(len(names), cols, squeeze=False)
    for row, name in enumerate(names):
        low, high = load_pair(by_name[name])
        outputs = [m.enhance(low) for m in methods]
        y, x = noisiest_window(outputs[0], high, size)

        def crop(image: np.ndarray, y: int = y, x: int = x) -> np.ndarray:
            return image[y : y + size, x : x + size]

        _show(axes[row, 0], high, "Reference (full)" if row == 0 else None, name)
        axes[row, 0].add_patch(
            Rectangle((x, y), size, size, fill=False, edgecolor="#e8a33d", linewidth=1.5)
        )
        for col, (method, output) in enumerate(zip(methods, outputs, strict=True), start=1):
            p, s = psnr_ssim(output, high)
            caption = f"{p:.2f} dB / {s:.3f}\nnoise σ {noise_sigma(output):.1f}"
            _show(
                axes[row, col],
                crop(output),
                method.name if row == 0 else None,
                caption,
                interpolation="nearest",
            )
        _show(
            axes[row, -1],
            crop(high),
            "Reference (crop)" if row == 0 else None,
            f"noise σ {noise_sigma(high):.1f}",
            interpolation="nearest",
        )
    fig.suptitle(
        f"Noise amplification, {size}x{size} crops where Zero-DCE adds the most local variation "
        "(caption: full-image PSNR / SSIM, estimated noise σ of the full image)",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare denoising before/after Zero-DCE.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Reuse the parameters in tuned_denoise.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    zcfg, ecfg = config["zerodce"], config["evaluation"]
    out_dir = Path(config["denoise"]["results_dir"])
    model = load_model(config["paths"]["weights"], "cpu", zcfg["features"], zcfg["iterations"])

    def enhancer(x: np.ndarray) -> np.ndarray:
        return enhance_array(model, x)

    with torch.inference_mode():
        if args.skip_tuning:
            with open(out_dir / "tuned_denoise.yaml", encoding="utf-8") as f:
                tuned = yaml.safe_load(f)
        else:
            tuned = tune(config, enhancer)
        print(yaml.safe_dump(tuned, sort_keys=False))

        methods = build_pipelines(config, tuned, enhancer, count_parameters(model))
        pairs = select_pairs(config, "eval")
        images = [load_pair(p) for p in tqdm(pairs, desc="Loading eval15")]
        results = [
            evaluate_method(m, images, ecfg["timing_warmup"], ecfg["timing_repeats"])
            for m in tqdm(methods, desc="Evaluating")
        ]
        label = f"{config['paths']['eval_subset']} (test set): denoising study"
        write_results(
            results,
            images,
            pairs,
            label,
            out_dir,
            "denoise_results",
            ecfg["timing_repeats"],
            command="python -m src.denoise",
        )
        sigmas = write_noise_table(methods, images, [p.name for p in pairs], out_dir)
        noise_figure(config, methods, out_dir / "noise_amplification.png")

    for r in results:
        print(
            f"{r.method.name:<24} PSNR {r.psnr.mean():6.2f}  SSIM {r.ssim.mean():.3f}  "
            f"noise {sigmas[r.method.name].mean():5.2f}  {r.cpu_ms.mean():7.1f} ms"
        )
    print(f"reference noise {sigmas['reference'].mean():.2f}; results in {out_dir}")


if __name__ == "__main__":
    main()
