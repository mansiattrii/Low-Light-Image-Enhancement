"""Train Zero-DCE on LOL low-light images with zero-reference losses.

Usage:
    python -m src.train --config config.yaml
    python -m src.train --config config.yaml --data-root /content/data/LOL \
        --output-dir /content/drive/MyDrive/lowlight/checkpoints

Training resumes automatically from ``<output-dir>/last.pth`` if it exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import LowLightCrops, list_pairs, load_pair, split_train_val, to_tensor
from src.losses import LossWeights, ZeroDCELoss
from src.metrics import psnr_ssim
from src.zerodce import DCENet, count_parameters, init_weights

LOSS_TERMS = ("spatial", "exposure", "color", "illumination")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_criterion(cfg: dict[str, Any]) -> ZeroDCELoss:
    """Create the combined loss from the ``zerodce`` config section."""
    formulation = cfg.get("loss_formulation", "paper")
    weights_key = "official_loss_weights" if formulation == "official" else "loss_weights"
    return ZeroDCELoss(
        weights=LossWeights(**cfg[weights_key]),
        exposure_level=cfg["exposure_level"],
        exposure_patch=cfg["exposure_patch"],
        spatial_region=cfg["spatial_region"],
        iterations=cfg["iterations"],
        formulation=formulation,
    )


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write a checkpoint atomically, so a disconnect mid-save cannot corrupt the old one."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(
    model: DCENet,
    criterion: ZeroDCELoss,
    images: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, float]:
    """Zero-reference loss and PSNR/SSIM against the references, on full-size images."""
    model.eval()
    losses, psnrs, ssims = [], [], []
    for low, high in images:
        x = to_tensor(low).unsqueeze(0).to(device)
        enhanced, alphas = model(x)
        loss, _ = criterion(x, enhanced, alphas)
        losses.append(loss.item())
        output = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        p, s = psnr_ssim(output, high)
        psnrs.append(p)
        ssims.append(s)
    return {
        "val_loss": float(np.mean(losses)),
        "val_psnr": float(np.mean(psnrs)),
        "val_ssim": float(np.mean(ssims)),
    }


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Save total loss, validation PSNR and each loss term as small multiples."""
    epochs = [h["epoch"] for h in history]
    train_color, val_color, muted = "#2a78d6", "#eb6834", "#52514e"
    fig = Figure(figsize=(13, 7), layout="constrained")
    total_ax, psnr_ax, *term_axes = fig.subplots(2, 3).flat

    total_ax.plot(
        epochs, [h["train_loss"] for h in history], color=train_color, lw=2, label="train"
    )
    total_ax.plot(
        epochs, [h["val_loss"] for h in history], color=val_color, lw=2, label="validation"
    )
    total_ax.set_title("Total zero-reference loss")
    total_ax.legend(frameon=False)

    best = max(history, key=lambda h: h["val_psnr"])
    psnr_ax.plot(epochs, [h["val_psnr"] for h in history], color=val_color, lw=2)
    psnr_ax.scatter(
        [best["epoch"]],
        [best["val_psnr"]],
        s=60,
        color=val_color,
        zorder=3,
        edgecolor="white",
        linewidth=2,
    )
    psnr_ax.text(
        0.98,
        0.04,
        f"best {best['val_psnr']:.2f} dB at epoch {best['epoch']}",
        transform=psnr_ax.transAxes,
        ha="right",
        fontsize=9,
        color=muted,
    )
    psnr_ax.set_title("Validation PSNR (dB)")

    for ax, term in zip(term_axes, LOSS_TERMS, strict=True):
        ax.plot(epochs, [h[f"train_{term}"] for h in history], color=train_color, lw=2)
        ax.set_title(f"Train {term} loss (unweighted)")

    for ax in fig.axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=120)


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(
    config: dict[str, Any],
    data_root: str | None = None,
    output_dir: str | None = None,
    epochs: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
) -> list[dict[str, float]]:
    """Run (or resume) training and return the per-epoch history."""
    paths, tcfg, zcfg = config["paths"], config["train"], config["zerodce"]
    data_root = data_root or paths["data_root"]
    out = Path(output_dir or tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    epochs = epochs or tcfg["epochs"]
    num_workers = tcfg["num_workers"] if num_workers is None else num_workers
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = config["seed"]
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {**config, "run": {"epochs": epochs, "device": str(dev)}}, f, sort_keys=False
        )

    pairs = list_pairs(data_root, paths["train_subset"])
    train_pairs, val_pairs = split_train_val(
        pairs, paths["val_split_file"], config["data"]["val_size"], seed
    )
    train_set = LowLightCrops(train_pairs, tcfg["crop_size"], augment=True)
    val_images = [load_pair(p) for p in val_pairs]

    seed_everything(seed)
    model = DCENet(zcfg["features"], zcfg["iterations"])
    init_weights(model)
    model.to(dev)
    criterion = build_criterion(zcfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    history: list[dict[str, float]] = []
    start_epoch, best_psnr = 1, -float("inf")
    last_path = out / "last.pth"
    if last_path.exists():
        state = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_psnr = state["history"], state["best_psnr"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    print(
        f"Device: {dev} | params: {count_parameters(model):,} | "
        f"train: {len(train_pairs)} | val: {len(val_pairs)} | epochs: {epochs}"
    )

    generator = torch.Generator()
    loader = DataLoader(
        train_set,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=dev.type == "cuda",
        generator=generator,
    )

    for epoch in range(start_epoch, epochs + 1):
        # Seeding per epoch makes every epoch's crops and order depend only on (seed, epoch),
        # so a resumed run sees the same data as an uninterrupted one.
        torch.manual_seed(seed + epoch)
        generator.manual_seed(seed + epoch)

        model.train()
        start = time.perf_counter()
        sums = dict.fromkeys(("loss", *LOSS_TERMS), 0.0)
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(dev, non_blocking=True)
            enhanced, alphas = model(batch)
            loss, terms = criterion(batch, enhanced, alphas)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg["grad_clip_norm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            optimizer.step()
            sums["loss"] += loss.item()
            for name in LOSS_TERMS:
                sums[name] += terms[name]

        n = len(loader)
        record = {
            "epoch": epoch,
            "train_loss": sums["loss"] / n,
            **{f"train_{name}": sums[name] / n for name in LOSS_TERMS},
            **validate(model, criterion, val_images, dev),
            "seconds": time.perf_counter() - start,
        }
        history.append(record)

        is_best = record["val_psnr"] > best_psnr
        if is_best:
            best_psnr = record["val_psnr"]
            save_checkpoint(
                out / "best.pth",
                {"model": model.state_dict(), "epoch": epoch, "val_psnr": best_psnr},
            )
            save_checkpoint(out / "zerodce.pth", model.state_dict())
        save_checkpoint(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
                "history": history,
            },
        )
        write_history_csv(history, out / "history.csv")
        plot_history(history, out / "loss_curves.png")

        print(
            f"Epoch {epoch:3d} | train {record['train_loss']:.4f} | "
            f"val {record['val_loss']:.4f} | PSNR {record['val_psnr']:.2f} | "
            f"SSIM {record['val_ssim']:.3f} | {record['seconds']:.0f}s"
            + ("  *best*" if is_best else "")
        )

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Zero-DCE with zero-reference losses.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", help="Override paths.data_root")
    parser.add_argument("--output-dir", help="Override train.output_dir (checkpoints, curves)")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--num-workers", type=int, help="Override train.num_workers")
    parser.add_argument(
        "--exposure-level", type=float, help="Override zerodce.exposure_level (ablation)"
    )
    parser.add_argument(
        "--loss-formulation",
        choices=["paper", "official"],
        help="Override zerodce.loss_formulation (official = formulas and weights of the code)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.exposure_level is not None:
        config["zerodce"]["exposure_level"] = args.exposure_level
    if args.loss_formulation is not None:
        config["zerodce"]["loss_formulation"] = args.loss_formulation
    train(
        config,
        args.data_root,
        args.output_dir,
        args.epochs,
        args.device,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
