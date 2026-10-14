from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from src.train import train
from src.zerodce import load_model


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    for i in range(6):
        for kind, scale in (("low", 40), ("high", 200)):
            path = tmp_path / "LOL" / "our485" / kind / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), rng.integers(0, scale, (40, 48, 3), dtype=np.uint8))
    return {
        "seed": 1,
        "paths": {
            "data_root": str(tmp_path / "LOL"),
            "train_subset": "our485",
            "val_split_file": str(tmp_path / "val.txt"),
        },
        "data": {"val_size": 2},
        "zerodce": {
            "features": 32,
            "iterations": 8,
            "exposure_level": 0.6,
            "exposure_patch": 16,
            "spatial_region": 4,
            "loss_weights": {"spatial": 1.0, "exposure": 1.0, "color": 0.5, "illumination": 20.0},
        },
        "train": {
            "output_dir": str(tmp_path / "ckpt"),
            "crop_size": 32,
            "batch_size": 2,
            "epochs": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "grad_clip_norm": 0.1,
            "num_workers": 0,
        },
    }


def test_train_writes_outputs_and_resumes(tiny_config: dict[str, Any]) -> None:
    out = Path(tiny_config["train"]["output_dir"])
    history = train(tiny_config, device="cpu")
    assert [h["epoch"] for h in history] == [1, 2]
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_psnr"]) for h in history)
    for name in ("last.pth", "best.pth", "zerodce.pth", "history.csv", "loss_curves.png"):
        assert (out / name).exists(), name

    resumed = train(tiny_config, epochs=3, device="cpu")
    assert [h["epoch"] for h in resumed] == [1, 2, 3]
    assert resumed[:2] == history

    model = load_model(out / "zerodce.pth")
    with torch.no_grad():
        enhanced, _ = model(torch.rand(1, 3, 20, 30))
    assert enhanced.shape == (1, 3, 20, 30)


def test_resumed_epoch_matches_uninterrupted_run(
    tiny_config: dict[str, Any], tmp_path: Path
) -> None:
    straight = dict(tiny_config, train={**tiny_config["train"], "output_dir": str(tmp_path / "a")})
    full = train(straight, epochs=2, device="cpu")

    split = dict(tiny_config, train={**tiny_config["train"], "output_dir": str(tmp_path / "b")})
    train(split, epochs=1, device="cpu")
    resumed = train(split, epochs=2, device="cpu")

    assert resumed[1]["train_loss"] == pytest.approx(full[1]["train_loss"], rel=1e-5)
