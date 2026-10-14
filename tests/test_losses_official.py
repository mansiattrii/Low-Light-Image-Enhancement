"""Check the 'official' loss formulation against a transcription of the official Myloss.py.

The reference functions below follow github.com/Li-Chongyi/Zero-DCE (Zero-DCE_code/Myloss.py)
with the .cuda() calls removed and the unused variables dropped.
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses import (
    LossWeights,
    ZeroDCELoss,
    color_constancy_loss_official,
    exposure_control_loss_official,
    illumination_smoothness_loss_official,
    spatial_consistency_loss_official,
)


def _ref_color(x: torch.Tensor) -> torch.Tensor:
    mean_rgb = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    drg, drb, dgb = (mr - mg) ** 2, (mr - mb) ** 2, (mb - mg) ** 2
    return torch.pow(drg**2 + drb**2 + dgb**2, 0.5)


def _ref_spa(org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
    def kernel(values: list[list[float]]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    kernels = [
        kernel([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]),
        kernel([[0, 0, 0], [0, 1, -1], [0, 0, 0]]),
        kernel([[0, -1, 0], [0, 1, 0], [0, 0, 0]]),
        kernel([[0, 0, 0], [0, 1, 0], [0, -1, 0]]),
    ]
    org_pool = F.avg_pool2d(torch.mean(org, 1, keepdim=True), 4)
    enh_pool = F.avg_pool2d(torch.mean(enhance, 1, keepdim=True), 4)
    return sum(
        (F.conv2d(org_pool, k, padding=1) - F.conv2d(enh_pool, k, padding=1)) ** 2 for k in kernels
    )


def _ref_exp(x: torch.Tensor, patch: int = 16, mean_val: float = 0.6) -> torch.Tensor:
    mean = F.avg_pool2d(torch.mean(x, 1, keepdim=True), patch)
    return torch.mean((mean - mean_val) ** 2)


def _ref_tv(x: torch.Tensor) -> torch.Tensor:
    batch, h, w = x.size(0), x.size(2), x.size(3)
    count_h, count_w = (h - 1) * w, h * (w - 1)
    h_tv = ((x[:, :, 1:, :] - x[:, :, : h - 1, :]) ** 2).sum()
    w_tv = ((x[:, :, :, 1:] - x[:, :, :, : w - 1]) ** 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch


@pytest.fixture
def tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    low = torch.rand(3, 3, 64, 48) * 0.2
    enhanced = torch.rand(3, 3, 64, 48)
    alphas = torch.rand(3, 24, 64, 48) * 2 - 1
    return low, enhanced, alphas


def test_matches_official_code(tensors: tuple[torch.Tensor, ...]) -> None:
    low, enhanced, alphas = tensors
    assert torch.allclose(color_constancy_loss_official(enhanced), _ref_color(enhanced).mean())
    assert torch.allclose(
        spatial_consistency_loss_official(enhanced, low), _ref_spa(enhanced, low).mean()
    )
    assert torch.allclose(exposure_control_loss_official(enhanced), _ref_exp(enhanced))
    assert torch.allclose(illumination_smoothness_loss_official(alphas), _ref_tv(alphas))


def test_official_total_matches_training_script(tensors: tuple[torch.Tensor, ...]) -> None:
    low, enhanced, alphas = tensors
    weights = LossWeights(spatial=1.0, exposure=10.0, color=5.0, illumination=200.0)
    total, _ = ZeroDCELoss(weights, formulation="official")(low, enhanced, alphas)
    expected = (
        200 * _ref_tv(alphas)
        + _ref_spa(enhanced, low).mean()
        + 5 * _ref_color(enhanced).mean()
        + 10 * _ref_exp(enhanced)
    )
    assert torch.allclose(total, expected)


def test_unknown_formulation() -> None:
    with pytest.raises(ValueError):
        ZeroDCELoss(formulation="other")
