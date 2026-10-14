import math

import pytest
import torch

from src.losses import (
    LossWeights,
    ZeroDCELoss,
    color_constancy_loss,
    exposure_control_loss,
    illumination_smoothness_loss,
    spatial_consistency_loss,
)


@pytest.fixture
def image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 3, 64, 64) * 0.2


def _is_finite_scalar(t: torch.Tensor) -> bool:
    return t.dim() == 0 and math.isfinite(t.item())


def test_spatial_zero_for_unchanged_image(image: torch.Tensor) -> None:
    assert spatial_consistency_loss(image, image).item() == pytest.approx(0.0)


def test_spatial_zero_for_constant_shift(image: torch.Tensor) -> None:
    # Adding a constant keeps every neighbour difference, so structure is preserved.
    assert spatial_consistency_loss(image + 0.3, image).item() == pytest.approx(0.0, abs=1e-10)


def test_spatial_penalizes_lost_structure(image: torch.Tensor) -> None:
    flat = torch.full_like(image, image.mean().item())
    loss = spatial_consistency_loss(flat, image)
    assert _is_finite_scalar(loss) and loss.item() > 0


FLOAT32_TOL = 1e-5


def test_exposure_zero_at_target_level() -> None:
    loss = exposure_control_loss(torch.full((1, 3, 32, 32), 0.6))
    assert loss.item() == pytest.approx(0.0, abs=FLOAT32_TOL)


def test_exposure_matches_distance_from_target(image: torch.Tensor) -> None:
    dark = exposure_control_loss(torch.full((1, 3, 32, 32), 0.1))
    bright = exposure_control_loss(torch.full((1, 3, 32, 32), 0.9))
    assert dark.item() == pytest.approx(0.5, abs=FLOAT32_TOL)
    assert bright.item() == pytest.approx(0.3, abs=FLOAT32_TOL)
    assert _is_finite_scalar(exposure_control_loss(image))


def test_color_zero_for_gray_image() -> None:
    gray = torch.rand(2, 1, 16, 16).expand(2, 3, 16, 16)
    assert color_constancy_loss(gray).item() == pytest.approx(0.0, abs=1e-12)


def test_color_penalizes_cast() -> None:
    red = torch.zeros(1, 3, 16, 16)
    red[:, 0] = 1.0
    # Channel means (1, 0, 0): (1-0)^2 + (1-0)^2 + (0-0)^2 = 2.
    assert color_constancy_loss(red).item() == pytest.approx(2.0)


def test_illumination_zero_for_constant_maps() -> None:
    assert illumination_smoothness_loss(torch.full((2, 24, 16, 16), 0.4), 8).item() == 0.0


def test_illumination_penalizes_noisy_maps() -> None:
    torch.manual_seed(0)
    smooth = torch.linspace(-1, 1, 16).view(1, 1, 1, 16).expand(1, 24, 16, 16)
    noisy = torch.rand(1, 24, 16, 16) * 2 - 1
    assert illumination_smoothness_loss(noisy, 8) > illumination_smoothness_loss(smooth, 8) > 0


def test_combined_loss_and_gradients(image: torch.Tensor) -> None:
    alphas = (torch.rand(2, 24, 64, 64) * 2 - 1).requires_grad_()
    enhanced = (image + 0.1 * alphas[:, :3]).clamp(0, 1)
    total, terms = ZeroDCELoss()(image, enhanced, alphas)
    assert _is_finite_scalar(total)
    assert set(terms) == {"spatial", "exposure", "color", "illumination"}
    total.backward()
    assert torch.isfinite(alphas.grad).all()


def test_zero_weight_removes_term(image: torch.Tensor) -> None:
    alphas = torch.rand(2, 24, 64, 64) * 2 - 1
    weights = LossWeights(spatial=0.0, exposure=1.0, color=0.0, illumination=0.0)
    total, terms = ZeroDCELoss(weights)(image, image, alphas)
    assert total.item() == pytest.approx(terms["exposure"])
