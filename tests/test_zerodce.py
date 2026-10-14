import pytest
import torch

from src.zerodce import (
    DCENet,
    apply_curves,
    count_parameters,
    enhance_array,
    init_weights,
    le_curve,
)

PAPER_PARAMETER_COUNT = 79_416


@pytest.fixture
def model() -> DCENet:
    torch.manual_seed(0)
    net = DCENet()
    init_weights(net)
    return net.eval()


def test_curve_stays_in_unit_range() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    alphas = torch.rand(2, 24, 32, 32) * 2 - 1
    y = apply_curves(x, alphas)
    assert y.min() >= 0.0 and y.max() <= 1.0


@pytest.mark.parametrize("alpha", [-1.0, -0.3, 0.0, 0.5, 1.0])
def test_curve_extreme_alphas_stay_in_range(alpha: float) -> None:
    x = torch.linspace(0, 1, 101).view(1, 1, 1, -1).expand(1, 3, 1, 101)
    y = apply_curves(x, torch.full((1, 24, 1, 101), alpha))
    assert y.min() >= 0.0 and y.max() <= 1.0 + 1e-6


def test_curve_fixes_black_and_white() -> None:
    alphas = torch.rand(1, 24, 8, 8) * 2 - 1
    zeros, ones = torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)
    assert torch.equal(apply_curves(zeros, alphas), zeros)
    assert torch.allclose(apply_curves(ones, alphas), ones)


def test_curve_is_monotonic_and_brightens_for_positive_alpha() -> None:
    x = torch.linspace(0, 1, 101)
    y = le_curve(x, torch.tensor(0.8))
    assert torch.all(y[1:] >= y[:-1])
    assert torch.all(y >= x)


def test_zero_alpha_is_identity() -> None:
    x = torch.rand(1, 3, 5, 5)
    assert torch.equal(apply_curves(x, torch.zeros(1, 24, 5, 5)), x)


def test_parameter_count_matches_paper(model: DCENet) -> None:
    assert count_parameters(model) == PAPER_PARAMETER_COUNT


@pytest.mark.parametrize("size", [(64, 64), (37, 53), (1, 1)])
def test_output_shapes(model: DCENet, size: tuple[int, int]) -> None:
    x = torch.rand(2, 3, *size)
    with torch.no_grad():
        enhanced, alphas = model(x)
    assert enhanced.shape == x.shape
    assert alphas.shape == (2, 24, *size)
    assert alphas.abs().max() <= 1.0


def test_enhance_array_roundtrip(model: DCENet) -> None:
    image = torch.rand(21, 34, 3).numpy()
    out = enhance_array(model, image)
    assert out.shape == image.shape and out.dtype == image.dtype
    assert out.min() >= 0.0 and out.max() <= 1.0
