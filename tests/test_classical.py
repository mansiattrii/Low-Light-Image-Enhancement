import numpy as np
import pytest

from src.classical import build_classical_methods, clahe, gamma_correction, histogram_equalization
from src.metrics import psnr_ssim


@pytest.fixture
def dark_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((37, 53, 3)) * 0.2).astype(np.float32)


@pytest.mark.parametrize(
    "enhance",
    [
        lambda x: gamma_correction(x, 0.4),
        histogram_equalization,
        lambda x: clahe(x, clip_limit=2.0, tile_size=8),
    ],
    ids=["gamma", "hist_eq", "clahe"],
)
def test_output_shape_range_dtype(enhance, dark_image: np.ndarray) -> None:
    out = enhance(dark_image)
    assert out.shape == dark_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize(
    "enhance",
    [lambda x: gamma_correction(x, 0.4), histogram_equalization],
    ids=["gamma", "hist_eq"],
)
def test_brightens_dark_image(enhance, dark_image: np.ndarray) -> None:
    assert enhance(dark_image).mean() > dark_image.mean()


def test_gamma_one_is_identity(dark_image: np.ndarray) -> None:
    np.testing.assert_allclose(gamma_correction(dark_image, 1.0), dark_image)


def test_gamma_preserves_black_and_white() -> None:
    image = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(gamma_correction(image, 0.3), image)


def test_gamma_rejects_non_positive(dark_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        gamma_correction(dark_image, 0.0)


def test_hist_eq_keeps_gray_pixels_gray(dark_image: np.ndarray) -> None:
    gray = np.repeat(dark_image[..., :1], 3, axis=2)
    out = histogram_equalization(gray)
    assert np.abs(out[..., 0] - out[..., 1]).max() <= 1 / 255 + 1e-6
    assert np.abs(out[..., 1] - out[..., 2]).max() <= 1 / 255 + 1e-6


def test_build_classical_methods(dark_image: np.ndarray) -> None:
    params = {"gamma": {"gamma": 0.5}, "clahe": {"clip_limit": 2.0, "tile_size": 4}}
    methods = build_classical_methods(params)
    assert set(methods) == {"gamma", "hist_eq", "clahe"}
    for enhance in methods.values():
        assert enhance(dark_image).shape == dark_image.shape


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_metrics_identical_images(dark_image: np.ndarray) -> None:
    psnr, ssim = psnr_ssim(dark_image, dark_image)
    assert psnr == np.inf
    assert ssim == pytest.approx(1.0)


def test_metrics_drop_with_noise(dark_image: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    noisy = np.clip(dark_image + rng.normal(0, 0.05, dark_image.shape), 0, 1)
    psnr, ssim = psnr_ssim(noisy, dark_image)
    assert 10 < psnr < 40
    assert ssim < 0.95
