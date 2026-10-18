import numpy as np
import pytest

from src.classical import bilateral_denoise, nlm_denoise
from src.denoise import (
    compose,
    make_denoiser,
    noise_sigma,
    noisiest_window,
    pipeline_name,
)

CONFIG = {"denoise": {"nlm_template_window": 7, "nlm_search_window": 21}}


def _noisy_flat(level: float = 0.5, sigma: float = 0.05, shape=(64, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(level + rng.normal(0.0, sigma, shape), 0.0, 1.0).astype(np.float32)


@pytest.mark.parametrize(
    "denoise",
    [lambda x: nlm_denoise(x, 10), lambda x: bilateral_denoise(x, 0.2, 4)],
    ids=["nlm", "bilateral"],
)
def test_denoisers_reduce_noise_and_keep_format(denoise) -> None:
    noisy = _noisy_flat()
    out = denoise(noisy)
    assert out.shape == noisy.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.std() < 0.5 * noisy.std()
    assert abs(out.mean() - noisy.mean()) < 0.01


def test_nlm_keeps_channel_order() -> None:
    # A pure red image must stay red: catches an RGB/BGR swap around OpenCV.
    red = np.zeros((32, 32, 3), dtype=np.float32)
    red[..., 0] = 0.8
    out = nlm_denoise(red, 5)
    assert out[..., 0].mean() > 0.75
    assert out[..., 2].mean() < 0.05


def test_compose_order() -> None:
    def enhance(x):
        return x * 2

    def denoise(x):
        return x + 1

    x = np.ones((2, 2, 3), dtype=np.float32)
    assert compose(enhance, denoise, "after")(x)[0, 0, 0] == 3  # (1 * 2) + 1
    assert compose(enhance, denoise, "before")(x)[0, 0, 0] == 4  # (1 + 1) * 2
    with pytest.raises(ValueError):
        compose(enhance, denoise, "sideways")


def test_make_denoiser_and_names() -> None:
    noisy = _noisy_flat()
    assert make_denoiser("nlm", {"h": 10}, CONFIG)(noisy).shape == noisy.shape
    params = {"sigma_color": 0.1, "sigma_space": 2}
    assert make_denoiser("bilateral", params, CONFIG)(noisy).shape == noisy.shape
    with pytest.raises(ValueError):
        make_denoiser("median", {}, CONFIG)
    assert pipeline_name("nlm", "after") == "zerodce->nlm"
    assert pipeline_name("bilateral", "before") == "bilateral->zerodce"


def test_noise_sigma_tracks_added_noise() -> None:
    clean = np.full((128, 128, 3), 0.5, dtype=np.float32)
    assert noise_sigma(clean) < 0.5
    estimate = noise_sigma(_noisy_flat(sigma=10 / 255, shape=(128, 128, 3)))
    assert 8.0 < estimate < 12.0


def test_noisiest_window_finds_the_noisy_patch() -> None:
    reference = np.full((80, 120, 3), 0.5, dtype=np.float32)
    output = reference.copy()
    output[40:56, 70:86] = _noisy_flat(sigma=0.2, shape=(16, 16, 3))
    y, x = noisiest_window(output, reference, 16)
    assert abs(y - 40) <= 2 and abs(x - 70) <= 2
    assert 0 <= y <= 80 - 16 and 0 <= x <= 120 - 16
