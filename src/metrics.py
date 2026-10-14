"""Full-reference image quality metrics shared by tuning and evaluation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.data import to_uint8


def psnr_ssim(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return (PSNR in dB, SSIM) between two RGB images in [0, 1].

    The prediction is quantized to 8 bits first so every method is scored on the
    image it would actually save, exactly like the 8-bit ground truth.
    SSIM uses the settings of Wang et al. (2004): an 11x11 Gaussian window with
    sigma 1.5 and population covariance, averaged over the three RGB channels.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    pred = to_uint8(prediction).astype(np.float64) / 255.0
    gt = to_uint8(target).astype(np.float64) / 255.0
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(
        gt,
        pred,
        data_range=1.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return float(psnr), float(ssim)
