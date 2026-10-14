"""The four zero-reference losses of Zero-DCE (paper Eq. 4-8).

None of them looks at a normal-light reference: they only compare the enhanced image
with the input, with fixed targets, or with itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn


def _region_means(image: Tensor, size: int) -> Tensor:
    """Mean intensity (over RGB) of non-overlapping ``size x size`` regions, shape (B, 1, h, w)."""
    return F.avg_pool2d(image.mean(dim=1, keepdim=True), size)


def spatial_consistency_loss(enhanced: Tensor, original: Tensor, region_size: int = 4) -> Tensor:
    """Preserve contrast between neighbouring regions (Eq. 4).

    For every region i and each of its four neighbours j, compares |Y_i - Y_j| in the
    enhanced image with |I_i - I_j| in the input, averaged over the K regions.
    """
    y = _region_means(enhanced, region_size)
    i = _region_means(original, region_size)

    def neighbour_diffs(t: Tensor) -> tuple[Tensor, Tensor]:
        horizontal = (t[..., :, 1:] - t[..., :, :-1]).abs()
        vertical = (t[..., 1:, :] - t[..., :-1, :]).abs()
        return horizontal, vertical

    y_h, y_v = neighbour_diffs(y)
    i_h, i_v = neighbour_diffs(i)
    # Each adjacent pair (i, j) appears in the neighbourhood of both i and j, hence the 2.
    per_image = 2.0 * (
        ((y_h - i_h) ** 2).sum(dim=(1, 2, 3)) + ((y_v - i_v) ** 2).sum(dim=(1, 2, 3))
    )
    num_regions = y.shape[-2] * y.shape[-1]
    return (per_image / num_regions).mean()


def exposure_control_loss(enhanced: Tensor, level: float = 0.6, patch_size: int = 16) -> Tensor:
    """Pull the mean intensity of each ``patch_size`` patch towards ``level`` (Eq. 5)."""
    return (_region_means(enhanced, patch_size) - level).abs().mean()


def color_constancy_loss(enhanced: Tensor) -> Tensor:
    """Gray-World prior: the image-wide R, G and B means should be equal (Eq. 6)."""
    r, g, b = enhanced.mean(dim=(2, 3)).unbind(dim=1)
    return ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()


def illumination_smoothness_loss(alphas: Tensor, iterations: int) -> Tensor:
    """Total-variation penalty on the curve-parameter maps (Eq. 7).

    Computes (|dA/dx| + |dA/dy|)^2 per pixel, averages over pixels, sums over the
    3 colour channels and all iterations, and divides by the number of iterations.
    """
    dx = (alphas[..., :-1, 1:] - alphas[..., :-1, :-1]).abs()
    dy = (alphas[..., 1:, :-1] - alphas[..., :-1, :-1]).abs()
    per_map = ((dx + dy) ** 2).mean(dim=(2, 3))
    return (per_map.sum(dim=1) / iterations).mean()


@dataclass(frozen=True)
class LossWeights:
    spatial: float = 1.0
    exposure: float = 1.0
    color: float = 0.5
    illumination: float = 20.0


class ZeroDCELoss(nn.Module):
    """Weighted sum of the four zero-reference losses (Eq. 8)."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        exposure_level: float = 0.6,
        exposure_patch: int = 16,
        spatial_region: int = 4,
        iterations: int = 8,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.exposure_level = exposure_level
        self.exposure_patch = exposure_patch
        self.spatial_region = spatial_region
        self.iterations = iterations

    def forward(
        self, original: Tensor, enhanced: Tensor, alphas: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Return the total loss and the unweighted value of each term for logging."""
        terms = {
            "spatial": spatial_consistency_loss(enhanced, original, self.spatial_region),
            "exposure": exposure_control_loss(enhanced, self.exposure_level, self.exposure_patch),
            "color": color_constancy_loss(enhanced),
            "illumination": illumination_smoothness_loss(alphas, self.iterations),
        }
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        return total, {name: value.item() for name, value in terms.items()}
