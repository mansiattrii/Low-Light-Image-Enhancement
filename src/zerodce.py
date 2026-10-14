"""Zero-DCE: DCE-Net and the iterative light-enhancement curve (Guo et al., CVPR 2020)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def le_curve(x: Tensor, alpha: Tensor) -> Tensor:
    """One light-enhancement step, ``LE(x) = x + alpha * x * (1 - x)`` (paper Eq. 1).

    For x in [0, 1] and alpha in [-1, 1] the output stays in [0, 1], is monotonic in x,
    and keeps 0 and 1 fixed.
    """
    return x + alpha * x * (1.0 - x)


def apply_curves(x: Tensor, alphas: Tensor) -> Tensor:
    """Apply the curve iteratively, one 3-channel alpha map per iteration (paper Eq. 2-3).

    ``alphas`` has shape (B, 3 * n, H, W); iteration k uses channels [3k, 3k + 3).
    """
    for alpha in torch.split(alphas, 3, dim=1):
        x = le_curve(x, alpha)
    return x


class DCENet(nn.Module):
    """Seven 3x3 conv layers with symmetric concatenation; predicts per-pixel curve maps.

    No pooling or batch normalization, so it runs on any input size and every alpha
    value depends only on a local neighbourhood of the input.
    """

    def __init__(self, features: int = 32, iterations: int = 8) -> None:
        super().__init__()
        self.iterations = iterations

        def conv(in_ch: int, out_ch: int) -> nn.Conv2d:
            return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv1 = conv(3, features)
        self.conv2 = conv(features, features)
        self.conv3 = conv(features, features)
        self.conv4 = conv(features, features)
        self.conv5 = conv(2 * features, features)
        self.conv6 = conv(2 * features, features)
        self.conv7 = conv(2 * features, 3 * iterations)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the enhanced image and the alpha maps, shaped (B, 3, H, W) and (B, 3n, H, W)."""
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = F.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        alphas = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        return apply_curves(x, alphas), alphas


def init_weights(model: nn.Module) -> None:
    """Initialize conv weights from N(0, 0.02) and biases to zero, as in the paper."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(
    path: str | Path, device: str | torch.device = "cpu", features: int = 32, iterations: int = 8
) -> DCENet:
    """Load DCE-Net weights from a plain state dict or a training checkpoint."""
    state = torch.load(path, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model = DCENet(features, iterations)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def enhance_array(
    model: DCENet, image: np.ndarray, device: str | torch.device = "cpu"
) -> np.ndarray:
    """Enhance one float32 RGB image in [0, 1] with shape (H, W, 3); returns the same layout."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    enhanced, _ = model(x)
    return enhanced.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
