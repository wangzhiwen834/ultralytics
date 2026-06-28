# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight perception modules for small vulnerable pedestrian detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .conv import Conv, DWConv

__all__ = ("AFF", "LFEM")


def _save_heatmap(x: torch.Tensor, file: Path) -> None:
    """Save a single-channel tensor as a normalized grayscale heatmap."""
    file.parent.mkdir(parents=True, exist_ok=True)
    x = x[0, 0].detach().float().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    Image.fromarray((x.numpy() * 255).astype(np.uint8)).save(file)


class LFEM(nn.Module):
    """Lightweight Feature Enhancement Module with Global Context Guidance.

    The module first extracts local edge cues with a depthwise convolution, then derives channel weights from global
    average pooled context instead of directly generating attention from the spatial feature map. This design follows
    Equation (5)-(6) in the LSVD method description and keeps the input and output shapes identical.
    """

    def __init__(self, c: int) -> None:
        """Initialize LFEM.

        Args:
            c (int): Number of input and output channels.
        """
        super().__init__()
        self.dw = DWConv(c, c, 3, 1)
        self.pw = Conv(c, c, 1, 1, act=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context = nn.Conv2d(c, c, 1, 1, groups=c, bias=True)
        self.gate = nn.Sigmoid()
        self.channel_weight = None
        self.attention_map = None
        self.gradcam_map = None
        self._activation = None
        self._gradient = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LFEM enhancement.

        Equation (5): ``w = sigmoid(Conv1x1(GAP(PWConv(SiLU(BN(DWConv(x)))))))``.
        Equation (6): ``y = x + f(x) * w``.
        """
        y = self.pw(self.dw(x))
        weight = self.gate(self.context(self.pool(y)))
        enhanced = y * weight
        self.channel_weight = weight.detach()
        self.attention_map = enhanced.mean(1, keepdim=True).detach()
        if enhanced.requires_grad:
            self._activation = enhanced
            enhanced.register_hook(self._save_gradient)
        return x + enhanced

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Cache gradients for LFEM Grad-CAM visualization."""
        self._gradient = grad.detach()

    def __getstate__(self) -> dict:
        """Drop transient visualization tensors during deepcopy/export."""
        state = self.__dict__.copy()
        for key in ("channel_weight", "attention_map", "gradcam_map", "_activation", "_gradient"):
            state[key] = None
        return state

    def save_attention_map(self, save_dir: str | Path, stage: int) -> None:
        """Save the LFEM attention response map used for paper visualization."""
        if self.attention_map is not None:
            _save_heatmap(self.attention_map, Path(save_dir) / f"stage{stage}_LFEM_attention.png")

    def save_gradcam(self, save_dir: str | Path, stage: int) -> None:
        """Save LFEM Grad-CAM after a backward pass has populated gradients."""
        if self._activation is None or self._gradient is None:
            return
        weight = self._gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weight * self._activation).sum(1, keepdim=True).relu()
        self.gradcam_map = cam.detach()
        _save_heatmap(cam, Path(save_dir) / f"stage{stage}_LFEM_gradcam.png")


class AFF(nn.Module):
    """Adaptive Feature Fusion module with branch calibration and Softmax weighting.

    Each branch is projected by a 1x1 Conv-BN block, calibrated by a global context channel gate, and assigned a
    Softmax-normalized scalar fusion weight. The two input feature maps must have the same spatial size before entering
    this module.
    """

    def __init__(self, channels: list[int], c: int) -> None:
        """Initialize AFF.

        Args:
            channels (list[int]): Input channels for the two feature maps.
            c (int): Output channels after adaptive fusion.
        """
        super().__init__()
        if len(channels) != 2:
            raise ValueError("AFF expects exactly two input feature maps.")
        self.cv1 = Conv(channels[0], c, 1, 1)
        self.cv2 = Conv(channels[1], c, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.calibrate1 = nn.Sequential(nn.Conv2d(c, c, 1, 1, groups=c, bias=True), nn.Sigmoid())
        self.calibrate2 = nn.Sequential(nn.Conv2d(c, c, 1, 1, groups=c, bias=True), nn.Sigmoid())
        self.score1 = nn.Conv2d(c, 1, 1, 1, bias=True)
        self.score2 = nn.Conv2d(c, 1, 1, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.out = Conv(c, c, 1, 1)
        self.fusion_weight = None

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse two feature maps with Softmax-normalized adaptive weights.

        Equation (7): ``f_i = p_i(x_i) * sigmoid(g_i(GAP(p_i(x_i))))``.
        Equation (8): ``[a_1, a_2] = softmax([s_1(GAP(f_1)), s_2(GAP(f_2))])``.
        Equation (9): ``y = Conv1x1(a_1 * f_1 + a_2 * f_2)``.
        """
        x1, x2 = x
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        x1 = x1 * self.calibrate1(self.pool(x1))
        x2 = x2 * self.calibrate2(self.pool(x2))
        weight = self.softmax(torch.cat((self.score1(self.pool(x1)), self.score2(self.pool(x2))), 1))
        self.fusion_weight = weight.detach()
        return self.out(x1 * weight[:, 0:1] + x2 * weight[:, 1:2])
