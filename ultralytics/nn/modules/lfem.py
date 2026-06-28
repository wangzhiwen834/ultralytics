# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight perception modules for small vulnerable pedestrian detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, DWConv

__all__ = ("AFF", "LFEM")


class LFEM(nn.Module):
    """Lightweight Feature Enhancement Module.

    The module enhances local edge and texture responses with depthwise convolution and a lightweight sigmoid attention
    map while preserving the input tensor shape.
    """

    def __init__(self, c: int) -> None:
        """Initialize LFEM.

        Args:
            c (int): Number of input and output channels.
        """
        super().__init__()
        self.local = nn.Sequential(DWConv(c, c, 3, 1), Conv(c, c, 1, 1, act=False))
        self.attn = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True), nn.Sigmoid())
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight feature enhancement to the input feature map."""
        y = self.local(x)
        return x + self.act(y * self.attn(y))


class AFF(nn.Module):
    """Adaptive Feature Fusion module for two aligned feature maps.

    AFF replaces a plain concatenation with learnable branch weights followed by a pointwise projection. The two input
    feature maps must have the same spatial size before entering this module.
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
        hidden = max(c // 4, 8)
        self.weight = nn.Sequential(Conv(c * 2, hidden, 1, 1), nn.Conv2d(hidden, 2, 1, 1, bias=True), nn.Sigmoid())
        self.out = Conv(c, c, 1, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse two feature maps with learned normalized sigmoid weights."""
        x1, x2 = x
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        weight = self.weight(torch.cat((x1, x2), 1))
        weight = weight / (weight.sum(1, keepdim=True) + 1e-6)
        return self.out(x1 * weight[:, 0:1] + x2 * weight[:, 1:2])
