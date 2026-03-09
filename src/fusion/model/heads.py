from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bias: float | None = None
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        if bias is not None:
            nn.init.constant_(self.layers[-1].bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DetectionHeads(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.heatmap = Head(in_channels, num_classes, bias=-2.19)
        self.offset = Head(in_channels, 2)
        self.depth = Head(in_channels, 1)
        self.size2d = Head(in_channels, 2)
        self.dim3d = Head(in_channels, 3)
        self.rotation = Head(in_channels, 2)
        self.velocity = Head(in_channels, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "heatmap": self.heatmap(x),
            "offset": self.offset(x),
            "depth": self.depth(x),
            "size2d": self.size2d(x),
            "dim3d": self.dim3d(x),
            "rotation": self.rotation(x),
            "velocity": self.velocity(x),
        }
