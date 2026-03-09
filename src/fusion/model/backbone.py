from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, width, stride=2),
            ConvBNAct(width, width, stride=1),
            ConvBNAct(width, width * 2, stride=2),
            ResidualBlock(width * 2),
            ResidualBlock(width * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class RadarEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, width // 2, stride=2),
            ConvBNAct(width // 2, width, stride=1),
            ConvBNAct(width, width, stride=2),
            ResidualBlock(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class FusionNeck(nn.Module):
    def __init__(
        self, image_channels: int, radar_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(image_channels, out_channels, kernel_size=1)
        self.radar_proj = nn.Conv2d(radar_channels, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(
                out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(out_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(
        self, image_feats: torch.Tensor, radar_feats: torch.Tensor
    ) -> torch.Tensor:
        image_feats = self.image_proj(image_feats)
        radar_feats = self.radar_proj(radar_feats)
        weight = self.gate(torch.cat([image_feats, radar_feats], dim=1))
        fused = image_feats + weight * radar_feats
        return self.refine(fused)
