from __future__ import annotations

import torch
from torch import nn

from fusion.config import FusionConfig
from fusion.model.backbone import FusionNeck, ImageEncoder, RadarEncoder
from fusion.model.heads import DetectionHeads


class CenterFusionDetector(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        width = config.model.width
        self.image_encoder = ImageEncoder(config.model.image_channels, width)
        self.radar_encoder = RadarEncoder(config.model.radar_channels, width)
        self.fusion_neck = FusionNeck(width * 2, width, config.model.head_channels)
        self.heads = DetectionHeads(
            config.model.head_channels, len(config.dataset.classes)
        )

    def forward(
        self, image: torch.Tensor, radar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        image_feats = self.image_encoder(image)
        radar_feats = self.radar_encoder(radar)
        fused = self.fusion_neck(image_feats, radar_feats)
        return self.heads(fused)
