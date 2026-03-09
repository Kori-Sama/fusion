from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fusion.config import FusionConfig


def _transpose_and_gather(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feat.shape
    feat = feat.view(batch, channels, -1).permute(0, 2, 1).contiguous()
    expanded = indices.unsqueeze(-1).expand(-1, -1, channels)
    return feat.gather(1, expanded)


def heatmap_focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos


class CenterFusionLoss(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.weights = config.loss

    def _regression_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = _transpose_and_gather(prediction, indices)
        mask = mask.unsqueeze(-1).expand_as(prediction)
        loss = F.l1_loss(prediction * mask, target * mask, reduction="sum")
        normalizer = mask.sum().clamp_min(1.0)
        return loss / normalizer

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        indices = batch["indices"].long()
        mask = batch["mask"].float()
        losses = {
            "heatmap": heatmap_focal_loss(outputs["heatmap"], batch["heatmap"]),
            "offset": self._regression_loss(
                outputs["offset"], batch["offset"], indices, mask
            ),
            "depth": self._regression_loss(
                outputs["depth"], batch["depth"], indices, mask
            ),
            "size2d": self._regression_loss(
                outputs["size2d"], batch["size2d"], indices, mask
            ),
            "dim3d": self._regression_loss(
                outputs["dim3d"], batch["dim3d"], indices, mask
            ),
            "rotation": self._regression_loss(
                outputs["rotation"], batch["rotation"], indices, mask
            ),
            "velocity": self._regression_loss(
                outputs["velocity"], batch["velocity"], indices, mask
            ),
        }
        total = (
            self.weights.heatmap * losses["heatmap"]
            + self.weights.offset * losses["offset"]
            + self.weights.depth * losses["depth"]
            + self.weights.size2d * losses["size2d"]
            + self.weights.dim3d * losses["dim3d"]
            + self.weights.rotation * losses["rotation"]
            + self.weights.velocity * losses["velocity"]
        )
        scalars = {key: float(value.detach().item()) for key, value in losses.items()}
        scalars["total"] = float(total.detach().item())
        return total, scalars
