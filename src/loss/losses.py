import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterFusionLoss(nn.Module):
    def __init__(self, hm_weight=1.0, off_weight=1.0, dep_weight=1.0, dim_weight=1.0, vel_weight=1.0):
        super(CenterFusionLoss, self).__init__()
        self.hm_weight = hm_weight
        self.off_weight = off_weight
        self.dep_weight = dep_weight
        self.dim_weight = dim_weight
        self.vel_weight = vel_weight

    def focal_loss(self, pred, target):
        """ 针对 Heatmap 的 Focal Loss """
        pred = torch.sigmoid(pred)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = torch.pow(1 - target, 4)

        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        loss = -(pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-4)
        return loss

    def l1_loss(self, pred, target, mask):
        """ 仅在目标中心点计算 L1 Loss """
        # mask shape: [B, 1, H, W]
        mask = mask.expand_as(target)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, outputs, targets):
        # 1. Heatmap Loss
        hm_loss = self.focal_loss(outputs['hm'], targets['hm'])
        
        # 2. Offset Loss
        off_loss = self.l1_loss(outputs['reg'], targets['reg'], targets['mask'].unsqueeze(1))
        
        # 3. Depth Loss (主视觉深度 + 雷达深度)
        dep_loss = self.l1_loss(outputs['dep'], targets['dep'], targets['mask'].unsqueeze(1))
        
        # 4. Dimension Loss
        dim_loss = self.l1_loss(outputs['dim'], targets['dim'], targets['mask'].unsqueeze(1))
        
        total_loss = (self.hm_weight * hm_loss + 
                      self.off_weight * off_loss + 
                      self.dep_weight * dep_loss + 
                      self.dim_weight * dim_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "hm": hm_loss.item(),
            "dep": dep_loss.item()
        }
