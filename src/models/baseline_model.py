"""
baseline_model.py — BaselineModel (대조군)

SiameseBackbone → LargerRegressor → F.normalize
(KernelNet 없음 — 동일 파라미터 예산을 Regressor에 투자)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import SiameseBackbone, get_feat_dim


class Regressor(nn.Module):
    """FC → ReLU → Dropout × n_layers → FC(out_dim)"""

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 n_layers: int = 1, dropout: float = 0.3, out_dim: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineModel(nn.Module):
    """
    Args:
        cfg: 병합된 설정 dict
          cfg["model"]["backbone"]         : resnet18 (고정)
          cfg["model"]["pretrained"]       : True
          cfg["model"]["freeze_backbone"]  : False
          cfg["model"]["regressor_hidden"] : 512 / 1024 / 2048
          cfg["model"]["dropout"]          : 0.3
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {})

        backbone_name = m.get("backbone", "resnet18")
        pretrained    = bool(m.get("pretrained", True))
        freeze        = bool(m.get("freeze_backbone", False))
        dropout       = float(m.get("dropout", 0.3))
        reg_hidden    = int(m.get("regressor_hidden", 1024))

        self.backbone  = SiameseBackbone(backbone_name, pretrained, freeze)

        feat_dim = get_feat_dim(backbone_name)
        in_dim   = feat_dim * 2 + 3   # feat_l + feat_r + head_pose

        self.regressor = Regressor(in_dim, hidden_dim=reg_hidden, dropout=dropout)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                head_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left      : (B, 3, 64, 64)
            right     : (B, 3, 64, 64)
            head_pose : (B, 3)
        Returns:
            (B, 3) 정규화된 시선 단위벡터
        """
        feat_l, feat_r = self.backbone(left, right)            # (B,512)×2
        x   = torch.cat([feat_l, feat_r, head_pose], dim=1)   # (B,1027)
        out = self.regressor(x)                                 # (B,3)
        return F.normalize(out, dim=1)