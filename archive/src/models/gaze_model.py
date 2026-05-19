"""
gaze_model.py — GazeEstimator (Phase 1: Fixed Filter)

구조:
  SiameseBackbone → concat([feat_l, feat_r, head_pose]) → Regressor → 3D 단위벡터

출력: (B,3) F.normalize 적용 단위벡터
손실: 1 - cosine_similarity (train.py 참조)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import SiameseBackbone, get_feat_dim


class Regressor(nn.Module):
    """FC → ReLU → Dropout × N → FC(→ out_dim)"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 1,
        dropout: float = 0.3,
        out_dim: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GazeEstimator(nn.Module):
    """
    Phase 1 모델.

    Args:
        cfg: 병합된 설정 dict
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model", {})

        backbone_name = model_cfg.get("backbone", "resnet18")
        pretrained    = model_cfg.get("pretrained", True)
        freeze        = model_cfg.get("freeze_backbone", False)
        dropout       = float(model_cfg.get("dropout", 0.3))
        hidden_dim    = int(model_cfg.get("regressor_hidden", 256))
        n_layers      = int(model_cfg.get("regressor_layers", 1))

        self.backbone = SiameseBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze,
        )
        feat_dim = get_feat_dim(backbone_name)
        in_dim   = feat_dim * 2 + 3  # feat_l + feat_r + head_pose

        self.regressor = Regressor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            out_dim=3,
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            left      : (B,3,H,W) float32
            right     : (B,3,H,W) float32
            head_pose : (B,3) float32
        Returns:
            (B,3) float32 — 정규화된 시선 단위벡터
        """
        feat_l, feat_r = self.backbone(left, right)
        x = torch.cat([feat_l, feat_r, head_pose], dim=1)
        out = self.regressor(x)
        return F.normalize(out, dim=1)
