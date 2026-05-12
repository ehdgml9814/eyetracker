"""
backbone.py — Siamese 백본 (resnet18 / mobilenet_v2)

feat_dim:
  resnet18    → 512
  mobilenet_v2 → 1280
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


_FEAT_DIM = {
    "resnet18":     512,
    "mobilenet_v2": 1280,
}


def get_feat_dim(backbone_name: str) -> int:
    if backbone_name not in _FEAT_DIM:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(_FEAT_DIM)}")
    return _FEAT_DIM[backbone_name]


class SiameseBackbone(nn.Module):
    """
    좌/우 눈 이미지를 공유 가중치 CNN에 통과시켜 특징 추출.

    Args:
        backbone_name : 'resnet18' | 'mobilenet_v2'
        pretrained    : ImageNet 사전학습 가중치 사용 여부
        freeze        : True면 백본 파라미터 고정 (regressor만 학습)
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        self.feat_dim = get_feat_dim(backbone_name)

        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone_name == "resnet18":
            base = tvm.resnet18(weights=weights)
            # 분류 헤드 제거
            self.encoder = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)

        elif backbone_name == "mobilenet_v2":
            base = tvm.mobilenet_v2(weights=weights)
            # features + adaptive_avg_pool
            self.encoder = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )  # (B,1280,1,1)

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B, feat_dim)"""
        feat = self.encoder(x)
        return feat.flatten(1)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            left  : (B,3,H,W)
            right : (B,3,H,W)
        Returns:
            feat_l, feat_r : each (B, feat_dim)
        """
        return self.forward_one(left), self.forward_one(right)
