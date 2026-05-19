"""
adaptive_filter.py — Phase 2 Adaptive Filter

FilterParamNet  : head_pose (B,3) → 필터 파라미터 (B,5)
AdaptiveFilter  : 미분 가능 필터 (Gaussian high-pass + Gabor)
GazeEstimatorV2 : Phase 2 전체 모델 — 출력 (B,3) 단위벡터
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import SiameseBackbone, get_feat_dim
from src.models.gaze_model import Regressor


# ── FilterParamNet ────────────────────────────────────────────────────────────

class FilterParamNet(nn.Module):
    """
    head_pose (B,3) → [hp_sigma, gabor_freq, gabor_theta, blend_hp, blend_gab]

    출력 범위:
      hp_sigma    : [0.5, 4.0]
      gabor_freq  : [0.1, 0.9]
      gabor_theta : [0, π]
      blend_hp    : [0, 1]
      blend_gab   : [0, 1]
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 5),
        )

    def forward(self, head_pose: torch.Tensor) -> torch.Tensor:
        raw = self.net(head_pose)
        hp_sigma    = torch.sigmoid(raw[:, 0:1]) * 3.5 + 0.5
        gabor_freq  = torch.sigmoid(raw[:, 1:2]) * 0.8 + 0.1
        gabor_theta = torch.sigmoid(raw[:, 2:3]) * math.pi
        blend_hp    = torch.sigmoid(raw[:, 3:4])
        blend_gab   = torch.sigmoid(raw[:, 4:5])
        return torch.cat([hp_sigma, gabor_freq, gabor_theta, blend_hp, blend_gab], dim=1)


# ── 커널 생성 헬퍼 ────────────────────────────────────────────────────────────

def _gaussian_kernel_2d(sigma: torch.Tensor, ksize: int = 11) -> torch.Tensor:
    """sigma: (B,) → (B,1,ksize,ksize)"""
    device = sigma.device
    half   = ksize // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")

    sigma = sigma.view(-1, 1, 1, 1)
    gauss = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2 + 1e-8))
    gauss = gauss / (gauss.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return gauss.unsqueeze(1)


def _gabor_kernel_2d(
    freq: torch.Tensor,
    theta: torch.Tensor,
    sigma: float = 2.0,
    ksize: int = 11,
) -> torch.Tensor:
    """freq, theta: (B,) → (B,1,ksize,ksize)"""
    device = freq.device
    B      = freq.shape[0]
    half   = ksize // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")

    theta = theta.view(B, 1, 1)
    freq  = freq.view(B, 1, 1)

    x_rot = gx * torch.cos(theta) + gy * torch.sin(theta)
    y_rot = -gx * torch.sin(theta) + gy * torch.cos(theta)

    gauss   = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
    carrier = torch.cos(2 * math.pi * freq * x_rot)
    kernel  = gauss * carrier
    kernel  = kernel / (kernel.abs().sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return kernel.unsqueeze(1)


# ── AdaptiveFilter ────────────────────────────────────────────────────────────

class AdaptiveFilter(nn.Module):
    """미분 가능 필터 — grouped convolution으로 배치별 다른 커널 적용."""

    def __init__(self, ksize: int = 11):
        super().__init__()
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1
        self.pad   = self.ksize // 2

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : (B,3,H,W)
            params : (B,5) FilterParamNet 출력
        Returns:
            (B,3,H,W)
        """
        B, C, H, W = x.shape
        hp_sigma    = params[:, 0]
        gabor_freq  = params[:, 1]
        gabor_theta = params[:, 2]
        blend_hp    = params[:, 3].view(B, 1, 1, 1)
        blend_gab   = params[:, 4].view(B, 1, 1, 1)

        # ── Gaussian high-pass ────────────────────────────────────────────────
        gauss_k  = _gaussian_kernel_2d(hp_sigma, self.ksize)          # (B,1,k,k)
        gauss_k  = gauss_k.expand(B, C, self.ksize, self.ksize)       # (B,C,k,k)
        blurred  = F.conv2d(
            x.reshape(1, B * C, H, W),
            gauss_k.reshape(B * C, 1, self.ksize, self.ksize),
            padding=self.pad, groups=B * C,
        ).reshape(B, C, H, W)
        high_pass = x + blend_hp * (x - blurred)

        # ── Gabor ─────────────────────────────────────────────────────────────
        gabor_k  = _gabor_kernel_2d(gabor_freq, gabor_theta, ksize=self.ksize)  # (B,1,k,k)
        gabor_k  = gabor_k.expand(B, C, self.ksize, self.ksize)
        gabor_r  = F.conv2d(
            high_pass.reshape(1, B * C, H, W),
            gabor_k.reshape(B * C, 1, self.ksize, self.ksize),
            padding=self.pad, groups=B * C,
        ).reshape(B, C, H, W)

        return high_pass + blend_gab * gabor_r


# ── GazeEstimatorV2 (Phase 2) ─────────────────────────────────────────────────

class GazeEstimatorV2(nn.Module):
    """
    Phase 2: head_pose → FilterParamNet → AdaptiveFilter → Backbone → Regressor
    출력: (B,3) 단위벡터
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
        fpn_hidden    = int(model_cfg.get("filter_param_net_hidden", 32))

        self.filter_param_net = FilterParamNet(hidden=fpn_hidden)
        self.adaptive_filter  = AdaptiveFilter(ksize=11)
        self.backbone = SiameseBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze,
        )
        feat_dim = get_feat_dim(backbone_name)
        self.regressor = Regressor(
            in_dim=feat_dim * 2 + 3,
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
        """Returns: (B,3) 정규화된 시선 단위벡터"""
        params = self.filter_param_net(head_pose)
        left   = self.adaptive_filter(left,  params)
        right  = self.adaptive_filter(right, params)

        feat_l, feat_r = self.backbone(left, right)
        x = torch.cat([feat_l, feat_r, head_pose], dim=1)
        return F.normalize(self.regressor(x), dim=1)
