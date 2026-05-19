"""
kernel_net.py — KernelNet + DynamicFilter

KernelNet   : (left, right) → kernels (B, 3, kernel_size, kernel_size)
DynamicFilter: (x, kernels) → filtered_x (B, 3, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelNet(nn.Module):
    """
    좌/우 눈 이미지 → 채널별 depthwise conv 커널 생성.

    Args:
        kernel_hidden : FC 히든 크기 (S=128 / M=512 / L=2048)
        kernel_size   : 출력 커널 크기 (기본 5, 홀수)
    """

    def __init__(self, kernel_hidden: int = 512, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

        # Conv Encoder: (B,3,64,64) → (B,128)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                           # (B,32,32,32)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                           # (B,64,16,16)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                              # (B,128,1,1)
            nn.Flatten(),                                              # (B,128)
        )

        # FC: 128 → kernel_hidden → 3*k²
        self.fc = nn.Sequential(
            nn.Linear(128, kernel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(kernel_hidden, 3 * kernel_size * kernel_size),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left  : (B, 3, 64, 64)
            right : (B, 3, 64, 64)
        Returns:
            kernels : (B, 3, kernel_size, kernel_size)
        """
        feat = (self.encoder(left) + self.encoder(right)) / 2.0   # (B,128)
        k    = self.fc(feat)                                        # (B, 3*k²)
        return k.reshape(-1, 3, self.kernel_size, self.kernel_size) # (B,3,k,k)


class DynamicFilter(nn.Module):
    """
    배치별로 다른 커널을 적용하는 depthwise conv.
    파라미터 없음 (연산 모듈).

    Args:
        kernel_size : KernelNet의 kernel_size와 동일하게 설정
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2

    def forward(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, 3, H, W)
            kernels : (B, 3, k, k)
        Returns:
            (B, 3, H, W)
        """
        B, C, H, W = x.shape
        k = self.kernel_size

        x_flat = x.reshape(1, B * C, H, W)           # (1, B*3, H, W)
        k_flat = kernels.reshape(B * C, 1, k, k)     # (B*3, 1, k, k)

        out = F.conv2d(x_flat, k_flat, padding=self.padding, groups=B * C)
        return out.reshape(B, C, H, W)                # (B, 3, H, W)
