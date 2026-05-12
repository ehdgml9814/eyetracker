"""
transforms.py — albumentations 증강 파이프라인

주의:
  - HorizontalFlip은 여기서 정의하지 않음 → dataset.py에서 수동 처리
  - Normalize / ToTensorV2 없음 → dataset.py _normalize()에서 처리
  - 입력/출력: HWC uint8 BGR
"""

from __future__ import annotations

import albumentations as A


def get_train_transforms(eye_size: int = 64) -> A.Compose:
    """
    학습용 증강 파이프라인.
    이미 eye_size×eye_size로 리사이즈된 눈 크롭이 입력.
    """
    return A.Compose([
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.CoarseDropout(
            max_holes=4,
            max_height=eye_size // 8,
            max_width=eye_size // 8,
            fill_value=0,
            p=0.2,
        ),
    ])


def get_val_transforms() -> A.Compose:
    """검증/테스트용 — 증강 없음."""
    return A.Compose([])
