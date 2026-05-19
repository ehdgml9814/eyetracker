"""
filter_pipeline.py — 포인트별 필터 적용 인터페이스

포인트:
  det  : 전체 프레임 → FaceLandmarker (눈 bbox)
  pose : 전체 프레임 → FaceLandmarker + PnP (head pose)
  crop : 64×64 눈 크롭 → GazeNet 입력

사용법:
    from src.utils.filter_pipeline import apply_det, apply_pose, apply_crop

    filtered_frame = apply_det(frame, cfg)
    filtered_frame = apply_pose(frame, cfg)
    filtered_crop  = apply_crop(crop, cfg)
"""

from __future__ import annotations

import numpy as np

from src.utils.config import get_filter_params
from src.utils.filters.clahe    import apply_clahe
from src.utils.filters.gamma    import apply_gamma
from src.utils.filters.bilateral import apply_bilateral
from src.utils.filters.high_pass import apply_high_pass
from src.utils.filters.gabor    import apply_gabor
from src.utils.filters.adaptive import apply_adaptive


# ── det / pose 용 필터 ────────────────────────────────────────────────────────

_DET_POSE_FILTERS = {
    "none":     lambda img, p: img,
    "clahe":    apply_clahe,
    "gamma":    apply_gamma,
    "bilateral": apply_bilateral,
}

# ── crop 용 필터 ──────────────────────────────────────────────────────────────

def _apply_both(image: np.ndarray, params: dict) -> np.ndarray:
    """high_pass → gabor 순서로 적용."""
    image = apply_high_pass(image, params)
    image = apply_gabor(image, params)
    return image


_CROP_FILTERS = {
    "none":      lambda img, p: img,
    "high_pass": apply_high_pass,
    "gabor":     apply_gabor,
    "both":      _apply_both,
    "adaptive":  apply_adaptive,  # Phase 2: 실제 처리는 모델 내부
}


# ── 공개 API ──────────────────────────────────────────────────────────────────

def _apply(image: np.ndarray, cfg: dict, point: str, registry: dict) -> np.ndarray:
    selected = cfg.get("category", {}).get(point, {}).get("selected", "none")
    if selected not in registry:
        raise ValueError(f"Unknown filter '{selected}' for point '{point}'")
    params = get_filter_params(cfg, selected, point)
    return registry[selected](image, params)


def apply_det(image: np.ndarray, cfg: dict) -> np.ndarray:
    """det 포인트 필터 적용 (전체 프레임)."""
    return _apply(image, cfg, "det", _DET_POSE_FILTERS)


def apply_pose(image: np.ndarray, cfg: dict) -> np.ndarray:
    """pose 포인트 필터 적용 (전체 프레임)."""
    return _apply(image, cfg, "pose", _DET_POSE_FILTERS)


def apply_crop(image: np.ndarray, cfg: dict) -> np.ndarray:
    """crop 포인트 필터 적용 (64×64 눈 크롭)."""
    return _apply(image, cfg, "crop", _CROP_FILTERS)
