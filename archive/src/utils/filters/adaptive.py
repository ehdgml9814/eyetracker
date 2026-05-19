"""
adaptive.py — Phase 2 어댑티브 필터 플레이스홀더.

실제 필터 적용은 src/models/adaptive_filter.py (AdaptiveFilter, PyTorch)에서 수행.
전처리 파이프라인에서는 호출되지 않음 — filter_pipeline.py 참조.
"""

import numpy as np


def apply_adaptive(image: np.ndarray, params: dict) -> np.ndarray:
    """어댑티브 필터는 모델 내부에서 처리 → 전처리 단계에서는 항등 변환."""
    return image
