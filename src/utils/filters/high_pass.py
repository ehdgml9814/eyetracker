"""고주파 필터 — 가우시안 블러를 뺀 후 원본과 가산."""

import cv2
import numpy as np


def apply_high_pass(image: np.ndarray, params: dict) -> np.ndarray:
    """
    high_pass = image + (image - gaussian_blur(image))
    결과를 [0,255] 클리핑 후 uint8 반환.

    Args:
        image : BGR uint8 ndarray
        params: high_pass_ksize (int, 0=자동), high_pass_sigma (float)
    Returns:
        BGR uint8 ndarray
    """
    sigma = float(params.get("high_pass_sigma", 2.0))
    ksize = int(params.get("high_pass_ksize", 0))
    if ksize == 0:
        # sigma에서 커널 크기 자동 계산 (최소 3, 반드시 홀수)
        ksize = max(3, int(sigma * 6) | 1)
    elif ksize % 2 == 0:
        ksize += 1

    blurred  = cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigma)
    high     = image.astype(np.float32) - blurred
    enhanced = image.astype(np.float32) + high
    return np.clip(enhanced, 0, 255).astype(np.uint8)
