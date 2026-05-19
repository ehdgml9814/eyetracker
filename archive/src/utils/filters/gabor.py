"""Gabor 필터 — 텍스처/엣지 강화."""

import cv2
import numpy as np


def apply_gabor(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Gabor 필터를 L 채널에 적용한 뒤 LAB → BGR 변환.
    응답은 원본 L과 가산하여 [0,255] 클리핑.

    Args:
        image : BGR uint8 ndarray
        params: gabor_ksize, gabor_sigma, gabor_freq, gabor_theta,
                gabor_gamma, gabor_psi
    Returns:
        BGR uint8 ndarray
    """
    ksize = int(params.get("gabor_ksize", 31))
    if ksize % 2 == 0:
        ksize += 1

    sigma  = float(params.get("gabor_sigma", 4.0))
    freq   = float(params.get("gabor_freq", 0.3))
    theta  = float(params.get("gabor_theta", 0.0))
    gamma  = float(params.get("gabor_gamma", 0.5))
    psi    = float(params.get("gabor_psi", 0.0))
    lambd  = 1.0 / max(freq, 1e-6)

    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
    )
    # 커널 정규화
    kernel /= (kernel.sum() + 1e-8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_f    = l.astype(np.float32)
    resp   = cv2.filter2D(l_f, cv2.CV_32F, kernel)
    l_new  = np.clip(l_f + resp, 0, 255).astype(np.uint8)

    lab = cv2.merge([l_new, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
