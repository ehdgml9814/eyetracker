"""쌍방향 필터 — 엣지 보존 스무딩."""

import cv2
import numpy as np


def apply_bilateral(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Args:
        image : BGR uint8 ndarray
        params: bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    Returns:
        BGR uint8 ndarray
    """
    d            = int(params.get("bilateral_d", 9))
    sigma_color  = float(params.get("bilateral_sigma_color", 75))
    sigma_space  = float(params.get("bilateral_sigma_space", 75))
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
