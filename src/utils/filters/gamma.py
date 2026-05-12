"""감마 보정 필터."""

import cv2
import numpy as np


def apply_gamma(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Args:
        image : BGR uint8 ndarray
        params: gamma_value (float, default 1.5)
    Returns:
        BGR uint8 ndarray
    """
    gamma = float(params.get("gamma_value", 1.5))
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255.0
    table = table.astype(np.uint8)
    return cv2.LUT(image, table)
