"""CLAHE 필터 — 적응형 히스토그램 균일화."""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Args:
        image : BGR uint8 ndarray
        params: clahe_clip_limit (float), clahe_tile_grid (int)
    Returns:
        BGR uint8 ndarray
    """
    clip_limit = float(params.get("clahe_clip_limit", 2.0))
    tile_grid  = int(params.get("clahe_tile_grid", 8))

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid),
    )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
