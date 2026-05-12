"""
visualize.py — 디버그 / 추론 오버레이 유틸리티
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_gaze_arrow(
    frame: np.ndarray,
    origin: tuple[int, int],
    pitch: float,
    yaw: float,
    length: int = 100,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    gaze 방향 화살표 오버레이.

    Args:
        frame  : BGR ndarray
        origin : (x, y) 화살표 시작점
        pitch  : 라디안 (위↑ 양수)
        yaw    : 라디안 (오른쪽→ 양수)
        length : 화살표 픽셀 길이
    """
    frame = frame.copy()
    dx = int(-np.sin(yaw) * np.cos(pitch) * length)
    dy = int(-np.sin(pitch) * length)
    end = (origin[0] + dx, origin[1] + dy)
    cv2.arrowedLine(frame, origin, end, color, thickness, tipLength=0.3)
    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str = "",
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    bbox (x1, y1, x2, y2) 직사각형 + 레이블 오버레이.
    """
    frame = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            frame, label, (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame


def draw_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.7,
) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA,
    )
    return frame
