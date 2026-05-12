"""
metrics.py — Angular Error 계산

gaze는 (N,3) 단위벡터.
Angular Error = arccos(dot(pred_vec, gt_vec)) * 180/pi
"""

from __future__ import annotations

import numpy as np
import torch


def angular_error_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Args:
        pred: (N,3) ndarray — 모델 출력 단위벡터
        gt  : (N,3) ndarray — 정답 단위벡터
    Returns:
        mean angular error (degrees)
    """
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    gt   = gt   / (np.linalg.norm(gt,   axis=1, keepdims=True) + 1e-8)
    dot  = np.clip(np.sum(pred * gt, axis=1), -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(dot))))


def angular_error_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Torch 배치용 (모니터링 전용, 역전파 없음).
    Args:
        pred: (B,3) — 모델 출력 (normalize 완료)
        gt  : (B,3) — 정답 단위벡터
    Returns:
        mean angular error (degrees, Python float)
    """
    with torch.no_grad():
        return angular_error_np(
            pred.detach().cpu().numpy(),
            gt.detach().cpu().numpy(),
        )


def gaze_vec_to_pitchyaw(gvec: np.ndarray) -> np.ndarray:
    """
    (N,3) 단위벡터 → (N,2) [pitch, yaw] 라디안.
    표시 목적(infer.py)에만 사용.

    MPIIGaze 좌표계 (x=오른쪽, y=아래, z=카메라 반대):
      pitch = arcsin(-y)      위↑ = 양수
      yaw   = arctan2(-x, -z) 오른쪽→ = 양수
    """
    gvec  = gvec / (np.linalg.norm(gvec, axis=-1, keepdims=True) + 1e-8)
    pitch = np.arcsin(-gvec[..., 1])
    yaw   = np.arctan2(-gvec[..., 0], -gvec[..., 2])
    return np.stack([pitch, yaw], axis=-1)
