"""
detector.py — MediaPipe FaceLandmarker (Tasks API, 0.10+)

EyeLandmarker:
    detect(frame_bgr)               → {'left': (x1,y1,x2,y2), 'right': ...}
    detect_with_landmarks(frame_bgr) → (bboxes, landmarks_norm)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── 랜드마크 인덱스 ──────────────────────────────────────────────────────────
# MediaPipe FaceLandmarker 478 점 중 눈 영역
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                  263, 466, 388, 387, 386, 385, 384, 398]

# PnP용 3D 모델 포인트 (mm, 원점=코끝)
_3D_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],  # 코끝
    [0.0,  -330.0, -65.0  ],  # 턱
    [-225.0, 170.0, -135.0],  # 왼쪽 눈 바깥
    [225.0,  170.0, -135.0],  # 오른쪽 눈 바깥
    [-150.0,-150.0, -125.0],  # 입 왼쪽
    [150.0, -150.0, -125.0],  # 입 오른쪽
], dtype=np.float64)

# 위 6점에 대응하는 FaceLandmarker 인덱스
_PNP_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# 모델 파일 URL
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model(model_path: str | Path) -> Path:
    """face_landmarker.task 파일 자동 다운로드."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[detector] 모델 다운로드 중: {_MODEL_URL}")
        try:
            urllib.request.urlretrieve(_MODEL_URL, model_path)
        except Exception as e:
            raise RuntimeError(
                f"FaceLandmarker 모델 다운로드 실패: {e}\n"
                f"수동 다운로드 후 {model_path}에 배치하세요."
            ) from e
        print(f"[detector] 저장 완료: {model_path}")
    return model_path


def _build_landmarker(model_path: Path) -> mp_vision.FaceLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_tracking_confidence=0.1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


class EyeLandmarker:
    """
    MediaPipe FaceLandmarker 기반 눈 영역 검출기.

    Args:
        model_path: face_landmarker.task 경로 (없으면 자동 다운로드)
        padding   : bbox 패딩 비율 (눈 폭/높이 대비)
    """

    def __init__(
        self,
        model_path: str | Path = "/workspace/data/models/face_landmarker.task",
        padding: float = 0.3,
    ):
        model_path = _ensure_model(model_path)
        self._landmarker = _build_landmarker(model_path)
        self._padding = padding

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _get_eye_bbox(
        self,
        landmarks,
        indices: list[int],
        h: int,
        w: int,
    ) -> Optional[tuple[int, int, int, int]]:
        """랜드마크 인덱스 → 패딩 포함 bbox (x1, y1, x2, y2)."""
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pw = (x2 - x1) * self._padding
        ph = (y2 - y1) * self._padding
        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(w, int(x2 + pw))
        y2 = min(h, int(y2 + ph))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _run(self, frame_bgr: np.ndarray):
        """BGR → MediaPipe 결과 반환."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Optional[dict[str, tuple[int, int, int, int]]]:
        """
        Returns:
            {'left': (x1,y1,x2,y2), 'right': (x1,y1,x2,y2)}
            검출 실패 시 None
        """
        result = self._run(frame_bgr)
        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lm = result.face_landmarks[0]

        left_bbox  = self._get_eye_bbox(lm, LEFT_EYE_IDX, h, w)
        right_bbox = self._get_eye_bbox(lm, RIGHT_EYE_IDX, h, w)

        if left_bbox is None or right_bbox is None:
            return None

        return {"left": left_bbox, "right": right_bbox}

    def detect_with_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Optional[tuple[dict, np.ndarray]]:
        """
        눈 bbox와 PnP에 필요한 2D 랜드마크 좌표를 함께 반환.

        Returns:
            (bboxes_dict, landmark_2d)  또는 None
            landmark_2d : (6, 2) float32 — _PNP_LANDMARK_IDX 순서
        """
        result = self._run(frame_bgr)
        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lm = result.face_landmarks[0]

        left_bbox  = self._get_eye_bbox(lm, LEFT_EYE_IDX, h, w)
        right_bbox = self._get_eye_bbox(lm, RIGHT_EYE_IDX, h, w)

        if left_bbox is None or right_bbox is None:
            return None

        bboxes = {"left": left_bbox, "right": right_bbox}

        pts_2d = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in _PNP_LANDMARK_IDX],
            dtype=np.float32,
        )

        return bboxes, pts_2d

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def solve_head_pose(
    pts_2d: np.ndarray,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """
    PnP 솔버로 헤드포즈 추정.

    Args:
        pts_2d      : (6, 2) float32 2D 랜드마크 (detect_with_landmarks 반환값)
        frame_shape : (height, width)

    Returns:
        (3,) float32 — [pitch, yaw, roll] 라디안
    """
    h, w = frame_shape
    focal = w
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array(
        [[focal, 0, cx],
         [0, focal, cy],
         [0, 0,  1  ]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _3D_MODEL_POINTS,
        pts_2d.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return np.zeros(3, dtype=np.float32)

    rmat, _ = cv2.Rodrigues(rvec)
    # sy = sqrt(rmat[0,0]^2 + rmat[1,0]^2)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw   = float(np.arctan2(-rmat[2, 0], sy))
        roll  = float(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = float(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = float(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0

    return np.array([pitch, yaw, roll], dtype=np.float32)
