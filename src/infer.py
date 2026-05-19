"""
infer.py — 실시간 웹캠 추론

모델 출력 (B,3) 단위벡터 → 화살표 표시를 위해 pitch/yaw 역산
(표시 목적에만 사용 — 학습/평가 파이프라인과 무관)

사용법:
  python src/infer.py --exp-dir /workspace/runs/exp_none_none_none_resnet18
  python src/infer.py --exp-dir /workspace/runs/exp_best --camera 0
  python src/infer.py --exp-dir /workspace/runs/exp_best --video input.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import EyeLandmarker, solve_head_pose
# from src.utils.filter_pipeline import apply_det, apply_pose, apply_crop
from src.utils.calibrate import load_calib
from src.utils.metrics import gaze_vec_to_pitchyaw
from src.utils.visualize import draw_gaze_arrow, draw_bbox, draw_text

# ImageNet 통계 (BGR)
_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def _normalize(img_chw: np.ndarray) -> np.ndarray:
    x = img_chw.astype(np.float32) / 255.0
    x[0] = (x[0] - _MEAN[0]) / _STD[0]
    x[1] = (x[1] - _MEAN[1]) / _STD[1]
    x[2] = (x[2] - _MEAN[2]) / _STD[2]
    return x


# def _build_model(cfg: dict) -> torch.nn.Module:
#     crop_selected = cfg.get("category", {}).get("crop", {}).get("selected", "none")
#     if crop_selected == "adaptive":
#         from src.models.adaptive_filter import GazeEstimatorV2
#         return GazeEstimatorV2(cfg)
#     else:
#         from src.models.gaze_model import GazeEstimator
#         return GazeEstimator(cfg)
def _build_model(cfg: dict) -> torch.nn.Module:
    model_type = cfg.get("model", {}).get("type", "proposed")
    if model_type == "proposed":
        from src.models.proposed_model import ProposedModel
        return ProposedModel(cfg)
    elif model_type == "baseline":
        from src.models.baseline_model import BaselineModel
        return BaselineModel(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from [proposed, baseline]")

#cfg 제거
def _prepare_eye(crop: np.ndarray, eye_size: int) -> np.ndarray:
    """원본 크롭 → crop 필터 → 리사이즈 → CHW 정규화 float32."""
    crop = cv2.resize(crop, (eye_size, eye_size))
    # crop = apply_crop(crop, cfg)
    return _normalize(crop.transpose(2, 0, 1))


def infer(exp_dir: Path, source: int | str, calib_path: str | None = None) -> None:
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    eye_size = int(cfg.get("dataset", {}).get("eye_size", 64))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg)
    model.load_state_dict(torch.load(str(exp_dir / "best.pt"), map_location=device))
    model = model.to(device)
    model.eval()

    # ── 카메라 캘리브레이션 로드 ───────────────────────────────────────────────
    # 우선순위: --calib 인수 > data/camera_calib.yaml > 근사값 fallback
    calib_candidates = [
        calib_path,
        "data/camera_calib.yaml",
    ]
    cam_matrix: np.ndarray | None = None
    dist_coeff: np.ndarray | None = None
    for candidate in calib_candidates:
        if candidate is None:
            continue
        result = load_calib(candidate)
        if result is not None:
            cam_matrix, dist_coeff = result
            print(f"[infer] 카메라 캘리브레이션 로드: {candidate}")
            print(f"        fx={cam_matrix[0,0]:.1f}  fy={cam_matrix[1,1]:.1f}"
                  f"  cx={cam_matrix[0,2]:.1f}  cy={cam_matrix[1,2]:.1f}")
            break
    if cam_matrix is None:
        print("[infer] 캘리브레이션 파일 없음 — focal=width 근사값 사용")

    model_path = Path(cfg["dataset"]["raw_dir"]).parent / "models" / "face_landmarker.task"
    landmarker = EyeLandmarker(model_path=str(model_path))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스 열기 실패: {source}")

    print(f"[infer] 시작 (q 또는 ESC로 종료)  device={device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # det_frame  = apply_det(frame, cfg)
        det_result = landmarker.detect(frame)

        if det_result is None:
            overlay = draw_text(frame, "No face detected", color=(0, 0, 255))
        else:
            # pose_frame  = apply_pose(frame, cfg)
            pose_result = landmarker.detect_with_landmarks(frame)

            if pose_result is None:
                overlay = frame.copy()
            else:
                _, pts_2d = pose_result
                head_pose = solve_head_pose(
                    pts_2d, frame.shape[:2], cam_matrix, dist_coeff
                )

                def crop_eye(bbox):
                    x1, y1, x2, y2 = bbox
                    return frame[y1:y2, x1:x2]

                l_crop = crop_eye(det_result["left"])
                r_crop = crop_eye(det_result["right"])

                if l_crop.size == 0 or r_crop.size == 0:
                    overlay = frame.copy()
                else:
                    l_t  = torch.from_numpy(_prepare_eye(l_crop, eye_size)).unsqueeze(0).to(device)
                    r_t  = torch.from_numpy(_prepare_eye(r_crop, eye_size)).unsqueeze(0).to(device)
                    hp_t = torch.from_numpy(head_pose).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gvec = model(l_t, r_t, hp_t).cpu().numpy()[0]  # (3,)

                    # 화살표 표시용으로만 pitch/yaw 역산
                    pitchyaw = gaze_vec_to_pitchyaw(gvec[np.newaxis])[0]
                    pitch, yaw = float(pitchyaw[0]), float(pitchyaw[1])

                    overlay = frame.copy()
                    for side, color in [("left", (255, 100, 0)), ("right", (0, 100, 255))]:
                        bbox = det_result[side]
                        cx   = (bbox[0] + bbox[2]) // 2
                        cy   = (bbox[1] + bbox[3]) // 2
                        overlay = draw_bbox(overlay, bbox, label=side, color=color)
                        overlay = draw_gaze_arrow(overlay, (cx, cy), pitch, yaw, color=color)

                    overlay = draw_text(
                        overlay,
                        f"pitch={np.degrees(pitch):.1f}  yaw={np.degrees(yaw):.1f}",
                    )

        cv2.imshow("GazeTracker", overlay)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[infer] 종료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--video",   default=None)
    parser.add_argument(
        "--calib",
        default=None,
        help="카메라 캘리브레이션 YAML 경로 (없으면 data/camera_calib.yaml 자동 탐색)",
    )
    args = parser.parse_args()

    infer(Path(args.exp_dir), args.video if args.video else args.camera, args.calib)


if __name__ == "__main__":
    main()
