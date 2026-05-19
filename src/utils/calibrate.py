"""
calibrate.py — 웹캠 체커보드 캘리브레이션

사용법:
  python src/utils/calibrate.py                        # 기본값으로 실행
  python src/utils/calibrate.py --cols 9 --rows 6      # 체커보드 크기 지정
  python src/utils/calibrate.py --out data/camera_calib.yaml
  python src/utils/calibrate.py --images data/calib_imgs/  # 저장된 이미지 사용

실행 방법:
  1. 체커보드를 웹캠에 비춘다
  2. SPACE 키 — 현재 프레임 캡처 (코너 검출 성공 시에만 저장됨)
  3. 목표 장수(--count, 기본 20장) 채우면 자동 캘리브레이션
  4. 결과를 --out 경로에 저장 (기본: data/camera_calib.yaml)
  5. q / ESC — 강제 종료 (캡처된 이미지로 즉시 캘리브레이션 시도)

출력 YAML 예시:
  camera_matrix:
    fx: 832.4
    fy: 831.9
    cx: 320.1
    cy: 240.6
  dist_coeffs: [-0.23, 0.11, 0.001, -0.001, -0.05]
  rms_error: 0.42
  image_size: [640, 480]
  board_cols: 9
  board_rows: 6
  square_size_mm: 25.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def _make_obj_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """체커보드 코너 3D 좌표 (Z=0 평면) 생성."""
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


def calibrate_from_images(
    images: list[np.ndarray],
    cols: int,
    rows: int,
    square_mm: float,
) -> dict:
    """
    이미지 리스트로 캘리브레이션 수행.

    Returns:
        dict with camera_matrix (K), dist_coeffs, rms_error, image_size
    """
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    objp = _make_obj_points(cols, rows, square_mm)
    board_size = (cols, rows)
    img_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(refined)

    if len(obj_points) < 5:
        raise RuntimeError(
            f"캘리브레이션에 충분한 이미지가 없습니다 (코너 검출 성공: {len(obj_points)}장, 최소 5장 필요)."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    return {
        "K": K,                        # (3,3) ndarray
        "dist": dist.flatten(),        # (5,) ndarray
        "rms_error": float(rms),
        "image_size": list(img_size),  # [w, h]
        "n_images": len(obj_points),
    }


def save_calib(result: dict, out_path: Path, cols: int, rows: int, square_mm: float) -> None:
    """캘리브레이션 결과를 YAML로 저장."""
    K    = result["K"]
    dist = result["dist"]

    data = {
        "camera_matrix": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "dist_coeffs": [float(v) for v in dist],
        "rms_error":   result["rms_error"],
        "image_size":  result["image_size"],
        "board_cols":  cols,
        "board_rows":  rows,
        "square_size_mm": float(square_mm),
        "n_images": result["n_images"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[calibrate] 저장 완료: {out_path}")
    print(f"  RMS 재투영 오차: {result['rms_error']:.4f} px")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  왜곡: {[f'{v:.4f}' for v in dist]}")


def load_calib(calib_path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    캘리브레이션 YAML 로드.

    Returns:
        (camera_matrix (3,3) float64, dist_coeffs (5,1) float64)
        파일 없으면 None
    """
    p = Path(calib_path)
    if not p.exists():
        return None

    with open(p) as f:
        data = yaml.safe_load(f)

    cm = data["camera_matrix"]
    K = np.array([
        [cm["fx"], 0.0,      cm["cx"]],
        [0.0,      cm["fy"], cm["cy"]],
        [0.0,      0.0,      1.0     ],
    ], dtype=np.float64)

    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def run_interactive(
    cols: int,
    rows: int,
    square_mm: float,
    target_count: int,
    out_path: Path,
    camera_idx: int,
    save_dir: Path | None,
) -> None:
    """웹캠으로 실시간 체커보드 캡처 → 캘리브레이션."""
    board_size = (cols, rows)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera_idx} 열기 실패")

    images_captured: list[np.ndarray] = []
    print(f"\n[calibrate] 체커보드를 카메라에 비추세요 (목표: {target_count}장)")
    print("  SPACE — 캡처 / q, ESC — 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
            status_color = (0, 255, 0)
            status_txt   = f"코너 검출 OK  [{len(images_captured)}/{target_count}]  SPACE=캡처"
        else:
            status_color = (0, 0, 255)
            status_txt   = f"코너 없음      [{len(images_captured)}/{target_count}]"

        cv2.putText(display, status_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[calibrate] 사용자 종료")
            break
        if key == ord(" ") and found:
            images_captured.append(frame.copy())
            print(f"  캡처 {len(images_captured)}/{target_count}")
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = save_dir / f"calib_{len(images_captured):04d}.jpg"
                cv2.imwrite(str(fname), frame)
            if len(images_captured) >= target_count:
                print("[calibrate] 목표 장수 달성, 캘리브레이션 시작...")
                break

    cap.release()
    cv2.destroyAllWindows()

    if not images_captured:
        print("[calibrate] 캡처된 이미지가 없습니다.")
        return

    print(f"[calibrate] {len(images_captured)}장으로 캘리브레이션 중...")
    result = calibrate_from_images(images_captured, cols, rows, square_mm)
    save_calib(result, out_path, cols, rows, square_mm)


def run_from_dir(
    image_dir: Path,
    cols: int,
    rows: int,
    square_mm: float,
    out_path: Path,
) -> None:
    """이미지 디렉토리에서 캘리브레이션."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"이미지 파일 없음: {image_dir}")

    images = [cv2.imread(str(p)) for p in paths]
    images = [img for img in images if img is not None]
    print(f"[calibrate] {len(images)}장 로드 완료, 캘리브레이션 중...")
    result = calibrate_from_images(images, cols, rows, square_mm)
    save_calib(result, out_path, cols, rows, square_mm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="웹캠 체커보드 캘리브레이션 → YAML 저장"
    )
    parser.add_argument("--cols",   type=int,   default=9,
                        help="체커보드 내부 코너 열 수 (기본: 9)")
    parser.add_argument("--rows",   type=int,   default=6,
                        help="체커보드 내부 코너 행 수 (기본: 6)")
    parser.add_argument("--square", type=float, default=25.0,
                        help="정사각형 한 변 크기 mm (기본: 25.0)")
    parser.add_argument("--count",  type=int,   default=20,
                        help="캡처 목표 장수 (기본: 20)")
    parser.add_argument("--camera", type=int,   default=0,
                        help="웹캠 인덱스 (기본: 0)")
    parser.add_argument("--out",    default="data/camera_calib.yaml",
                        help="결과 YAML 저장 경로 (기본: data/camera_calib.yaml)")
    parser.add_argument("--images", default=None,
                        help="기존 이미지 디렉토리 경로 (지정 시 웹캠 대신 사용)")
    parser.add_argument("--save-images", default=None,
                        help="캡처한 이미지를 저장할 디렉토리 (선택)")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.images:
        run_from_dir(Path(args.images), args.cols, args.rows, args.square, out_path)
    else:
        save_dir = Path(args.save_images) if args.save_images else None
        run_interactive(
            cols=args.cols,
            rows=args.rows,
            square_mm=args.square,
            target_count=args.count,
            out_path=out_path,
            camera_idx=args.camera,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
