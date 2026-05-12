# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run inside the Docker container (`docker-compose run --rm dev` or `train`):

```bash
# 전처리 (1회)
docker-compose run --rm dev python src/data/preprocess.py

# 실험 실행
docker-compose run --rm train python src/run.py experiments/phase1_step1_det.yaml
docker-compose run --rm train python src/run.py experiments/phase1_step1_det.yaml --list
docker-compose run --rm train python src/run.py experiments/phase1_step1_det.yaml --dry-run
docker-compose run --rm train python src/run.py experiments/phase1_step1_det.yaml --force

# 개별 실행
docker-compose run --rm train python src/train.py --exp-dir /workspace/runs/my_exp --set model.backbone=resnet18
docker-compose run --rm train python src/eval.py --exp-dir /workspace/runs/my_exp
docker-compose run --rm dev python src/compare.py

# 실시간 추론
docker-compose run --rm dev python src/infer.py --exp-dir /workspace/runs/exp_best

# 데이터 다운로드
docker-compose run --rm dev python src/data/download.py --source official
```

## Architecture

### Pipeline Flow

```
원본 프레임
  ├── apply_det(frame, cfg)   → FaceLandmarker → 눈 bbox
  └── apply_pose(frame, cfg)  → FaceLandmarker + PnP → head_pose

bbox로 원본 프레임 크롭 (필터 미적용)
  └── apply_crop(crop, cfg)   → GazeNet 입력

GazeEstimator(left, right, head_pose) → gaze 3D 단위벡터
```

### Key Design Decisions

- **gaze 표현**: pitch/yaw 대신 3D 단위벡터 (N,3) — 정보손실 없음, 극점 불안정 없음
- **손실 함수**: `1 - cosine_similarity(pred, gt)` — angular error를 직접 최소화
- **평가**: `arccos(dot(pred, gt))` — angular error (degrees)
- **필터 포인트 3개**: `det` (검출용), `pose` (헤드포즈용), `crop` (GazeNet 입력용)
- **두 단계 파라미터 조회**: `filters.<name>.params`(전역) + `category.<point>.params`(포인트 오버라이드)
- **HorizontalFlip**: `dataset.py`에서 수동 처리 (left↔right 이미지 swap + gaze x성분 부호 반전). albumentations 미사용
- **정규화**: `dataset.py._normalize()`에서만 수행. `transforms.py`에는 Normalize/ToTensorV2 없음
- **Phase 1**: 고정 필터 (`GazeEstimator`)
- **Phase 2**: 미분가능 어댑티브 필터 (`GazeEstimatorV2`, crop=adaptive 시 자동 선택)
- **실험 자동 스킵**: `result.yaml`에 `test_angular_err`가 있으면 재실행 안 함

### MediaPipe 0.10 (Tasks API)

`mp.solutions` 없음. `mediapipe.tasks.python.vision.FaceLandmarker` 사용.
Dockerfile에 `libegl1` + `libgles2` 필수.
모델 파일: `data/models/face_landmarker.task` (없으면 자동 다운로드).

### Data Format

```
data/raw/Data/
  Original/p??/day??/NNNN.jpg          ← 원본 얼굴 이미지
  Normalized/p??/day??.mat             ← gaze(3D 단위벡터) + head_pose 원본

mat['data']['left/right']['gaze']      # (N,3) 3D 단위벡터
좌/우안 평균 후 정규화하여 저장
np.atleast_2d() 필수 (단일 샘플 시 shape (3,))
```

### HDF5 Output

```
data/processed/{train,val,test}.h5
  left_eye  : (N, 3, 64, 64) uint8 BGR CHW
  right_eye : (N, 3, 64, 64) uint8 BGR CHW
  gaze      : (N, 3) float32  3D 단위벡터 (MPIIGaze 좌표계: x=오른쪽, y=아래, z=카메라반대)
  head_pose : (N, 3) float32  [pitch, yaw, roll] rad (PnP 추정)
```

### Experiment YAML Format

```yaml
name: phase1_step1_det
sweep:
  fixed:
    category.pose.selected: none
  grid:
    category.det.selected: [none, clahe, gamma, bilateral]
    model.backbone: [resnet18, mobilenet_v2]
```

`grid`의 Cartesian product → N개 실험 자동 전개.
Step 2~4, Phase 2 YAML의 `fixed` 값은 이전 단계 결과를 보고 직접 수정 필요.

### Config Override

```bash
--set category.det.selected=clahe
--set model.backbone=mobilenet_v2
--set filters.clahe.params.clahe_clip_limit=3.0
```

### Output Structure

```
runs/exp_{det}_{pose}_{crop}_{backbone}/
  config.yaml       # 실험 설정 전체
  train_log.yaml    # epoch별 로그 (리스트)
  best.pt           # 최적 체크포인트
  result.yaml       # val/test angular error
```
