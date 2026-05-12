# cv_eyetracker

MPIIGaze 데이터셋 기반 시선 추적 시스템.  
**필터 × 백본 조합 실험**을 통해 최적 구성을 탐색하는 것이 핵심 목표다.

---

## 팀 구성

| 이름 | 역할 |
|------|------|
| 윤동희 | 필터 개발 / 실험 비교 분석 / 보고서 작성 |
| 유성현 | 개발환경 구축 / 파이프라인 설계·구현 / CNN+MediaPipe 모델 구현 |

---

## 목차

1. [개발환경 세팅](#1-개발환경-세팅)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [전체 파이프라인](#3-전체-파이프라인)
4. [필터 시스템](#4-필터-시스템)
5. [모델 구조](#5-모델-구조)
6. [설정 파일](#6-설정-파일)
7. [실험 설계](#7-실험-설계)
8. [실행 방법](#8-실행-방법)
9. [로그 포맷](#9-로그-포맷)

---

## 1. 개발환경 세팅

### 서비스 구성

| 서비스 | 베이스 이미지 | GPU | shm | 용도 |
|--------|------------|-----|-----|------|
| `dev` | `python:3.10-slim` | ❌ | 2 GB | 데이터 다운로드·전처리·코드 테스트 |
| `train` | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | ✅ 전체 | 8 GB | GPU 학습 |

### 사전 요구사항

| 항목 | dev | train |
|------|-----|-------|
| Docker Desktop | ✅ | ✅ |
| NVIDIA GPU | ❌ | ✅ |
| NVIDIA Driver | ❌ | ≥ 520.x |
| NVIDIA Container Toolkit | ❌ | ✅ |

#### NVIDIA Container Toolkit 설치 (Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 초기 설정

**1. Kaggle API 키 발급**

[kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings) 에서 발급 후 프로젝트 루트에 `kaggle.json` 생성:

```json
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
```

> `kaggle.json`은 `.gitignore`에 등록되어 있어 커밋되지 않는다.

**2. Docker 이미지 빌드**

```bash
docker-compose build dev    # 전처리·테스트용
docker-compose build train  # GPU 학습용
```

**3. 컨테이너 실행 (VS Code Remote Containers 연결용)**

```bash
docker-compose up -d dev train
```

### 환경 검증

```bash
# dev — 패키지 확인
docker-compose run --rm dev python -c "
import torch, cv2, mediapipe, h5py, albumentations
print('torch     :', torch.__version__)
print('cv2       :', cv2.__version__)
print('mediapipe :', mediapipe.__version__)
print('h5py      :', h5py.__version__)
print('OK')
"

# train — GPU 확인
docker-compose run --rm train python -c "
import torch
print('torch :', torch.__version__)
print('CUDA  :', torch.cuda.is_available())
print('cuDNN :', torch.backends.cudnn.version())
print('GPU   :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

# 임포트 경로 확인 (PYTHONPATH=/workspace)
docker-compose run --rm dev python -c "
from src.utils.config import load_config
from src.models.gaze_model import GazeEstimator
from src.models.detector import EyeLandmarker
print('import OK')
"

# Kaggle 인증 확인
docker-compose run --rm dev kaggle datasets list --max-size 1
```

---

## 2. 프로젝트 구조

```
cv_eyetracker/
│
├── Dockerfile                    # dev / train 멀티스테이지 빌드
├── docker-compose.yml
├── requirements_dev.txt          # CPU 환경 의존성
├── requirements_train.txt        # GPU 환경 의존성 (cu118)
├── kaggle.json                   # ⚠️ 직접 생성 필요 (gitignore)
│
├── configs/
│   ├── static.yaml               # 고정 설정 (경로, 전처리 임계값, 인프라)
│   ├── dynamic.yaml              # 모델 구조 + 학습 하이퍼파라미터
│   └── filters.yaml              # 필터 포인트별 선택 + 파라미터
│
├── data/
│   ├── raw/                      # MPIIGaze 원본 — gitignore
│   └── processed/                # HDF5 — gitignore
│       ├── train.h5
│       ├── val.h5
│       └── test.h5
│
├── src/
│   ├── data/
│   │   ├── download.py           # Kaggle → data/raw/
│   │   ├── preprocess.py         # 전처리 → HDF5
│   │   └── dataset.py            # MPIIGazeDataset (DataLoader용)
│   │
│   ├── models/
│   │   ├── detector.py           # EyeLandmarker (MediaPipe FaceMesh)
│   │   ├── backbone.py           # build_backbone()
│   │   ├── gaze_model.py         # GazeEstimator (Phase 1/2 통합)
│   │   └── adaptive_filter.py    # FilterParamNet + AdaptiveFilter (Phase 2)
│   │
│   ├── utils/
│   │   ├── config.py             # 설정 로드·병합·조회
│   │   ├── filter_pipeline.py    # apply_det / apply_pose / apply_crop
│   │   ├── filters/              # 필터 구현체
│   │   │   ├── clahe.py
│   │   │   ├── gamma.py
│   │   │   ├── bilateral.py
│   │   │   ├── high_pass.py
│   │   │   ├── gabor.py
│   │   │   └── adaptive.py       # 추론용 numpy 버전
│   │   ├── transforms.py         # albumentations 증강
│   │   ├── metrics.py            # angular_error()
│   │   └── visualize.py          # 학습 커브 / 웹캠 오버레이
│   │
│   ├── train.py                  # 학습 루프
│   ├── eval.py                   # test set 평가
│   ├── compare.py                # 실험 결과 비교표
│   └── infer.py                  # 실시간 웹캠 추론
│
└── runs/                         # 실험 결과 — gitignore
    ├── exp_{det}_{pose}_{crop}_{backbone}/
    │   ├── config.yaml
    │   ├── train_log.yaml
    │   ├── best.pt
    │   └── result.yaml
    ├── experiments_summary.yaml
    └── preprocess/
        └── YYYYMMDD_HHMMSS.yaml
```

---

## 3. 전체 파이프라인

### STEP 0 — 데이터 준비

```
Kaggle API
    │
    ▼
MPIIGaze (~213,000장, 15명)  →  data/raw/
    p00/day01/0000.jpg  +  0000.txt ([pitch_rad, yaw_rad])
    p01/day01/ ...
    ...
```

### STEP 1 — 전처리 (1회 실행)

전처리는 **필터를 적용하지 않은 원본 크롭**을 HDF5에 저장한다.  
필터는 학습 시 실시간으로 적용하므로, 전처리는 한 번만 실행하면 된다.

```
data/raw/ 이미지
    │
    ▼
┌─────────────────────────────────────────┐
│  품질 필터링 (4가지 기준)                 │
│                                         │
│  ① 밝기: mean ∉ [30, 220]  → 스킵      │
│  ② 블러: Laplacian var < 50 → 스킵      │
│  ③ 시선: |pitch| or |yaw| > 40° → 스킵  │
│  ④ MediaPipe 검출 실패 → 스킵            │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
 apply_det(frame)         apply_pose(frame)
 (det 필터 적용)           (pose 필터 적용)
       │                       │
       ▼                       ▼
 MediaPipe FaceMesh       MediaPipe FaceMesh
 468 랜드마크              468 랜드마크
 → 좌/우안 bbox            → PnP Solver
                           → head_pose
                             (pitch, yaw, roll)
       │                       │
       └───────────┬───────────┘
                   │
       원본 프레임에서 bbox 크롭  ← 크롭은 항상 필터 미적용 원본 사용
                   │
           64×64 리사이즈
                   │
                   ▼
     ┌─────────────────────────────┐
     │          HDF5 저장           │
     │                             │
     │  left_eye  (N, 3, 64, 64)  │  ← uint8 RGB, CHW
     │  right_eye (N, 3, 64, 64)  │
     │  gaze      (N, 2)          │  ← [pitch, yaw] 라디안 (MPIIGaze 레이블)
     │  head_pose (N, 3)          │  ← [pitch, yaw, roll] 라디안 (MediaPipe 추정)
     └─────────────────────────────┘

     분할 비율 (random shuffle, seed=42):
       train : val : test = 80% : 10% : 10%
```

> **왜 원본 크롭을 저장하는가?**
> 필터(clahe, gabor 등)를 HDF5에 구워버리면 필터를 바꿀 때마다 전처리를 다시 해야 한다.
> 원본을 저장해두면 필터만 바꿔서 학습을 반복할 수 있다.

### STEP 2 — 학습

```
HDF5 (left_eye, right_eye, head_pose, gaze)
    │
    ▼
dataset.py — __getitem__ 처리 순서:

  1. HDF5에서 (3,H,W) uint8 로드
  2. (H,W,3) HWC 변환
  3. HorizontalFlip (50% 확률, train only)
       └── 수동 처리:
           ① left_eye ↔ right_eye 교환
           ② 각 이미지 좌우 반전 ([:, ::-1])
           ③ gaze yaw 부호 반전 ([pitch, -yaw])
           → albumentations는 gaze 보정 불가 → 수동 필수
  4. apply_crop(eye, cfg)     ← crop 필터 실시간 적용
  5. albumentations 증강 (train only)
       Rotate(±10°) / ShiftScaleRotate / RandomBrightnessContrast / GaussNoise
  6. _normalize(): float32 변환 + ImageNet 정규화 + CHW 변환
       (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  7. torch.Tensor 반환
    │
    ▼
GazeEstimator
    │
    ▼
MSELoss(pred, gaze_label)
    │
    ▼
Angular Error (°) 모니터링
```

### STEP 3 — 평가

```
best.pt 로드
    │
    ▼
test.h5 (증강 없음, crop 필터만 적용)
    │
    ▼
Angular Error 계산 → result.yaml 에 test_angular_err 기록
```

### STEP 4 — 실험 비교

```
runs/exp_*/result.yaml 전체 로드
    │
    ▼
Phase 1 Step 1~3 순위표 출력
Phase 2 Adaptive 비교
    │
    ▼
experiments_summary.yaml 저장
```

### STEP 5 — 실시간 추론

```
웹캠 프레임 (원본 보존)
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
apply_det          apply_pose             │
(det 필터)         (pose 필터)            │
    │                  │                  │
    ▼                  ▼                  │
MediaPipe          MediaPipe              │
→ 눈 bbox         → head_pose            │
    │                  │                  │
    └──────────┬────────┘                  │
               │                          │
    원본 프레임에서 bbox 크롭 ◄────────────┘
               │
         apply_crop
         (crop 필터)
               │
         GazeEstimator (best.pt)
               │
         pitch, yaw
               │
         cv2 오버레이 → 화면 출력
```

---

## 4. 필터 시스템

### 필터 포인트 개념

필터는 3개의 독립 포인트에 각각 다르게 적용된다.

| 포인트 | 적용 대상 | 이미지 크기 | 지원 필터 |
|--------|----------|------------|----------|
| `det` | 전체 프레임 → MediaPipe 눈 랜드마크 검출 입력 | ~640×480 | none / clahe / gamma / bilateral |
| `pose` | 전체 프레임 → MediaPipe 헤드포즈 추정 입력 | ~640×480 | none / clahe / gamma / bilateral |
| `crop` | 눈 크롭 → GazeEstimator 입력 | 64×64 | none / high_pass / gabor / both / adaptive |

- `det`와 `pose`는 같은 필터 종류를 사용하지만 **파라미터가 완전히 독립**
- `crop`은 이미지가 64×64로 작으므로 커널 크기를 별도로 설정해야 함
- 3개 포인트 모두 `none`이면 필터 없이 원본 그대로 사용

### 파라미터 우선순위 (two-level lookup)

```
최종 파라미터 = filters.<name>.params (전역 기본값)
               + category.<point>.params (포인트 오버라이드, 우선)
```

예시:
```yaml
# filters.yaml
filters:
  bilateral:
    params:
      bilateral_d: 9        # 전역 기본값

category:
  det:
    selected: bilateral
    params:
      bilateral_d: 9        # det는 9 (전체 프레임)
  crop:
    selected: bilateral
    params:
      bilateral_d: 3        # crop은 3 (64×64이므로 더 작게)
```

### 필터별 동작 원리

#### det / pose 포인트 필터

| 필터 | 동작 | 목적 |
|------|------|------|
| `none` | 원본 그대로 | 기준선(baseline) |
| `clahe` | 제한 대비 적응 히스토그램 평활화 | 조명 불균일 보정 |
| `gamma` | 픽셀값^(1/γ) 적용 | 전체 밝기 조정 |
| `bilateral` | 노이즈 제거 + 엣지 보존 필터 | 랜드마크 검출 정확도 향상 |

#### crop 포인트 필터

| 필터 | 동작 | 목적 |
|------|------|------|
| `none` | 원본 그대로 | 기준선 |
| `high_pass` | `I + 0.5 × (I - GaussianBlur(I, σ))` | 홍채 경계·엣지 강조 |
| `gabor` | 방향성 Gabor 커널 적용 후 정규화 | 홍채 패턴 방향성 강조 |
| `both` | high_pass → gabor 순차 적용 | 두 필터 효과 결합 |
| `adaptive` | FilterParamNet이 예측한 파라미터로 동적 혼합 | Phase 2 전용 |

#### Adaptive 필터 (Phase 2)

```
head_pose (B, 3)
    │
    ▼
FilterParamNet
  FC(3 → 32) → ReLU → FC(32 → 5)
    │
    ▼
filter_params:
  high_pass_sigma  (softplus → 양수)
  gabor_freq       (softplus → 양수)
  gabor_theta      (자유 라디안)
  blend_high_pass  (sigmoid → [0, 1])
  blend_gabor      (sigmoid → [0, 1])
    │
    ▼
AdaptiveFilter (differentiable, F.conv2d)
  output = (hp_result × blend_hp + gabor_result × blend_gab) / (blend_hp + blend_gab)
    │
    ▼
눈 크롭 (필터 적용됨) → Backbone
```

> 학습 시: `models/adaptive_filter.py` (PyTorch, 미분 가능)  
> 추론 시: `utils/filters/adaptive.py` (numpy, 속도 우선)

---

## 5. 모델 구조

### 입출력

| 항목 | Shape | 설명 |
|------|-------|------|
| `left_eye` | (B, 3, 64, 64) | 좌안 크롭 — ImageNet 정규화 float32 |
| `right_eye` | (B, 3, 64, 64) | 우안 크롭 |
| `head_pose` | (B, 3) | [pitch, yaw, roll] 라디안 |
| **출력** | **(B, 2)** | **[pitch, yaw] 라디안** |

### Phase 1 — Fixed Filter

```
left_eye  ─┐
            ├── Siamese Backbone (좌/우안 가중치 공유)
right_eye ─┘
            ├── resnet18    → feat_dim = 512
            └── mobilenet_v2 → feat_dim = 1280

feat_l (B, feat_dim)
feat_r (B, feat_dim)
    │
concat([feat_l, feat_r, head_pose])  →  (B, feat_dim×2 + 3)
    │
FC(→ regressor_hidden=256) → ReLU → Dropout(0.3)
    × regressor_layers (기본 1회)
    │
FC(256 → 2)
    │
pitch, yaw  (라디안)
```

**왜 Siamese(공유 가중치)인가?**  
좌안과 우안은 같은 종류의 이미지이므로 동일한 특징 추출기를 사용해도 된다.  
파라미터 수를 절반으로 줄이고 두 눈의 일관성을 강제하는 효과가 있다.

### Phase 2 — Adaptive Filter

Phase 1 구조에서 Backbone 앞에 AdaptiveFilter가 추가된다.

```
head_pose → FilterParamNet → filter_params
                                  │
eye_crop → AdaptiveFilter ────────┘ → 필터 적용된 크롭 → Backbone → feat
```

`category.crop.selected = adaptive` 로 설정 시 자동으로 Phase 2로 동작한다.

### 손실 함수 및 평가 지표

**학습**: MSE Loss — `L = mean((pred_pitch - gt_pitch)² + (pred_yaw - gt_yaw)²)`

**평가**: Angular Error (°)

```
pred_vec   = [cos(pitch)·sin(yaw),  -sin(pitch),  cos(pitch)·cos(yaw)]
target_vec = [cos(pitch)·sin(yaw),  -sin(pitch),  cos(pitch)·cos(yaw)]

angular_error = arccos(clip(pred_vec · target_vec, -1, 1)) × 180/π
```

낮을수록 좋음. 논문 SoTA 기준 ~4° 수준.

---

## 6. 설정 파일

설정은 3개 파일로 분리되어 있고, 학습 시 CLI `--set`으로 오버라이드할 수 있다.

### static.yaml — 고정 설정

```yaml
dataset:
  raw_dir: /workspace/data/raw
  processed_dir: /workspace/data/processed
  train_split: 0.8          # train:val:test = 0.8:0.1:0.1
  eye_size: 64

preprocess:
  brightness_min: 30
  brightness_max: 220
  blur_threshold: 50
  gaze_angle_max_deg: 40
```

### dynamic.yaml — 모델·학습 변수

```yaml
model:
  backbone: resnet18         # resnet18 | mobilenet_v2
  pretrained: true
  freeze_backbone: false
  regressor_hidden: 256
  regressor_layers: 1
  dropout: 0.3
  filter_param_net_hidden: 32  # Phase 2 전용

train:
  batch_size: 256
  epochs: 50
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_scheduler: cosine
```

### filters.yaml — 필터 설정

```yaml
category:
  det:
    selected: none            # 적용할 필터 선택
    params: {}                # 포인트별 파라미터 오버라이드
  pose:
    selected: none
    params: {}
  crop:
    selected: none
    params: {}

filters:                      # 전역 기본값 (모든 포인트 공통)
  clahe:
    params:
      clahe_clip_limit: 2.0
      clahe_tile_grid: 8
  bilateral:
    params:
      bilateral_d: 9
      bilateral_sigma_color: 75
      bilateral_sigma_space: 75
  high_pass:
    params:
      high_pass_ksize: 0      # 0 = sigma로 자동 계산
      high_pass_sigma: 2.0
  gabor:
    params:
      gabor_ksize: 31
      gabor_sigma: 4.0
      gabor_freq: 0.3
      gabor_theta: 0.0
      gabor_gamma: 0.5
      gabor_psi: 0.0
```

### CLI 오버라이드

```bash
python src/train.py \
  --set category.det.selected=clahe \
  --set category.pose.selected=bilateral \
  --set category.crop.selected=gabor \
  --set category.crop.params.gabor_ksize=15 \   # crop 전용 커널 크기
  --set model.backbone=resnet18 \
  --set train.lr=5e-5
```

---

## 7. 실험 설계

### 실험 변수

| 변수 | 후보 |
|------|------|
| det 필터 | none / clahe / gamma / bilateral |
| pose 필터 | none / clahe / gamma / bilateral |
| crop 필터 | none / high_pass / gabor / both |
| 백본 | resnet18 / mobilenet_v2 |

### Phase 1 — 순차 최적화 (26회)

한 번에 한 포인트씩 탐색한다.

```
Step 1: det  탐색  (pose=none, crop=none 고정) × 2 backbone → 8회 → best_det 확정
Step 2: pose 탐색  (det=best,  crop=none 고정) × 2 backbone → 8회 → best_pose 확정
Step 3: crop 탐색  (det=best,  pose=best 고정) × 2 backbone → 8회 → best_crop 확정
Step 4: best 조합 재확인                        × 2 backbone → 2회
```

### Phase 2 — Adaptive (2회)

```
det=best, pose=best, crop=adaptive × 2 backbone
```

**총 28회 실험**

### 실험 결과 디렉토리

```
runs/exp_{det}_{pose}_{crop}_{backbone}/
  config.yaml       ← 해당 실험의 전체 설정
  train_log.yaml    ← epoch별 loss / angular error
  best.pt           ← val angular error 최소 시점의 가중치
  result.yaml       ← 실험 요약 (val/test error)
```

---

## 8. 실행 방법

### STEP 0 — 데이터 다운로드

```bash
docker-compose run --rm dev python src/data/download.py
```

### STEP 1 — 전처리 (1회)

```bash
docker-compose run --rm dev python src/data/preprocess.py
```

완료 후 `data/processed/train.h5`, `val.h5`, `test.h5` 생성.  
로그: `runs/preprocess/YYYYMMDD_HHMMSS.yaml`

### STEP 2 — 학습

```bash
# 단일 실험
docker-compose run --rm train python src/train.py \
  --set category.det.selected=clahe \
  --set category.crop.selected=high_pass \
  --set model.backbone=resnet18

# Phase 1 전체 스크립트 (best_det 결정 후 Step 2~4 반복)
for det in none clahe gamma bilateral; do
  for bb in resnet18 mobilenet_v2; do
    docker-compose run --rm train python src/train.py \
      --set category.det.selected=$det \
      --set model.backbone=$bb
  done
done
```

### STEP 3 — 평가

```bash
# 단일 실험
docker-compose run --rm train python src/eval.py \
  --exp runs/exp_clahe_none_high_pass_resnet18

# 전체 일괄 평가
docker-compose run --rm train python src/eval.py
```

### STEP 4 — 실험 비교

```bash
python src/compare.py
# 출력: 터미널 순위표 + runs/experiments_summary.yaml
```

### STEP 5 — 실시간 추론

```bash
python src/infer.py \
  --exp runs/exp_clahe_none_high_pass_resnet18 \
  --cam 0
```

종료: `q` 키

---

## 9. 로그 포맷

### 전처리 로그 (`runs/preprocess/YYYYMMDD_HHMMSS.yaml`)

```yaml
note: 필터 미적용 원본 크롭 저장 — 필터는 학습 시 실시간 적용
total_images: 213659
processed: 211423
skipped_brightness: 893
skipped_blur: 600
skipped_gaze_angle: 343
skipped_mediapipe_fail: 400
skip_rate_pct: 1.04
output:
  train: 169138
  val: 21142
  test: 21143
elapsed_sec: 2712
```

### 학습 로그 (`train_log.yaml`)

```yaml
- epoch: 1
  lr: 0.0001
  train_loss: 0.0842
  val_loss: 0.0791
  val_angular_err: 6.21
- epoch: 2
  lr: 0.000099
  train_loss: 0.0761
  val_loss: 0.0703
  val_angular_err: 5.87
```

### 실험 결과 (`result.yaml`)

```yaml
experiment:
  det_filter: clahe
  pose_filter: none
  crop_filter: high_pass
  backbone: resnet18
best:
  epoch: 47
  val_angular_err: 4.18
  test_angular_err: 4.35
```

### 전체 집계 (`experiments_summary.yaml`)

```yaml
experiments:
  - det_filter: clahe
    pose_filter: none
    crop_filter: high_pass
    backbone: resnet18
    test_angular_err: 4.35
  ...
phase1_best:
  det_filter: clahe
  pose_filter: none
  crop_filter: high_pass
  backbone: resnet18
  test_angular_err: 4.35
phase2_best:
  crop_filter: adaptive
  backbone: resnet18
  test_angular_err: 4.01
improvement_deg: 0.34
```
