# 데이터 1개의 전체 처리 흐름

---

## 원본 파일 구조

```
data/raw/Data/
├── Original/p00/day01/0001.jpg        ← 원본 얼굴 이미지 (640×480 BGR)
└── Normalized/p00/day01.mat           ← gaze 레이블
```

`.mat` 파일 내용:
```python
mat["filenames"]             → "0001.jpg"
mat["data"]["left"]["gaze"]  → [ 0.12, -0.08, -0.99]   # 왼눈이 본 방향 3D 벡터
mat["data"]["right"]["gaze"] → [ 0.11, -0.07, -0.99]   # 오른눈이 본 방향 3D 벡터
```

> **MPIIGaze 좌표계** : x = 오른쪽, y = 아래, z = 카메라 반대 방향

---

## STEP 1 — 전처리 (`preprocess.py`, 1회 실행)

### 1-1. gaze 벡터 계산
```
좌/우안 평균 → 단위벡터 정규화

avg  = ([0.12, -0.08, -0.99] + [0.11, -0.07, -0.99]) / 2
     = [0.115, -0.075, -0.990]

gvec = avg / ‖avg‖  =  [0.115, -0.075, -0.989]   shape: (3,)  float32
```

### 1-2. 품질 필터
| 검사 항목 | 기준 | 판정 |
|-----------|------|------|
| 밝기 (회색조 평균) | 30 ~ 220 | ✅ |
| 선명도 (Laplacian 분산) | ≥ 50 | ✅ |
| 시선 각도 (`arccos(-gvec[2])`) | ≤ 40° | ✅ (≈8.5°) |

탈락 시 해당 샘플은 HDF5에 포함되지 않음.

### 1-3. det 포인트 필터 → 눈 bbox 검출
```
apply_det(frame, cfg)
  └─ cfg.category.det.selected = "none"  → frame 그대로 사용
     cfg.category.det.selected = "clahe" → CLAHE 적용 후 사용

EyeLandmarker.detect(det_frame)
  └─ FaceLandmarker (Tasks API, mediapipe 0.10)
  → left_bbox  = (210, 185, 260, 210)   # (x1, y1, x2, y2)
  → right_bbox = (370, 183, 420, 208)
```

### 1-4. pose 포인트 필터 → 헤드포즈 추정
```
apply_pose(frame, cfg)
  └─ cfg.category.pose.selected 기반 필터 적용

EyeLandmarker.detect_with_landmarks(pose_frame)
  └─ 코끝·턱·눈 바깥·입 끝 6점 2D 좌표 반환  shape: (6, 2)

solve_head_pose(pts_2d, frame_shape)
  └─ cv2.solvePnP (3D 얼굴 모델 ↔ 2D 랜드마크)
  → head_pose = [-0.08, 0.03, 0.01]   # [pitch, yaw, roll]  라디안  shape: (3,)
```

### 1-5. 원본 프레임에서 눈 크롭 (필터 미적용)
```
크롭은 det/pose 필터와 무관하게 항상 원본 frame 사용

left_crop  = frame[185:210, 210:260]          # (25, 50, 3)  HWC BGR
left_crop  = cv2.resize(left_crop, (64, 64))  # (64, 64, 3)
left_chw   = left_crop.transpose(2, 0, 1)     # (3, 64, 64)  CHW
```

### 1-6. HDF5 저장
```
data/processed/train.h5  (or val.h5 / test.h5)
  left_eye  [i]  →  (3, 64, 64)  uint8   BGR CHW
  right_eye [i]  →  (3, 64, 64)  uint8   BGR CHW
  gaze      [i]  →  (3,)         float32  [0.115, -0.075, -0.989]
  head_pose [i]  →  (3,)         float32  [-0.08,  0.03,  0.01]
```

분할 비율: train 80% / val 10% / test 10%  (seed=42 고정)

---

## STEP 2 — 배치 구성 (`dataset.py`, 학습 중 매 스텝)

DataLoader가 idx=42를 요청하면:

### 2-1. HDF5 읽기
```
left_chw  = h5["left_eye"][42]    (3, 64, 64)  uint8
right_chw = h5["right_eye"][42]   (3, 64, 64)  uint8
gaze      = h5["gaze"][42]        (3,)          float32  [0.115, -0.075, -0.989]
head_pose = h5["head_pose"][42]   (3,)          float32  [-0.08, 0.03, 0.01]
```

### 2-2. HorizontalFlip (학습 시 50% 확률)
```
이번 샘플: flip 발동

① 이미지 교환
   left_chw  ↔  right_chw

② 각 이미지 좌우 픽셀 반전
   left_chw  = left_chw[:, :, ::-1]   # (3,64,64)
   right_chw = right_chw[:, :, ::-1]

③ gaze x성분 부호 반전
   gaze[0] *= -1
   [0.115, -0.075, -0.989]  →  [-0.115, -0.075, -0.989]
   (MPIIGaze 좌표: x=오른쪽 방향 → 좌우 반전 시 x 부호가 바뀜)
```

> albumentations의 HorizontalFlip 미사용.  
> gaze 벡터 수정을 직접 제어하기 위해 수동 처리.

### 2-3. crop 포인트 필터 적용
```
left_hwc = left_chw.transpose(1, 2, 0)    # (64, 64, 3)  HWC 변환

apply_crop(left_hwc, cfg)
  └─ cfg.category.crop.selected 기준:
     "none"      → 그대로
     "high_pass" → 고주파 강화 (Gaussian 차분)
     "gabor"     → Gabor 텍스처 강화
     "both"      → high_pass → gabor 순차 적용
     "adaptive"  → 항등 변환 (실제 처리는 모델 내부 AdaptiveFilter)
```

### 2-4. albumentations 증강 (학습 시)
```
transforms(image=left_hwc)["image"]
  └─ Rotate(±10°)                      50%
     RandomBrightnessContrast(±20%)    50%
     GaussNoise                        30%
     CoarseDropout (소규모 마스킹)       20%

※ HorizontalFlip, Normalize, ToTensorV2 미포함
```

### 2-5. CHW 변환 + 정규화
```
(64, 64, 3) HWC  →  (3, 64, 64) CHW  →  float32

x = pixel / 255.0
x[0] = (x[0] - 0.406) / 0.225    # B 채널 (ImageNet 통계, BGR 순서)
x[1] = (x[1] - 0.456) / 0.224    # G 채널
x[2] = (x[2] - 0.485) / 0.229    # R 채널
값 범위: 대략 [-2.1, 2.6]
```

### 2-6. 반환값
```python
{
  "left"     : Tensor (3, 64, 64)  float32   # 정규화된 왼눈 이미지
  "right"    : Tensor (3, 64, 64)  float32   # 정규화된 오른눈 이미지
  "gaze"     : Tensor (3,)         float32   # [-0.115, -0.075, -0.989]
  "head_pose": Tensor (3,)         float32   # [-0.08,  0.03,  0.01]
}
```

---

## STEP 3 — 모델 순전파 (`gaze_model.py`, 배치 단위)

배치 크기 256 기준:

### 3-1. SiameseBackbone
```
left  (256, 3, 64, 64)  ─┐
                          ├─ 동일한 가중치의 CNN ─┬─ feat_l (256, 512)
right (256, 3, 64, 64)  ─┘                       └─ feat_r (256, 512)

backbone = resnet18 → feat_dim = 512
backbone = mobilenet_v2 → feat_dim = 1280
```

### 3-2. concat + Regressor
```
x = concat([feat_l, feat_r, head_pose])
  = (256, 512 + 512 + 3)  =  (256, 1027)

Regressor:
  FC(1027 → 256) → ReLU → Dropout(0.3) → FC(256 → 3)
  출력: (256, 3)
```

### 3-3. 출력 정규화
```
pred = F.normalize(out, dim=1)   # 각 행을 단위벡터로
     = (256, 3)  float32

예) 샘플 한 개: [-0.113, -0.073, -0.991]
```

---

## STEP 4 — 손실 계산 (`train.py`)

```
loss = mean(1 - cosine_similarity(pred, gaze))
     = mean(1 - dot(pred[i], gaze[i]))        # 둘 다 단위벡터이므로

예)
  pred = [-0.113, -0.073, -0.991]
  gaze = [-0.115, -0.075, -0.989]
  dot  = 0.9999
  loss = 1 - 0.9999 = 0.0001
```

**왜 MSE(pitch, yaw) 대신 cosine loss?**
- `1 - cos(θ)`는 angular error의 단조 증가 함수 → loss 최소화 = angular error 직접 최소화
- pitch/yaw 변환 없음 → 극점 불안정(arcsin 포화) 없음
- yaw 주기성 문제(-π ↔ π) 없음

---

## STEP 5 — 평가 (`eval.py`)

```
pred = [-0.113, -0.073, -0.991]   # 모델 출력 단위벡터
gt   = [-0.115, -0.075, -0.989]   # 정답 단위벡터

dot          = clip(pred · gt, -1, 1)  =  0.9999
angular_err  = degrees(arccos(0.9999)) ≈  0.8°

전체 테스트셋 평균 → test_angular_err: 4.xx°
→ result.yaml에 기록
```

---

## 전체 흐름 요약

```
0001.jpg  +  day01.mat
       │
       ▼  preprocess.py (1회)
       │
       ├─ gaze: 좌/우 평균 → 단위벡터 (3,)
       ├─ 품질 필터 (밝기 / 블러 / 시선각도)
       ├─ apply_det  → FaceLandmarker → 눈 bbox
       ├─ apply_pose → FaceLandmarker + PnP → head_pose (3,)
       └─ 원본 프레임 64×64 크롭
       │
       ▼  HDF5 저장
       │
       │  left_eye  (3,64,64) uint8
       │  right_eye (3,64,64) uint8
       │  gaze      (3,)      float32  ← 3D 단위벡터
       │  head_pose (3,)      float32
       │
       ▼  dataset.py (매 스텝)
       │
       ├─ HFlip 50%: 이미지 swap + gaze[0] *= -1
       ├─ apply_crop → crop 포인트 필터
       ├─ albumentations 증강
       └─ 정규화 → float32 텐서
       │
       ▼  GazeEstimator.forward()
       │
       ├─ SiameseBackbone: 좌/우 → feat (512,) × 2
       ├─ concat(feat_l, feat_r, head_pose) → (1027,)
       ├─ Regressor FC → (3,)
       └─ F.normalize → 단위벡터 pred (3,)
       │
       ▼  loss = 1 - dot(pred, gaze)  →  역전파  →  가중치 업데이트
       │
       ▼  eval: angular_err = degrees(arccos(dot(pred, gt)))
```
