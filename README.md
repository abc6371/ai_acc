# 영상 기반 숫자 인식 파이프라인 (YOLO Classification + cuDNN 추론)

영상(mp4)에서 프레임 단위로 숫자를 실시간 인식하고, NVIDIA cuDNN 기반 CNN으로 최종 분류/벤치마크까지 수행하는 파이프라인입니다.

```
영상(mp4)
  → [mp4_to_pgm.py] YOLO 분류 모델로 프레임별 숫자 인식 + 연속 프레임 검증
  → 28x28 grayscale PGM 저장
  → [mnistCUDNN] cuDNN 기반 CNN으로 최종 추론
  → [benchmark.sh] 정답 라벨 대비 정확도 / 추론 속도 측정
```

## 구성 파일

| 파일 | 역할 |
|---|---|
| `mp4_to_pgm.py` | 영상 → YOLO 분류 → PGM 변환 |
| `mnistCUDNN.cpp` | cuDNN 기반 숫자 분류 추론 엔진 (NVIDIA 샘플 기반, 일부 수정) |
| `Makefile` | 최신 GPU 아키텍처 대응 빌드 설정 |
| `ExampleShell-1.sh` | 정확도/속도 벤치마크 스크립트 |

## 1. `mp4_to_pgm.py` — 프레임 검증 로직

단순히 프레임마다 YOLO 추론 결과를 그대로 쓰지 않고, 아래 로직으로 순간적 오탐지를 걸러냅니다.

- `CONF_THRESH`(0.7) 미만 confidence는 무시
- 동일 라벨이 `CONFIRM_FRAMES`(15) 프레임 연속으로 나와야 "확정"
- 확정 구간 내에서 confidence가 가장 높은 프레임만 저장
- 확정된 프레임은 grayscale 반전 후 28×28로 리사이즈하여 PGM으로 저장 (MNIST 계열 CNN 입력 규격에 맞춤)

## 2. `mnistCUDNN.cpp` — NVIDIA cuDNN 샘플 기반 수정

NVIDIA cuDNN SDK의 `mnistCUDNN` 샘플을 기반으로 하며, 아래 부분을 직접 수정/추가했습니다.

- **Deprecated API 대응**: 구버전에서 쓰이던 `cudnnGetConvolutionForwardAlgorithm`이 최신 cuDNN에서 제거되어, `cudnnFindConvolutionForwardAlgorithm`으로 알고리즘 자동 선택 로직을 재작성
- **추론 시간 측정**: `std::chrono`로 이미지별 순수 추론 시간(ms)을 측정해 출력
- **배치 평가 모드 추가**: `data/output/` 디렉토리의 PGM 파일들을 순회하며 일괄 분류 + 숫자별 분포(히스토그램) + 총 소요시간을 집계하는 evaluation 블록 추가 (원본 샘플은 고정된 5장짜리 baseline 테스트만 지원)
- **단일 이미지 추론 옵션**: `image=<파일명>` 인자로 특정 이미지 하나만 분류 가능 (`ExampleShell-1.sh`에서 사용)

> ⚠️ **라이선스 안내**: 이 파일은 NVIDIA cuDNN SDK 샘플(`cudnn_samples_v8`)을 기반으로 수정한 것으로, 원본 저작권/EULA 고지를 그대로 유지하고 있습니다. 컴파일에 필요한 `fp16_dev.h/.cu`, `fp16_emu.h/.cpp`, `gemv.h`, `error_util.h`와 `data/` 폴더의 사전 학습 가중치(`conv1.bin` 등)는 NVIDIA cuDNN SDK에 포함되어 있으므로 별도로 첨부하지 않았습니다. 빌드하려면 [NVIDIA cuDNN SDK](https://developer.nvidia.com/cudnn)를 설치한 뒤 `cudnn_samples_v8/mnistCUDNN/` 디렉토리에 이 저장소의 `mnistCUDNN.cpp`, `Makefile`을 덮어써야 합니다.

## 3. `Makefile` — 빌드 설정 수정

- `SMS := 75 80 86` — Turing/Ampere 세대(T4, RTX 30xx, A100 등) Compute Capability에 맞춰 GENCODE 설정
- `NVCC` 경로를 로컬 CUDA 설치 경로에 맞게 고정

## 4. `ExampleShell-1.sh` — 정확도/속도 벤치마크

`pgm_output/` 안의 PGM 파일들을 정답 라벨(`answers` 배열)과 비교해 정확도와 프레임당 평균 추론 시간을 집계합니다.

```bash
./ExampleShell-1.sh
```

출력 예시:
```
============= SUMMARY =============
Total Images              : [값 입력]
Correct Predictions       : [값 입력]
Average Inference / Image : [값 입력] ms
=====================================
```

## 환경

```
python >= 3.9
ultralytics
opencv-python
```

```bash
pip install ultralytics opencv-python
```

빌드: CUDA Toolkit + cuDNN SDK (Compute Capability 7.5 / 8.0 / 8.6 대응 GPU)

## 실행 순서

```bash
python mp4_to_pgm.py       # 영상 → PGM 변환
make                        # cuDNN 추론 엔진 빌드
./ExampleShell-1.sh         # 정확도/속도 벤치마크
```
