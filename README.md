# HRV-Drowsy  
**Heart Rate Variability 기반 졸음 상태 감지 (Sleep Onset Detection)**  

이 프로젝트는 **RRI/SDNN 기반 HRV 시계열 데이터**를 활용하여 **각성(Wake) → 졸음 단계(N1/N2)** 전이를 감지하는 모델을 제공합니다.  
수집된 RR 간격 데이터를 기반으로 HRV 피처를 추출하고, **Random Forest Classifier**를 이용해 수면 온셋(첫 졸음 전환)을 탐지합니다.  

주요 목표는 **실시간 추론**과 **온디바이스 적용(CoreML, ONNX 변환)** 입니다.  

---

## 📂 프로젝트 구조

```
hrv-drowsy/
├─ data/                    # 원본/학습/테스트 데이터
│  ├─ train/               # 학습용 CSV
│  └─ test/                # 추론용 CSV
├─ models/
│  └─ model.pkl            # 학습된 모델
├─ outputs/                # 추론 결과(csv, plot 저장)
├─ utils.py                # 공통 유틸 (데이터 로딩, HRV feature, 후처리, 시각화)
├─ train.py                # 모델 학습 스크립트
├─ test.py                 # 추론/평가 스크립트
├─ requirements.txt        # 의존성 목록 (버전 고정)
├─ .gitignore              # data/models/outputs/ 제외
└─ README.md
```

---

## 📊 데이터셋

본 프로젝트는 **[Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)** 공개 데이터셋을 기반으로 합니다.  

- **데이터 위치**  
  - `polysomnography/annotations-events-profusion/`  
- **필수 컬럼**  
  - `time_s` : 시간 (초 단위)  
  - `rr_ms` : RR 간격(ms)  
  - (학습 시 필요) `stage` : 수면 단계 (`W`, `N1`, `N2`, `N3`, `REM`)  

### 데이터 다운로드
1. [NSRR (National Sleep Research Resource)](https://sleepdata.org/datasets/shhs) 회원가입 및 **데이터 접근 승인** 필요  
2. 다운받은 CSV를 `data/train/`, `data/test/` 에 배치

### 인용 (Citation)
- Zhang GQ, et al. The National Sleep Research Resource: towards a sleep data commons. JAMIA. 2018;25(10):1351–1358. https://doi.org/10.1093/jamia/ocy064
- Quan SF, et al. The Sleep Heart Health Study: design, rationale, and methods. Sleep. 1997;20(12):1077–1085. https://doi.org/10.1093/sleep/20.12.1077

### Acknowledgements
This work uses data from the Sleep Heart Health Study (SHHS), available via the National Sleep Research Resource (NSRR).
Please include the following Acknowledgement text exactly as shown:

"The Sleep Heart Health Study (SHHS) was supported by National Heart, Lung, and Blood Institute cooperative agreements U01HL53916 (University of California, Davis), U01HL53931 (New York University), U01HL53934 (University of Minnesota), U01HL53937 and U01HL64360 (Johns Hopkins University), U01HL53938 (University of Arizona), U01HL53940 (University of Washington), U01HL53941 (Boston University), and U01HL63463 (Case Western Reserve University). The National Sleep Research Resource was supported by the National Heart, Lung, and Blood Institute (R24 HL114473, 75N92019R002)."

---

## ⚙️ 설치

```bash
# 가상환경 생성
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### requirements.txt

```ini
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.4.2
matplotlib==3.8.4
joblib==1.3.2
```

---

## 🚀 실행 방법

### 1) 학습 (Training)

```bash
python train.py --data_dir data/train --cv
```

**옵션:**
- `--win 120` : 윈도우 크기(초 단위)
- `--step 10` : 스트라이드(초 단위)
- `--awake_ratio 0.75` : W 비율 ≥ 값 → Label=0
- `--drowsy_ratio 0.75` : (N1+N2) 비율 ≥ 값 → Label=1
- `--cv` : GroupKFold 기반 threshold 탐색 실행

**결과:**
학습된 모델 번들이 `models/model.pkl` 로 저장됩니다.

### 2) 추론 (Inference)

**단일 파일 추론**

```bash
python test.py --model_path models/model.pkl \
  --input data/test/sample.csv \
  --out_dir outputs --plot --thr 0.09 \
  --gt data/test/sample.csv
```

**폴더 전체 추론**

```bash
python test.py --model_path models/model.pkl \
  --input data/test --out_dir outputs --plot --thr 0.09
```

**옵션:**
- `--thr 0.09` : 수면 확률 임계값 지정 (기본 0.5)
- `--gt file.csv` : stage 컬럼 포함된 원본 CSV와 GT 비교 플롯
- `--no-post` : 후처리 미적용 (원시 예측 사용)
- `--plot` : 결과 시각화 표시

**출력:**
- `outputs/<파일명>_pred.csv` : 확률/예측/후처리 결과 저장
- 플롯: 확률, 예측, HRV feature, (옵션) GT hypnogram 표시

---

## 🧩 모델 개요

### Features
- **기본**: SDNN, RMSSD, LF/HF, HR
- **이전 윈도우 값**: p_XXX
- **변화량**: d_XXX
- **Rolling mean/slope** (L=3)

### Labels
- W 비율 ≥ 0.75 → Label=0 (Awake)
- N1+N2 비율 ≥ 0.75 → Label=1 (Drowsy)
- 그 외 → Label=-1 (Uncertain)

### Target
(prev Label==0) & (curr Label==1) → Sleep Onset Transition

### Classifier
RandomForestClassifier (n_estimators=600, class_weight="balanced_subsample")

### Post-processing
- m-of-n smoothing
- cooldown window 적용

---

## 📈 플롯

- Sleep Probability + Threshold
- Raw Prediction (0/1)
- HRV Features (SDNN, RMSSD)
- (옵션) GT hypnogram 비교

---

## 📜 License

MIT License  
단, SHHS 데이터는 별도 라이선스를 따릅니다.
