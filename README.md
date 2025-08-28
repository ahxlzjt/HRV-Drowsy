# HRV-Drowsy  
**Heart Rate Variability-based Sleep Onset Detection**  

This project provides a model for detecting **Wake → Sleep stage (N1/N2)** transitions using **RRI/SDNN-based HRV time series data**.  
The system extracts HRV features from collected RR interval data and uses a **Random Forest Classifier** to detect sleep onset (first transition to sleep).

---

## 📂 Project Structure

```
hrv-drowsy/
├─ data/                    # Raw/training/test data
│  ├─ train/               # Training CSV files
│  └─ test/                # Inference CSV files
├─ models/
│  └─ model.pkl            # Trained model
├─ outputs/                # Inference results (csv, plots, evaluation results)
├─ realtime/               # Real-time processing
│  └─ send_trigger_from_stream.py  # TCP trigger transmission client
├─ utils.py                # Common utilities (data loading, HRV features, post-processing, visualization)
├─ train.py                # Model training script
├─ test.py                 # Inference/evaluation script
├─ metrics.py              # Sleep onset timing quantitative evaluation (MAE/RMSE/success rate)
├─ requirements.txt        # Dependencies
├─ .gitignore              
└─ README.md
```

---

## 📊 Dataset

This project is based on the **[Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)** public dataset.  

- **Data Location**  
  - `polysomnography/annotations-events-profusion/`  
- **Required Columns**  
  - `time_s` : Time (seconds) - if missing, approximated from RR cumulative sum
  - `rr_ms` : RR intervals (ms) - automatically recognizes various names (`rri_ms`, `ibi_ms`, `RR_ms`, etc.)
  - (Required for training) `stage` : Sleep stages (`W`, `N1`, `N2`, `N3`, `REM`)  

### Data Download
1. Register at [NSRR (National Sleep Research Resource)](https://sleepdata.org/datasets/shhs) and obtain **data access approval**  
2. Place downloaded CSV files in `data/train/`, `data/test/`

### Acknowledgements
This work uses data from the Sleep Heart Health Study (SHHS), available via the National Sleep Research Resource (NSRR).
Please include the following Acknowledgement text exactly as shown:

"The Sleep Heart Health Study (SHHS) was supported by National Heart, Lung, and Blood Institute cooperative agreements U01HL53916 (University of California, Davis), U01HL53931 (New York University), U01HL53934 (University of Minnesota), U01HL53937 and U01HL64360 (Johns Hopkins University), U01HL53938 (University of Arizona), U01HL53940 (University of Washington), U01HL53941 (Boston University), and U01HL63463 (Case Western Reserve University). The National Sleep Research Resource was supported by the National Heart, Lung, and Blood Institute (R24 HL114473, 75N92019R002)."

---

## ⚙️ Installation

```bash
# Create virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
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

## 🚀 Execution Methods

### 1) Training

```bash
python train.py --data_dir data/train --cv
```

**Main Options:**
- `--win 120` : Window length (seconds)
- `--step 10` : Stride (seconds)
- `--awake_ratio 0.75` : W ratio threshold within window
- `--drowsy_ratio 0.75` : (N1+N2) ratio threshold within window
- `--cv` : Window F1-based threshold search using GroupKFold

**Output:**
- `models/model.pkl`
  - Includes: `model(RandomForest), scaler, feature_names, windowing, postproc, best_threshold`

**Note:** The `fit_scaler_and_model` import path in `train.py` should be unified to `utils`. Example: `from utils import fit_scaler_and_model`

### 2) Inference/Visualization

**Single File:**

```bash
python test.py \
  --model_path models/model.pkl \
  --input data/test/sample.csv \
  --out_dir outputs \
  --plot \
  --gt data/test/sample.csv        # GT comparison (requires stage column in original CSV)
```

**Entire Folder:**

```bash
python test.py \
  --model_path models/model.pkl \
  --input data/test \
  --out_dir outputs \
  --plot
```

**Options:**
- Threshold: Default uses bundle (`best_threshold` or `postproc.prob_threshold`), can be overridden with `--thr`
- `--no-post` : Skip post-processing for GT comparison (use raw predictions)

**Output:**
- `outputs/<filename>_pred.csv` (prob, y_pred, y_pred_processed, threshold, etc.), plots (optional)

### 3) Metrics

`metrics.py` provides two types of evaluation:
- **Window-level Confusion Metrics**: Window-unit binary classification performance (Accuracy/Precision/Recall/F1)
  - `--mode train`: applies threshold/m-of-n/cooldown post-processing stored in model bundle
  - `--mode test`: uses fixed threshold and **no post-processing** by default (can be overridden)
- **Sleep Onset Timing (Test Only)**: **Onset timing error** close to real usage (MAE, Median AE, RMSE, Mean Bias, Success@K)

#### 3.1 Window-level Confusion Metrics

**Train (Apply optimal parameters after CV):**

```bash
python metrics.py \
  --model_path models/model.pkl \
  --input data/train \
  --mode train \
  --glob "*.csv" \
  --save_table outputs/train_eval.csv
```

Output:
```
===== Window Confusion Matrix (train) =====
Accuracy:         89.7%
Precision:        88.4%
Recall:           86.1%
F1-score:         87.2%
[Saved] outputs/train_eval.csv
```

**Test (Default: fixed thr, no post-processing):**

```bash
python metrics.py \
  --model_path models/model.pkl \
  --input data/test \
  --mode test \
  --glob "*.csv" \
  --save_table outputs/test_eval.csv
```

Output:
```
===== Window Confusion Matrix (test) =====
Accuracy:         86.3%
Precision:        84.7%
Recall:           82.9%
F1-score:         83.8%
[Saved] outputs/test_eval.csv
```

If needed, test can also override threshold/post-processing with `--thr --post_m --post_n --cooldown_w` for comparison on same criteria.

#### 3.2 Sleep Onset Timing (Test Only)

Batch sleep onset timing error evaluation.

```bash
python metrics.py \
  --model_path models/model.pkl \
  --input data/test \
  --mode test \
  --glob "*.csv" \
  --tol 10 \
  --save_table outputs/onset_eval.csv \
  --debug
```

**Metrics:**
- MAE, Median AE, RMSE, Mean Bias
- Success@5/10/15/`tol` (minutes)

Output:
```
===== Sleep Onset Timing (test) =====
Valid onset pairs: 22/22
MAE (min):        8.48
Median AE (min):  7.58
RMSE (min):       10.92
Mean Bias (min):  -0.78   # + late prediction, - early prediction
Success@5  min:    31.8%
Success@10 min:    68.2%
Success@15 min:    90.9%
[Saved] outputs/onset_eval.csv
```

### 4) Real-time Trigger Transmission (Offline Stream → TCP)

Calculate window-wise probabilities from offline CSV and transmit `"1\n"` via TCP only on **rising edges (0→1)**.

**Transmission (Client):**

```bash
python realtime/send_trigger_from_stream.py \
  --csv data/test/sample.csv \
  --model models/model.pkl \
  --thr_json outputs/best_threshold.json \
  --win 120 --step 30 \
  --m 2 --n 5 \
  --rt_factor 10.0 \
  --host <RASPBERRY_PI_IP> --port 5055
```

**Operation:**
- Binary stream where `prob ≥ thr` is stabilized with m-of-n, transmits `"1\n"` on **0→1** transitions
- Transmission interval: `step / rt_factor` seconds
- Threshold: `--thr` > `--thr_json` > (if not provided, bundle `best_threshold` or default)

**Reception (Raspberry Pi):**

```python
import socket
s = socket.socket()
s.bind(("0.0.0.0", 5055)); s.listen(1)
conn, _ = s.accept()
while True:
    data = conn.recv(16)
    if not data: break
    print(data.decode().strip())
```

---

## 🧩 Model/Feature Design

### Features
- **Basic**: `SDNN, RMSSD, LF_HF, HR`
- **Previous values**: `p_XXX`
- **Differences**: `d_XXX`
- **Rolling**: `XXX_meanL3`, `XXX_slopeL3` (L=3)

### Labeling (Training Windows)
- W ratio ≥ `awake_ratio` → `Label = 0`
- (N1+N2) ratio ≥ `drowsy_ratio` → `Label = 1`
- Others → `Label = -1` (excluded from training)

### Target (Transition Events)
- `(prev Label==0) & (curr Label==1)` → Sleep onset transition

### Classifier
- `RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample")`

### Post-processing
- `m-of-n` → Additional `cooldown` can be applied if needed (see evaluation script)

---

## 📈 Visualization

- Sleep Probability + Threshold
- Raw/Processed Prediction (0/1)
- HRV Features (SDNN, RMSSD)
- (Optional) GT hypnogram comparison
- Sleep onset time indication (vertical dotted line)

---

## 📜 License

- **Code**: MIT License
- **Data**: Follow respective dataset licenses and citation requirements
