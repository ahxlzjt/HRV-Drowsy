# common.py
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt


# ======================
# === Column schemas ===
# ======================
RR_CANDS     = ['rr_ms','rri_ms','ibi_ms','RR_ms','RR','IBI_ms','RRI','RRI_ms']
STAGE_CANDS  = ['stage','Stage','sleep_stage','SleepStage','stage_code']
TIME_CANDS   = ['timestamp_sec','time_sec','t_sec','time_s','TimeSec','sec','seconds']
VALID_STAGES = {'W','N1','N2'}  # 학습 라벨 구성 시 사용


# ==========================
# === Loading & Parsing  ===
# ==========================
def read_csv_flex(path: Path) -> pd.DataFrame:
    """다양한 열 이름을 허용하여 RR/Stage/Time을 표준화하여 반환."""
    df = pd.read_csv(path, on_bad_lines='skip')
    cols = {c.lower(): c for c in df.columns}

    def pick(cands):
        for k in cands:
            if k in cols:
                return cols[k]
        return None

    rr_col = pick(RR_CANDS)
    st_col = pick(STAGE_CANDS)  # 추론 시 없어도 됨
    tm_col = pick(TIME_CANDS)

    if rr_col is None:
        raise ValueError(f"[{path.name}] RR/IBI 열을 찾지 못함")

    rr = pd.to_numeric(df[rr_col], errors="coerce").to_numpy(dtype=float)

    stage = None
    if st_col is not None:
        st_raw = df[st_col].astype(str).to_numpy()

        def norm_stage(x):
            s = str(x).strip().upper()
            if s in ('W','0'): return 'W'
            if s in ('N1','1'): return 'N1'
            if s in ('N2','2'): return 'N2'
            if s in ('N3','3'): return 'N3'
            if s in ('REM','R','4','5'): return 'REM'
            return 'UNK'

        stage = np.array([norm_stage(x) for x in st_raw], dtype=str)

    if tm_col is not None:
        t = pd.to_numeric(df[tm_col], errors="coerce").to_numpy(dtype=float)
    else:
        # time 없으면 RR 누적합으로 근사 (ms → s)
        rr_fill = np.where(np.isfinite(rr), rr, np.nanmedian(rr))
        t = np.cumsum(rr_fill) / 1000.0

    m = np.isfinite(rr) & np.isfinite(t)
    rr = rr[m]
    t = t[m]
    if stage is not None:
        stage = stage[m]

    subj = re.sub(r'[^0-9A-Za-z]+','_', Path(path).stem)
    out = {'subject_id': subj, 'time_s': t, 'rr_ms': rr}
    if stage is not None:
        out['stage'] = stage
    return pd.DataFrame(out)


# =========================
# === HRV & Windowing   ===
# =========================
def hrv_features(times_s: np.ndarray, rr_ms: np.ndarray) -> Tuple[float,float,float,float]:
    """SDNN, RMSSD, LF/HF, HR 계산."""
    x = np.asarray(rr_ms, dtype=float)
    t = np.asarray(times_s, dtype=float)
    m = np.isfinite(x) & np.isfinite(t)
    x, t = x[m], t[m]
    if len(x) < 6 or (t[-1] - t[0]) < 30:
        return np.nan, np.nan, np.nan, np.nan

    mean_rr = float(np.nanmean(x))
    hr = 60000.0 / mean_rr if mean_rr > 0 else np.nan
    sdnn = float(np.nanstd(x, ddof=1))
    dx = np.diff(x)
    rmssd = float(np.sqrt(np.nanmean(dx * dx))) if len(dx) > 0 else np.nan

    fs = 4.0
    grid = np.arange(t[0], t[-1], 1.0 / fs) if t[-1] > t[0] else np.array([])
    if grid.size >= 8:
        xi = np.interp(grid, t, x)
        xi = xi - np.nanmean(xi)
        nperseg = min(256, len(xi))
        if nperseg >= 16:
            f, pxx = welch(xi, fs=fs, nperseg=nperseg)

            def bp(lo, hi):
                mm = (f >= lo) & (f < hi)
                return float(np.trapz(pxx[mm], f[mm])) if np.any(mm) else 0.0

            lf, hf = bp(0.04, 0.15), bp(0.15, 0.40)
            lfhf = (lf / hf) if (hf > 1e-12) else np.nan
        else:
            lfhf = np.nan
    else:
        lfhf = np.nan

    return sdnn, rmssd, lfhf, hr


def build_windows(
    df_subj: pd.DataFrame,
    win: int = 120,
    step: int = 30,
    awake_ratio: float = 0.75,
    drowsy_ratio: float = 0.75,
    label_from_stage: bool = False
) -> pd.DataFrame:
    """윈도우 단위 HRV 피처 생성. stage가 있을 때만 학습 모드에서 라벨 생성."""
    req = ['time_s', 'rr_ms']
    for c in req:
        if c not in df_subj.columns:
            raise ValueError(f"[build_windows] Missing required column: {c}")

    sid = df_subj['subject_id'].iloc[0] if 'subject_id' in df_subj.columns else 'unknown'
    t = df_subj['time_s'].to_numpy(dtype=float)
    rr = df_subj['rr_ms'].to_numpy(dtype=float)
    st = df_subj['stage'].to_numpy() if ('stage' in df_subj.columns) else None

    if len(t) == 0:
        return pd.DataFrame([])

    rows = []
    start, end = float(t[0]), float(t[-1])
    cur = start

    while cur + win <= end:
        m = (t >= cur) & (t < cur + win)
        if m.sum() >= 8:
            sdnn, rmssd, lfhf, hr = hrv_features(t[m], rr[m])

            row = {
                'subject_id': sid,
                't0': cur, 't1': cur + win, 't_center': cur + win / 2.0,
                'SDNN': sdnn, 'RMSSD': rmssd, 'LF_HF': lfhf, 'HR': hr,
            }

            if label_from_stage and (st is not None):
                s = st[m]
                rW = float(np.mean(s == 'W'))
                rN1 = float(np.mean(s == 'N1'))
                rN2 = float(np.mean(s == 'N2'))
                rN12 = rN1 + rN2
                if rW >= awake_ratio:
                    y = 0
                elif rN12 >= drowsy_ratio:
                    y = 1
                else:
                    y = -1
                row['Label'] = y

            rows.append(row)
        cur += step

    df = pd.DataFrame(rows)
    if not df.empty:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_transition_features(
    feat_df: pd.DataFrame,
    drop_unlabeled_for_training: bool = False
) -> pd.DataFrame:
    """이전 윈도우 값/델타/rolling(mean, slope) 추가."""
    if feat_df is None or len(feat_df) == 0:
        return pd.DataFrame([])

    cols = ['SDNN', 'RMSSD', 'LF_HF', 'HR']
    for c in cols + ['subject_id', 't_center']:
        if c not in feat_df.columns:
            raise ValueError(f"[make_transition_features] Missing required column: {c}")

    feat = feat_df.sort_values(['subject_id', 't_center']).reset_index(drop=True).copy()
    prev = feat.groupby('subject_id')[cols].shift(1)
    prev.columns = [f'p_{c}' for c in cols]

    curX = feat[cols].to_numpy()
    prevX = prev.to_numpy()
    Xd = curX - prevX
    Xd = pd.DataFrame(Xd, columns=[f'd_{c}' for c in cols])

    out = pd.concat([feat, prev, Xd], axis=1)

    # rolling mean/slope
    for c in cols:
        out[f'{c}_meanL3'] = out.groupby('subject_id')[c].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        out[f'{c}_slopeL3'] = out.groupby('subject_id')[c].transform(
            lambda x: x.rolling(3, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else np.nan,
                raw=False
            )
        )

    if drop_unlabeled_for_training and ('Label' in out.columns):
        out = out[out['Label'].isin([0, 1])].copy()

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def get_feature_names() -> List[str]:
    return [
        'SDNN','RMSSD','LF_HF','HR',
        'p_SDNN','p_RMSSD','p_LF_HF','p_HR',
        'd_SDNN','d_RMSSD','d_LF_HF','d_HR',
        'SDNN_meanL3','RMSSD_meanL3','LF_HF_meanL3','HR_meanL3',
        'SDNN_slopeL3','RMSSD_slopeL3','LF_HF_slopeL3','HR_slopeL3'
    ]


# ============================
# === Model IO & Training  ===
# ============================
def fit_scaler_and_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 600,
    min_samples_leaf: int = 2,
    random_state: int = 42
) -> Tuple[StandardScaler, RandomForestClassifier]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(Xs, y)
    return scaler, clf


def save_bundle(
    path: Path,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_names: List[str],
    windowing: Dict,
    postproc: Optional[Dict] = None,
    best_threshold: Optional[float] = None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "windowing": windowing,
        "postproc": (postproc or {"prob_threshold": 0.5, "m": 3, "n": 5, "cooldown_w": 2, "k_margin": 1}),
        "best_threshold": (0.5 if best_threshold is None else float(best_threshold)),
    }, path)


def load_bundle(path: Path) -> Dict:
    return joblib.load(path)


def predict_proba_from_features(bundle: Dict, feat_df: pd.DataFrame) -> np.ndarray:
    feat_names = bundle["feature_names"]
    missing = [c for c in feat_names if c not in feat_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = feat_df[feat_names].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = bundle["scaler"].transform(X)
    prob = bundle["model"].predict_proba(Xs)[:, 1]
    return prob


# =========================
# === Post-processing   ===
# =========================
def apply_m_of_n(y_bin: np.ndarray, m: int, n: int) -> np.ndarray:
    """슬라이딩 윈도우에서 최근 n개 중 m개 이상 1이면 1."""
    if n <= 1:
        return y_bin.copy()
    out = y_bin.copy()
    cnt = 0
    for i in range(len(out)):
        cnt += out[i]
        if i >= n:
            cnt -= out[i - n]
        out[i] = 1 if cnt >= m else 0
    return out


def apply_cooldown(y_bin: np.ndarray, cooldown_w: int) -> np.ndarray:
    """1 발생 후 cooldown_w 개수만큼 0으로 고정."""
    if cooldown_w <= 0:
        return y_bin
    out = y_bin.copy()
    i = 0
    while i < len(out):
        if out[i] == 1:
            for k in range(1, cooldown_w + 1):
                if i + k < len(out):
                    out[i + k] = 0
            i += cooldown_w + 1
        else:
            i += 1
    return out


# =========================
# === Visualization     ===
# =========================
def plot_inference_results(df: pd.DataFrame, subject_id: Optional[str] = None):
    """라벨이 없어도 예측만으로 플롯."""
    if df is None or len(df) == 0:
        print("Nothing to plot.")
        return

    if subject_id is None:
        subject_id = str(df['subject_id'].iloc[0])

    sd = df[df['subject_id'] == subject_id].sort_values('t_center').copy()
    if len(sd) == 0:
        print(f"No data for {subject_id}")
        return

    t_hr = sd['t_center'] / 3600.0
    has_label = ('Label' in sd.columns)

    fig_rows = 4 if has_label else 3
    fig, ax = plt.subplots(fig_rows, 1, figsize=(14, 10 if has_label else 8), sharex=True)
    if fig_rows == 3:
        ax = list(ax) + [None]

    if has_label:
        ax[0].plot(t_hr, sd['Label'], 'o-', linewidth=1.5, markersize=3)
        ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(['W', 'N1/N2'])
        ax[0].set_ylabel('Stage'); ax[0].set_title(f'{subject_id} - Sleep Onset Prediction')
        ax[0].grid(True, alpha=0.3)

    if 'prob' in sd.columns:
        idx = 1 if has_label else 0
        ax[idx].plot(t_hr, sd['prob'], linewidth=2, label='Sleep Probability')
        thr = sd['threshold'].iloc[0] if 'threshold' in sd.columns else 0.5
        ax[idx].axhline(thr, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({thr:.3f})')
        ax[idx].set_ylabel('Prob'); ax[idx].set_ylim(0, 1)
        ax[idx].legend(); ax[idx].grid(True, alpha=0.3)

    if 'y_pred' in sd.columns:
        idx = 2 if has_label else 1
        ax[idx].step(t_hr, sd['y_pred'].astype(int), where='post', linewidth=2, label='Pred (Raw)')
        ax[idx].set_ylabel('Pred (0/1)'); ax[idx].set_ylim(-0.2, 1.2)
        ax[idx].legend(); ax[idx].grid(True, alpha=0.3)

    idx = 3 if has_label else 2
    if idx < len(ax) and ax[idx] is not None:
        ax2 = ax[idx].twinx()
        if 'SDNN' in sd.columns:
            ax[idx].plot(t_hr, sd['SDNN'], 'g-', linewidth=1.5, label='SDNN')
        if 'RMSSD' in sd.columns:
            ax2.plot(t_hr, sd['RMSSD'], 'b-', linewidth=1.5, label='RMSSD')
        ax[idx].set_ylabel('SDNN', color='g')
        ax2.set_ylabel('RMSSD', color='b')
        ax[idx].legend(loc='upper left'); ax2.legend(loc='upper right')
        ax[idx].grid(True, alpha=0.3)

    # 최초 0→1 전환선
    if 'y_pred_processed' in sd.columns or 'y_pred' in sd.columns:
        pred_seq = sd['y_pred_processed'].values if 'y_pred_processed' in sd.columns else sd['y_pred'].values
        prev = np.concatenate(([0], pred_seq[:-1]))
        onset_idx = np.where((prev == 0) & (pred_seq == 1))[0]
        if len(onset_idx) > 0:
            onset_time = t_hr.iloc[onset_idx[0]]
            for a in ax:
                if a is not None:
                    a.axvline(onset_time, color='red', linestyle=':', alpha=0.8, linewidth=2)
            print(f"Predicted sleep onset: {onset_time:.2f} hours")

    plt.xlabel('Time (hours)')
    plt.tight_layout()
    plt.show()
