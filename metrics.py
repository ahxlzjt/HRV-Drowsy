# metrics.py
# ===========================================
# Sleep Onset 타이밍 평가 (train/test 모드 지원)
# - test: thr=0.09 기본, 후처리 미적용
# - train: best_threshold.json/모델 번들에서 임계값 해석, 후처리 적용
#   (CLI로 thr/postproc 오버라이드 가능)
# ===========================================
import os, json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib

from utils import (
    read_csv_flex, build_windows, make_transition_features,
    predict_proba_from_features, apply_m_of_n, apply_cooldown
)

# ---------- 유틸 ----------
def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace("_", "")

def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")

def _summarize_pred(prob, y_raw, y_pred, thr):
    return {
        "thr": float(thr),
        "prob_mean": float(np.mean(prob)) if len(prob) else float("nan"),
        "prob_p95": _pct(prob, 95),
        "prob_max": float(np.max(prob)) if len(prob) else float("nan"),
        "pos_before": int(np.sum(y_raw)),
        "pos_after": int(np.sum(y_pred)),
    }

# ---------- Raw CSV에서 time/stage 읽기 ----------
def _read_time_stage(csv_path: str):
    df = pd.read_csv(csv_path)

    # 대소문자/언더스코어 무시 매핑
    cols = {_norm_key(c): c for c in df.columns}

    def pick(cands):
        for k in cands:
            nk = _norm_key(k)
            if nk in cols:
                return cols[nk]
        return None

    tm_col = pick(['timestamp_sec','time_sec','t_sec','time_s','timesec','sec','seconds','timestamp','time'])
    st_col = pick(['stage','sleep_stage','sleepstage','stage_code','stg','label','sleep_stage_code'])
    if st_col is None:
        raise ValueError(f"[{csv_path}] stage 컬럼을 찾지 못함")
    if tm_col is None:
        raise ValueError(f"[{csv_path}] time 컬럼을 찾지 못함")

    t = pd.to_numeric(df[tm_col], errors="coerce").to_numpy(float)
    s_raw = df[st_col].astype(str)

    def norm_stage(x: str) -> str:
        s = str(x).strip().upper()
        if s in ('W','0'): return 'W'
        if s in ('N1','1'): return 'N1'
        if s in ('N2','2'): return 'N2'
        if s in ('N3','3'): return 'N3'
        if s in ('REM','4','5','R'): return 'REM'
        return 'UNK'

    stage = np.array([norm_stage(x) for x in s_raw], dtype=str)
    m = np.isfinite(t)
    return t[m], stage[m]

# ---------- 첫 온셋 ----------
def _first_onset_time_from_raw(time_s, stage_tokens):
    if len(stage_tokens) == 0:
        return None
    prev = np.concatenate(([stage_tokens[0]], stage_tokens[:-1]))
    idx = np.where((prev == 'W') & np.isin(stage_tokens, ['N1','N2']))[0]
    return float(time_s[idx[0]]) if len(idx) > 0 else None

def _first_onset_time_from_pred(t_center_s, pred01):
    if len(pred01) == 0:
        return None
    prev = np.concatenate(([0], pred01[:-1]))
    idx = np.where((prev == 0) & (pred01 == 1))[0]
    return float(t_center_s[idx[0]]) if len(idx) > 0 else None

# ---------- 임계값 해석 ----------
def _resolve_threshold(csv_name, model_bundle, mode, thr_override, best_thr_path, prefer="auto"):
    """
    prefer: 'auto' | 'f1' | 'youden' | 'global'
    우선순위:
      1) thr_override
      2) (test) 0.09 고정
      3) best_thr_path:
         - 'best_threshold' (전역)
         - 'per_file'[csv_name]
         - prefer에 따른 'best_f1.thr' 또는 'best_youden.thr'
         - 'table' 리스트/테이블에서 file 매칭
      4) model_bundle['best_threshold']
      5) 0.09 (안전 폴백)
    """
    # 1) CLI 최우선
    if thr_override is not None:
        return float(thr_override)

    # 2) test 모드 고정
    if mode == "test":
        return 0.09

    # 3) train 모드: JSON에서 최대한 유연하게
    thr = None
    obj = None
    p = Path(best_thr_path) if best_thr_path else None
    if p and p.exists() and p.is_file():
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except Exception:
            obj = None

    # per-file 매핑, 전역, best_f1/youden, table 지원
    if isinstance(obj, dict):
        # per_file 우선 시도 (파일별 키가 있는 경우)
        if thr is None and isinstance(obj.get("per_file"), dict):
            v = obj["per_file"].get(csv_name)
            if isinstance(v, (int, float)):
                thr = float(v)

        # 전역 best_threshold
        if thr is None and isinstance(obj.get("best_threshold"), (int, float)):
            thr = float(obj["best_threshold"])

        # prefer에 따른 선택 (auto는 f1 우선, 없으면 youden)
        def pick_from_metric(metric_key):
            v = obj.get(metric_key)
            if isinstance(v, dict):
                t = v.get("thr", None)
                if isinstance(t, (int, float)):
                    return float(t)
            return None

        if thr is None:
            if prefer in ("auto", "f1"):
                thr = pick_from_metric("best_f1")
        if thr is None:
            if prefer in ("auto", "youden"):
                thr = pick_from_metric("best_youden")

        # table(list/dict)에서 file 매칭
        if thr is None and "table" in obj:
            try:
                df = pd.DataFrame(obj["table"])
                row = df.loc[df["file"] == csv_name]
                # 우선 best_threshold, 없으면 thr
                for c in ["best_threshold", "thr"]:
                    if c in df.columns and len(row) > 0:
                        val = row[c].iloc[0]
                        if isinstance(val, (int, float)):
                            thr = float(val)
                            break
            except Exception:
                pass

    elif isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
            if "file" in df.columns:
                row = df.loc[df["file"] == csv_name]
                for c in ["best_threshold", "thr"]:
                    if c in df.columns and len(row) > 0:
                        val = row[c].iloc[0]
                        if isinstance(val, (int, float)):
                            thr = float(val)
                            break
        except Exception:
            pass

    # 4) 모델 번들
    if thr is None:
        bt = model_bundle.get("best_threshold", None)
        if isinstance(bt, (int, float)):
            thr = float(bt)

    # 5) 안전 폴백
    if thr is None:
        thr = 0.09

    return thr

# ---------- 단일 CSV 평가 ----------
def _eval_one_csv(model_bundle, csv_path, mode, thr_override, best_thr_path,
                  post_m_override=None, post_n_override=None, cooldown_override=None, debug=False, prefer="auto"):
    df = read_csv_flex(csv_path)
    win = int(model_bundle["windowing"]["win"])
    step = int(model_bundle["windowing"]["step"])

    feat_win = build_windows(df, win=win, step=step, label_from_stage=False)
    feat = make_transition_features(feat_win, drop_unlabeled_for_training=False)
    if feat.empty:
        raise RuntimeError(f"[{csv_path.name}] 윈도우 피처 생성 실패")

    # threshold 결정 (안전화)
    thr = _resolve_threshold(csv_path.name, model_bundle, mode, thr_override, best_thr_path, prefer=prefer)

    prob = predict_proba_from_features(model_bundle, feat)
    y_raw = (prob >= thr).astype(int)

    # postproc (train만 적용)
    if mode == "train":
        post = model_bundle.get("postproc", {})
        m = int(post.get("m", 3))
        n = int(post.get("n", 5))
        cooldown_w = int(post.get("cooldown_w", 2))
        if post_m_override is not None: m = int(post_m_override)
        if post_n_override is not None: n = int(post_n_override)
        if cooldown_override is not None: cooldown_w = int(cooldown_override)
        y_proc = apply_m_of_n(y_raw, m, n)
        y_proc = apply_cooldown(y_proc, cooldown_w)
        y_pred = y_proc
    else:
        y_pred = y_raw

    t_pred = feat["t_center"].to_numpy(float)

    # 디버그 출력
    if debug:
        info = _summarize_pred(prob, y_raw, y_pred, thr)
        if mode == "train":
            print(f"[{csv_path.name}] thr={info['thr']:.4f}, prob(mean/p95/max)={info['prob_mean']:.4f}/{info['prob_p95']:.4f}/{info['prob_max']:.4f}, "
                  f"pos(before/after)={info['pos_before']}/{info['pos_after']}")
        else:
            print(f"[{csv_path.name}] thr={info['thr']:.4f}, prob(mean/p95/max)={info['prob_mean']:.4f}/{info['prob_p95']:.4f}/{info['prob_max']:.4f}, "
                  f"pos={info['pos_before']}")

    # GT & Pred onset
    t_raw, st_raw = _read_time_stage(str(csv_path))
    gt_onset = _first_onset_time_from_raw(t_raw, st_raw)
    pr_onset = _first_onset_time_from_pred(t_pred, y_pred)

    return {
        "file": csv_path.name,
        "thr": thr,
        "gt_onset_min": (None if gt_onset is None else gt_onset/60.0),
        "pred_onset_min": (None if pr_onset is None else pr_onset/60.0),
        "delta_min": (None if (gt_onset is None or pr_onset is None) else (pr_onset-gt_onset)/60.0),
    }

# ---------- 전체 평가 ----------
def evaluate_onset(input_path, model_path, mode="test",
                   tol_minutes=10, thr_override=None,
                   glob_pattern="*.csv", save_table=None,
                   best_thr_path="outputs/best_threshold.json",
                   post_m=None, post_n=None, cooldown_w=None, debug=False, prefer="auto"):
    bundle = joblib.load(model_path)
    p = Path(input_path)

    if p.is_dir():
        files = sorted(Path(p).glob(glob_pattern))
    elif p.is_file() and p.suffix.lower()==".csv":
        files = [p]
    else:
        raise FileNotFoundError(f"입력 경로 오류: {input_path}")

    rows=[]
    for fp in files:
        try:
            r = _eval_one_csv(bundle, fp, mode, thr_override, best_thr_path,
                              post_m_override=post_m, post_n_override=post_n, cooldown_override=cooldown_w,
                              debug=debug, prefer=prefer)
            rows.append(r)
        except Exception as e:
            print(f"[스킵] {fp.name}: {e}")

    if not rows:
        raise RuntimeError("유효한 평가 결과 없음")

    tab = pd.DataFrame(rows)

    valid = tab["delta_min"].dropna().to_numpy(float)
    n_valid = len(valid); n_total = len(tab)

    abs_err = np.abs(valid)
    mae   = float(np.mean(abs_err)) if n_valid else np.nan
    medae = float(np.median(abs_err)) if n_valid else np.nan
    rmse  = float(np.sqrt(np.mean(valid**2))) if n_valid else np.nan
    bias  = float(np.mean(valid)) if n_valid else np.nan
    succ5   = float(np.mean(abs_err<=5)) if n_valid else np.nan
    succ10  = float(np.mean(abs_err<=10)) if n_valid else np.nan
    succ15  = float(np.mean(abs_err<=15)) if n_valid else np.nan
    succTol = float(np.mean(abs_err<=tol_minutes)) if n_valid else np.nan

    print(f"\n===== Sleep Onset Timing ({mode}) =====")
    print(f"Valid onset pairs: {n_valid}/{n_total}")
    print(f"MAE (min):        {mae:.2f}")
    print(f"Median AE (min):  {medae:.2f}")
    print(f"RMSE (min):       {rmse:.2f}")
    print(f"Mean Bias (min):  {bias:.2f}")
    print(f"Success@5  min:   {succ5*100:5.1f}%")
    print(f"Success@10 min:   {succ10*100:5.1f}%")
    print(f"Success@15 min:   {succ15*100:5.1f}%")
    print(f"Success@{tol_minutes} min: {succTol*100:5.1f}%")

    if save_table:
        Path(save_table).parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(save_table, index=False)
        print(f"[저장] {save_table}")

    return {
        "n_valid": n_valid, "n_total": n_total,
        "MAE_min": mae, "MedAE_min": medae, "RMSE_min": rmse, "MeanBias_min": bias,
        "Success@5": succ5, "Success@10": succ10, "Success@15": succ15, f"Success@{tol_minutes}": succTol
    }, tab

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/model.pkl")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["train","test"], default="test")
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--glob", type=str, default="*.csv")
    ap.add_argument("--save_table", type=str, default=None)
    ap.add_argument("--best_thr_path", type=str, default="outputs/best_threshold.json")
    ap.add_argument("--post_m", type=int, default=None, help="train 모드 후처리 m 오버라이드")
    ap.add_argument("--post_n", type=int, default=None, help="train 모드 후처리 n 오버라이드")
    ap.add_argument("--cooldown_w", type=int, default=None, help="train 모드 cooldown window 오버라이드")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--prefer", type=str, choices=["auto","f1","youden","global"], default="auto",
                    help="JSON에 여러 후보가 있을 때 선택 우선순위 (auto=f1 우선→youden)")
    args = ap.parse_args()

    evaluate_onset(
        args.input, args.model_path,
        mode=args.mode, tol_minutes=args.tol,
        thr_override=args.thr, glob_pattern=args.glob,
        save_table=args.save_table, best_thr_path=args.best_thr_path,
        post_m=args.post_m, post_n=args.post_n, cooldown_w=args.cooldown_w,
        debug=args.debug, prefer=args.prefer
    )

if __name__ == "__main__":
    main()
