# test.py
import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    read_csv_flex, build_windows, make_transition_features,
    get_feature_names, load_bundle, predict_proba_from_features,
    apply_m_of_n, apply_cooldown, plot_inference_results
)

def _stage_tokens(stage_series: pd.Series) -> np.ndarray:
    s = stage_series.copy()
    # 숫자 인코딩(0=W,1=N1,2=N2,3=N3,4/5=REM) 자동 감지
    try:
        sn = pd.to_numeric(s, errors='coerce')
        if sn.notna().mean() > 0.7:
            out=[]
            for v in sn:
                if pd.isna(v): out.append('UNK')
                elif v == 0:   out.append('W')
                elif v == 1:   out.append('N1')
                elif v == 2:   out.append('N2')
                elif v == 3:   out.append('N3')
                elif v in (4,5): out.append('REM')
                else: out.append('UNK')
            return np.array(out, dtype=str)
    except Exception:
        pass
    s = s.astype(str).str.strip().str.upper()
    s = s.replace({'S1':'N1','S2':'N2','S3':'N3','NREM1':'N1','NREM2':'N2','NREM3':'N3','R':'REM'})
    allowed = {'W','N1','N2','N3','REM'}
    return np.array([x if x in allowed else 'UNK' for x in s], dtype=str)

def _first_onset_time_from_raw(time_h: np.ndarray, stage_tokens: np.ndarray):
    """GT: W -> (N1|N2) 최초 전이 시간(h). 없으면 None."""
    if len(stage_tokens) == 0:
        return None
    prev = np.concatenate(([stage_tokens[0]], stage_tokens[:-1]))
    idx = np.where((prev == 'W') & np.isin(stage_tokens, ['N1','N2']))[0]
    return (time_h[idx[0]] if len(idx) > 0 else None)

def _first_onset_time_from_pred(time_h: np.ndarray, pred01: np.ndarray):
    """Pred: 0 -> 1 최초 전이 시간(h). 없으면 None."""
    if len(pred01) == 0:
        return None
    prev = np.concatenate(([0], pred01[:-1]))
    idx = np.where((prev == 0) & (pred01 == 1))[0]
    return (time_h[idx[0]] if len(idx) > 0 else None)

def plot_gt_vs_pred_from_raw(raw_csv_path: str, pred_df: pd.DataFrame, use_postprocessed: bool=True, title_prefix="[GT(raw) vs Pred]"):
    """raw_csv_path: time_s, stage 포함 원본 CSV 경로"""
    raw = pd.read_csv(raw_csv_path)
    for c in ['time_s','stage']:
        if c not in raw.columns:
            raise ValueError(f"[GT 비교] '{c}' 컬럼이 raw CSV에 필요합니다: {raw_csv_path}")

    # 시간/스테이지 준비
    gt_t_h = raw['time_s'].to_numpy(float) / 3600.0
    gt_st  = _stage_tokens(raw['stage'])

    pred = pred_df.sort_values('t_center').copy()
    pr_t_h = pred['t_center'].to_numpy(float) / 3600.0
    prob   = pred['prob'].to_numpy(float)
    thr    = float(pred['threshold'].iloc[0]) if 'threshold' in pred.columns else 0.5
    y_pred = (pred['y_pred_processed'] if (use_postprocessed and 'y_pred_processed' in pred.columns) else pred['y_pred']).to_numpy(int)

    # 온셋 계산
    gt_onset = _first_onset_time_from_raw(pr_t_h if len(gt_t_h)==0 else gt_t_h, gt_st)
    pr_onset = _first_onset_time_from_pred(pr_t_h, y_pred)

    # 플롯
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    # (1) GT hypnogram
    stage_levels = ['W','N1','N2','N3','REM']
    st_map = {s:i for i,s in enumerate(stage_levels)}
    y = np.array([st_map.get(s, -1) for s in gt_st], dtype=float)
    ax[0].step(gt_t_h, y, where='post', linewidth=2, label='GT stage (raw)')
    ax[0].set_yticks(range(len(stage_levels))); ax[0].set_yticklabels(stage_levels)
    ax[0].set_ylim(-0.5, len(stage_levels)-0.5)
    if gt_onset is not None:
        ax[0].axvline(gt_onset, linestyle=':', linewidth=2, alpha=0.8, color='red')
    ax[0].set_ylabel('Stage'); ax[0].grid(True, alpha=0.3); ax[0].legend(loc='upper right')

    # (2) Prob + Thr
    ax[1].plot(pr_t_h, prob, linewidth=2, label='Prob')
    ax[1].axhline(thr, linestyle='--', alpha=0.8, label=f"Thr={thr:.3f}", color='red')
    ax[1].set_ylim(0,1); ax[1].set_ylabel('Prob')
    ax[1].grid(True, alpha=0.3); ax[1].legend(loc='upper right')

    # (3) Pred 0/1
    ax[2].step(pr_t_h, y_pred.astype(int), where='post', linewidth=2, label='Pred (0/1)')
    if pr_onset is not None:
        ax[2].axvline(pr_onset, linestyle=':', linewidth=2, alpha=0.8, color='red')
    ax[2].set_yticks([0,1]); ax[2].set_yticklabels(['W','Sleep'])
    ax[2].set_ylim(-0.2,1.2); ax[2].set_xlabel('Time (hours)'); ax[2].set_ylabel('Pred')
    ax[2].grid(True, alpha=0.3); ax[2].legend(loc='upper right')

    plt.suptitle(f"{title_prefix}", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    # 온셋 오차 출력
    if (gt_onset is not None) and (pr_onset is not None):
        print(f"Onset error = {abs(pr_onset - gt_onset)*60:.1f} min  (Pred {pr_onset:.2f}h vs GT {gt_onset:.2f}h)")
    elif gt_onset is None:
        print("No GT onset found in raw hypnogram.")
    elif pr_onset is None:
        print("No Pred onset found in prediction sequence.")

def infer_one_csv(bundle, csv_path: Path, out_dir: Path, plot: bool = True,
                  override_thr: float | None = None, gt_csv: str | None = None,
                  use_postprocessed: bool = True) -> Path:
    df = read_csv_flex(csv_path)

    win = bundle["windowing"]["win"]
    step = bundle["windowing"]["step"]
    feat_win = build_windows(df, win=win, step=step, label_from_stage=False)
    feat = make_transition_features(feat_win, drop_unlabeled_for_training=False)

    if feat.empty:
        raise RuntimeError(f"No features generated for {csv_path.name}")

    feat_names = bundle["feature_names"]
    missing = [c for c in feat_names if c not in feat.columns]
    if missing:
        raise KeyError(f"Missing features for inference: {missing}")

    # Prob & Binary
    prob = predict_proba_from_features(bundle, feat)
    post = bundle.get("postproc", {})
    thr = (override_thr if override_thr is not None
           else float(bundle.get("best_threshold", post.get("prob_threshold", 0.5))))
    y_raw = (prob >= thr).astype(int)

    # Post-processing
    post = bundle.get("postproc", {})
    m = int(post.get("m", 3)); n = int(post.get("n", 5)); cooldown_w = int(post.get("cooldown_w", 2))
    y_proc = apply_m_of_n(y_raw, m=m, n=n)
    y_proc = apply_cooldown(y_proc, cooldown_w=cooldown_w)

    # Assemble
    feat['prob'] = prob
    feat['y_pred'] = y_raw
    feat['y_pred_processed'] = y_proc
    feat['threshold'] = thr

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{csv_path.stem}_pred.csv"
    feat.to_csv(out_csv, index=False)

    print(f"[저장] {out_csv.name}  (thr={thr:.3f}, m={m}, n={n}, cooldown={cooldown_w})")

    if plot:
        if gt_csv is not None:
            plot_gt_vs_pred_from_raw(gt_csv, feat, use_postprocessed=use_postprocessed,
                                     title_prefix=f"[GT vs Pred] {feat['subject_id'].iloc[0]}")
        else:
            plot_inference_results(feat)

    return out_csv


def main(args):
    bundle = load_bundle(Path(args.model_path))
    in_path = Path(args.input)
    override_thr = args.thr
    gt_path = args.gt
    use_postprocessed = (not args.no_post)

    if in_path.is_dir():
        if gt_path is not None:
            print("[경고] --gt는 단일 파일 추론에서만 사용 권장(디렉토리 입력시 무시).")
        files = sorted(glob.glob(str(in_path / "*.csv")))
        if not files:
            raise SystemExit(f"CSV가 없습니다: {in_path}")
        for fp in files:
            infer_one_csv(bundle, Path(fp), Path(args.out_dir), plot=args.plot,
                          override_thr=override_thr, gt_csv=None,  # 디렉토리 모드에서는 GT 비교 생략
                          use_postprocessed=use_postprocessed)
    else:
        infer_one_csv(bundle, in_path, Path(args.out_dir), plot=args.plot,
                      override_thr=override_thr, gt_csv=gt_path,
                      use_postprocessed=use_postprocessed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/model.pkl")
    ap.add_argument("--input", type=str, default="data/test")   # CSV 파일 또는 디렉토리
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--plot", action="store_true", help="시각화 표시")
    ap.add_argument("--thr", type=float, default=None, help="Override probability threshold (e.g., 0.09)")
    ap.add_argument("--gt", type=str, default=None, help="Raw CSV with 'time_s' and 'stage' to plot GT vs Pred")
    ap.add_argument("--no-post", action="store_true", help="Use raw y_pred instead of postprocessed for GT comparison")
    args = ap.parse_args()
    main(args)
