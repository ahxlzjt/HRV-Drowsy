# train.py
import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from utils import (
    read_csv_flex, build_windows, make_transition_features,
    get_feature_names, fit_scaler_and_model, save_bundle
)

def main(args):
    data_dir = Path(args.data_dir)
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        raise SystemExit(f"CSV가 없습니다: {data_dir}")

    # 1) 윈도우 및 라벨 생성
    win, step = args.win, args.step
    parts = []
    for fp in files:
        try:
            df = read_csv_flex(Path(fp))
            w = build_windows(
                df, win=win, step=step,
                awake_ratio=args.awake_ratio, drowsy_ratio=args.drowsy_ratio,
                label_from_stage=True  # 학습 시에만 라벨 생성
            )
            if not w.empty:
                parts.append(w)
        except Exception as e:
            print(f"[스킵] {Path(fp).name}: {e}")

    if not parts:
        raise SystemExit("유효한 윈도우가 없습니다.")
    win_df = pd.concat(parts, ignore_index=True)

    # 2) 전이 피처 생성 (+ 학습 행만 유지)
    feat_df = make_transition_features(win_df, drop_unlabeled_for_training=True)

    # 3) 전이 라벨: prev Label==0 & curr Label==1
    prev_lab = feat_df.groupby('subject_id')['Label'].shift(1)
    y_tr = ((prev_lab == 0) & (feat_df['Label'] == 1)).fillna(False).astype(int).to_numpy()

    feature_names = get_feature_names()
    missing = [c for c in feature_names if c not in feat_df.columns]
    if missing:
        raise KeyError(f"학습에 필요한 피처 누락: {missing}")

    X = feat_df[feature_names].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    groups = feat_df['subject_id'].to_numpy()

    print(f"[전이 샘플] {len(y_tr)} (양성 비율={y_tr.mean():.3f})")
    if y_tr.sum() == 0:
        print("경고: 양성(전이) 라벨이 없습니다. 임계값은 0.5로 고정됩니다.")

    # 4) 임계값 선택(선택): GroupKFold로 윈도우 F1 기준
    best_thr = 0.5
    if args.cv and y_tr.sum() > 0 and len(np.unique(groups)) >= 2:
        gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
        probs_all, y_all = [], []
        for tr_idx, va_idx in gkf.split(X, y_tr, groups):
            from utils import fit_scaler_and_model
            scaler, clf = fit_scaler_and_model(
                X[tr_idx], y_tr[tr_idx],
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state
            )
            from sklearn.preprocessing import StandardScaler
            Xv = scaler.transform(X[va_idx])
            pv = clf.predict_proba(Xv)[:, 1]
            probs_all.append(pv); y_all.append(y_tr[va_idx])
        probs_all = np.concatenate(probs_all)
        y_all = np.concatenate(y_all)

        thrs = np.arange(0.05, 0.96, 0.01)
        f1s = [f1_score(y_all, probs_all >= t) for t in thrs]
        best_thr = float(thrs[int(np.argmax(f1s))])
        print(f"[CV] best threshold = {best_thr:.3f} (F1={np.max(f1s):.3f})")

    # 5) 최종 학습(전체 데이터)
    from utils import fit_scaler_and_model
    scaler, clf = fit_scaler_and_model(
        X, y_tr,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

    # 6) 번들 저장
    model_path = Path(args.model_path)
    save_bundle(
        model_path, clf, scaler, feature_names,
        windowing={"win": win, "step": step, "awake_ratio": args.awake_ratio, "drowsy_ratio": args.drowsy_ratio},
        postproc={"prob_threshold": best_thr, "m": args.m, "n": args.n, "cooldown_w": args.cooldown_w, "k_margin": args.k_margin},
        best_threshold=best_thr
    )
    print(f"[완료] 모델 저장: {model_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/train")
    ap.add_argument("--model_path", type=str, default="models/model.pkl")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--awake_ratio", type=float, default=0.75)
    ap.add_argument("--drowsy_ratio", type=float, default=0.75)

    # RandomForest 하이퍼파라미터
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--random_state", type=int, default=42)

    # 임계값/후처리
    ap.add_argument("--cv", action="store_true", help="GroupKFold로 best threshold 탐색")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k_margin", type=int, default=1)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cooldown_w", type=int, default=2)

    args = ap.parse_args()
    main(args)
