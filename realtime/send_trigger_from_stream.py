# realtime/send_trigger_from_stream.py
import argparse, json, socket, time, joblib
from pathlib import Path
import numpy as np

from utils import read_csv_flex, build_windows, make_transition_features, predict_proba_from_features

class MonOfN:
    def __init__(self, m=2, n=5):
        from collections import deque
        self.m, self.buf = m, deque(maxlen=n)
    def update(self, bit):
        self.buf.append(int(bit))
        return 1 if sum(self.buf) >= self.m else 0

def load_thr(thr_arg, thr_json, default=0.09):
    if thr_arg is not None:
        return float(thr_arg)
    if thr_json and Path(thr_json).exists():
        with open(thr_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for k in ("best_threshold", "threshold", "thr"):
            if k in obj:
                return float(obj[k])
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="models/model.pkl")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--thr_json", default="outputs/best_threshold.json")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rt_factor", type=float, default=10.0, help="재생 가속(스텝/rt_factor 초마다 1윈도우)")
    ap.add_argument("--host", required=True, help="라즈베리파이 IP/호스트")
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()

    thr = load_thr(args.thr, args.thr_json)
    model = joblib.load(args.model)

    df = read_csv_flex(args.csv)
    wdf = build_windows(df_subj=df, win=args.win, step=args.step,
                        awake_ratio=0.75, drowsy_ratio=0.75, label_from_stage=False)
    X = make_transition_features(wdf)
    prob = predict_proba_from_features(model, X)
    if prob.ndim == 2:
        prob = prob[:, 1]

    filt = MonOfN(m=args.m, n=args.n)
    prev = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    print(f"[TX] connected to {args.host}:{args.port}  thr={thr:.4f}  m-of-n={args.m}/{args.n}")

    dt = max(0.02, args.step / max(1.0, args.rt_factor))
    try:
        for p in prob:
            bit = 1 if p >= thr else 0
            bit = filt.update(bit)
            # rising-edge만 전송
            if bit == 1 and prev == 0:
                s.sendall(b"1\n")
                print("↑1", end="", flush=True)  # 로컬 확인 로그
            else:
                print("0", end="", flush=True)
            prev = bit
            time.sleep(dt)
        print("\n[TX] done")
    finally:
        s.close()

if __name__ == "__main__":
    main()
