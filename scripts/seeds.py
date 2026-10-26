"""Train and evaluate a method with several seeds on the synthetic protocol; report mean and spread.

    python scripts/seeds.py --method draem --seeds 0 1 2 3 4
    python scripts/seeds.py --method cae   --seeds 0 1 2 3 4

Each seed runs scripts/train.py (or train_draem.py) and scripts/evaluate.py with default settings; a seed whose
weights and results already exist is reused (--retrain recomputes it).
Outputs: models/seeds/<stem>_s<seed>.npz, results/seeds/<method>_s<seed>.json and results/seeds_<method>.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afp_cae.pipeline import METHODS

KEYS = ["tile_auroc", "tile_ap", "pixel_auroc", "instance_recall", "mean_iou", "false_alarms_per_m",
        "false_alarms_on_normal_frames"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(METHODS), default="draem")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--retrain", action="store_true", help="recompute seeds that already have results")
    a = ap.parse_args()
    stem = METHODS[a.method][0]
    train = ROOT / "scripts" / ("train_draem.py" if a.method == "draem" else "train.py")
    (ROOT / "models" / "seeds").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "seeds").mkdir(parents=True, exist_ok=True)

    runs = {}
    for s in a.seeds:
        w = ROOT / "models" / "seeds" / f"{stem}_s{s}.npz"
        res = ROOT / "results" / "seeds" / f"{a.method}_s{s}.json"
        if w.exists() and res.exists() and not a.retrain:
            print(f"--- seed {s}: reusing {res.name}", flush=True)
            runs[s] = json.load(open(res))["quality"]
            continue
        print(f"--- seed {s}", flush=True)
        subprocess.run([sys.executable, str(train), "--seed", str(s), "--out", str(w),
                        "--log", str(ROOT / "results" / "seeds" / f"train_log_{a.method}_s{s}.json")], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--method", a.method, "--weights", str(w),
                        "--out", str(res), "--threshold", str(ROOT / "results" / "seeds" / f"threshold_{a.method}_s{s}.json")],
                       check=True, stdout=subprocess.DEVNULL)
        runs[s] = json.load(open(res))["quality"]

    q = list(runs.values())
    summary = {"method": a.method, "seeds": a.seeds,
               **{k: {"mean": float(np.mean([r[k] for r in q])), "std": float(np.std([r[k] for r in q], ddof=1))
                      if len(q) > 1 else 0.0, "per_seed": [r[k] for r in q]} for k in KEYS},
               "recall_by_type_mean": {t: float(np.mean([r["recall_by_type"][t] for r in q]))
                                       for t in q[0]["recall_by_type"]}}
    out = ROOT / "results" / f"seeds_{a.method}.json"
    json.dump(summary, open(out, "w"), indent=1)
    for k in KEYS:
        print(f"{k:30s} {summary[k]['mean']:.4f} +- {summary[k]['std']:.4f}   {np.round(summary[k]['per_seed'], 4).tolist()}")
    print("written", out)


if __name__ == "__main__":
    main()
