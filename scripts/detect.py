"""Run defect detection on one height map and report the defects found.

    python scripts/detect.py --demo                       # synthetic frame with a random defect
    python scripts/detect.py --input scan.npy --normalize --png out.png --json out.json
    python scripts/detect.py --input scan.npy --backend onnx

The input is a 2-D array saved with numpy.save: height map across x along the layup direction,
0.1 mm/px. Values must be in [0, 1] (use --normalize for raw heights; min-max scaling).
NOTE: the shipped weights were trained on synthetic scans; for real data retrain the model
(scripts/train.py --data DIR) and recalibrate the threshold (scripts/calibrate.py --data DIR).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from afp_cae.model import load_params
from afp_cae.pipeline import (DEFAULT, anomaly_map, detect, jax_predictor, normalize_scan, onnx_predictor,
                              preprocess)
from afp_cae.synth import DEFECT_TYPES, make_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help=".npy height map")
    src.add_argument("--demo", action="store_true", help="use a synthetic frame with a random defect")
    ap.add_argument("--normalize", action="store_true", help="min-max scale the input to [0, 1]")
    ap.add_argument("--backend", choices=["jax", "onnx"], default="jax")
    ap.add_argument("--weights", default=str(ROOT / "models" / "afp_cae.npz"))
    ap.add_argument("--onnx", default=str(ROOT / "models" / "afp_cae.onnx"))
    ap.add_argument("--threshold", default=str(ROOT / "models" / "threshold.json"),
                    help="JSON with key 'tau' or a number")
    ap.add_argument("--seed", type=int, default=None, help="seed for --demo")
    ap.add_argument("--json", help="write detections to this JSON file")
    ap.add_argument("--png", help="save input / anomaly map / detections figure")
    a = ap.parse_args()

    if a.demo:
        rng = np.random.default_rng(a.seed)
        kind = DEFECT_TYPES[rng.integers(len(DEFECT_TYPES))]
        frame, mask = make_frame(rng, kind)
        print("demo frame with defect:", kind)
    else:
        frame = np.load(a.input).astype(np.float32)
        if frame.ndim != 2:
            sys.exit("input must be a 2-D height map")
        if a.normalize:
            frame = normalize_scan(frame)
        mask = None
    if frame.shape[0] < DEFAULT.tile or frame.shape[1] < DEFAULT.tile:
        sys.exit(f"height map must be at least {DEFAULT.tile} x {DEFAULT.tile} px")

    try:
        tau = float(a.threshold)
    except ValueError:
        tau = float(json.load(open(a.threshold))["tau"])
    predict = onnx_predictor(a.onnx) if a.backend == "onnx" else jax_predictor(load_params(a.weights))

    x = preprocess(frame)
    A = anomaly_map(x, predict, DEFAULT)
    found = detect(A, tau, DEFAULT)
    print(f"tau = {tau:.4f}; defects found: {len(found)}")
    for d in found:
        print(f"  bbox {d['bbox_mm']} mm, area {d['area_mm2']} mm^2, peak score {d['peak_score']:.3f}")
    if a.json:
        json.dump({"tau": tau, "defects": found}, open(a.json, "w"), indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(3, 1, figsize=(10, 3 * 10 * x.shape[0] / x.shape[1] + 1.2))
        ax[0].imshow(x, cmap="gray", vmin=0, vmax=1); ax[0].set_title("height map (pre-processed)")
        ax[1].imshow(A, cmap="Blues", vmin=0, vmax=max(A.max(), 1.2 * tau)); ax[1].set_title("anomaly map A")
        ax[2].imshow(x, cmap="gray", vmin=0, vmax=1); ax[2].set_title(f"detections (A > {tau:.3f})")
        for d in found:
            x0, y0, x1, y1 = d["bbox_px"]
            ax[2].add_patch(Rectangle((x0 - 2, y0 - 2), x1 - x0 + 3, y1 - y0 + 3, fill=False, ec="#c2481c", lw=1.2))
        if mask is not None:
            ax[2].contour(mask, [0.5], colors=["#1baf7a"], linewidths=0.8, linestyles="dotted")
        for axi in ax:
            axi.set_xticks([]); axi.set_yticks([])
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)
        print("saved", a.png)


if __name__ == "__main__":
    main()
