"""Calibrate the decision threshold tau on defect-free scans (eq. 2: tau = max over V0 of A).

    python scripts/calibrate.py --data scans/validation_defect_free --normalize
    python scripts/calibrate.py --synthetic 100          # synthetic validation set (as in the abstract)
    python scripts/calibrate.py --synthetic 100 --method draem

`--margin` multiplies the maximum (e.g. 1.05 for a 5 % safety margin); `--quantile` < 1 uses a
quantile of per-scan maxima instead of the maximum (tolerates a few false alarms, raises recall).
Writes models/threshold.json (DRAEM-CAE: models/threshold_draem.json), which scripts/detect.py reads.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from afp_cae.pipeline import DEFAULT, METHODS, anomaly_map, backend, load_scans, preprocess
from afp_cae.synth import make_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="folder with defect-free height maps (*.npy)")
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic defect-free frames (seed 1)")
    ap.add_argument("--normalize", action="store_true", help="min-max scale real scans to [0, 1]")
    ap.add_argument("--method", choices=list(METHODS), default="cae")
    ap.add_argument("--weights", default=None, help="default: models/afp_cae.npz or models/draem_cae.npz")
    ap.add_argument("--quantile", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="default: models/threshold.json or models/threshold_draem.json")
    a = ap.parse_args()
    a.out = a.out or str(ROOT / "models" / f"threshold{METHODS[a.method][1]}.json")

    if a.data:
        scans = load_scans(a.data, a.normalize)
    else:
        rng = np.random.default_rng(1)
        scans = [make_frame(rng)[0] for _ in range(a.synthetic)]
    predict, kind = backend(a.method, a.weights, models_dir=ROOT / "models")
    maxima = np.array([anomaly_map(preprocess(s), predict, DEFAULT, kind=kind).max() for s in scans])
    tau = float(np.quantile(maxima, a.quantile) * a.margin)
    print(f"{len(scans)} defect-free scans; per-scan max A: median {np.median(maxima):.4f}, "
          f"max {maxima.max():.4f}; tau = {tau:.5f}")
    json.dump({"tau": tau, "calibration": f"quantile {a.quantile} x margin {a.margin} of per-scan max A "
                                          f"over {len(scans)} defect-free scans ({a.data or 'synthetic'})",
               "method": a.method, "pipeline": DEFAULT.__dict__}, open(a.out, "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
