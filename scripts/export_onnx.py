"""Export the trained CAE to ONNX, check parity with JAX and benchmark ONNX Runtime on CPU.

    python scripts/export_onnx.py [--threads 2]
    python scripts/export_onnx.py --method draem [--threads 2]

Outputs: models/afp_cae.onnx and results/ort_bench.json
         (DRAEM-CAE: models/draem_cae.onnx and results/ort_bench_draem.json)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
from scipy.ndimage import label

from afp_cae import draem, model as cae
from afp_cae.onnx_export import build_draem_onnx, build_onnx
from afp_cae.pipeline import DEFAULT, anomaly_map, jax_predictor, onnx_predictor, preprocess, tiles
from afp_cae.synth import H, PX_MM, make_frame


def bench(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return [float(np.median(ts) * 1e3), float(np.percentile(ts, 95) * 1e3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["cae", "draem"], default="cae")
    ap.add_argument("--weights", default=None, help="default: models/afp_cae.npz or models/draem_cae.npz")
    ap.add_argument("--onnx", default=None, help="default: models/afp_cae.onnx or models/draem_cae.onnx")
    ap.add_argument("--threshold", default=None, help="default: models/threshold.json or models/threshold_draem.json")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None, help="default: results/ort_bench.json or results/ort_bench_draem.json")
    a = ap.parse_args()
    is_draem = a.method == "draem"
    suffix = "_draem" if is_draem else ""
    stem = "draem_cae" if is_draem else "afp_cae"
    weights = a.weights or str(ROOT / "models" / f"{stem}.npz")
    a.onnx = a.onnx or str(ROOT / "models" / f"{stem}.onnx")
    a.threshold = a.threshold or str(ROOT / "models" / f"threshold{suffix}.json")
    a.out = a.out or str(ROOT / "results" / f"ort_bench{suffix}.json")
    net = draem if is_draem else cae

    model = (build_draem_onnx if is_draem else build_onnx)(net.load_params(weights, as_jax=False))
    onnx.save(model, a.onnx)
    print("saved", a.onnx)

    cfg = DEFAULT
    kind = "map" if is_draem else "residual"
    f, _ = make_frame(np.random.default_rng(9), "gap")
    t, _, _ = tiles(preprocess(f), cfg.tile, cfg.stride)
    ort_pred = onnx_predictor(a.onnx, a.threads)
    jax_pred = (draem.jax_predictor if is_draem else jax_predictor)(net.load_params(weights))
    diff = float(np.abs(ort_pred(t) - jax_pred(t)).max())
    print(f"max |ORT - JAX| = {diff:.2e}")
    assert diff < 1e-4, "ONNX graph does not reproduce the JAX model"

    tau = json.load(open(a.threshold))["tau"]
    sess = ort_pred.session
    x1, xb = t[:1, None].copy(), t[:, None].copy()
    r = {
        "backend": "ONNX Runtime CPUExecutionProvider, FP32", "intra_op_threads": a.threads,
        "max_abs_diff_vs_jax": diff,
        "tile_latency_ms_median_p95": bench(lambda: sess.run(None, {"x": x1}), 500),
        "frame_cae_latency_ms_median_p95": bench(lambda: sess.run(None, {"x": xb}), 200),
        "frame_pipeline_ms_median_p95": bench(lambda: label(anomaly_map(preprocess(f), ort_pred, cfg, kind=kind) > tau), 200),
        "onnx_size_MB": os.path.getsize(a.onnx) / 2 ** 20,
    }
    r["tiles_per_s"] = len(t) / (r["frame_cae_latency_ms_median_p95"][0] / 1e3)
    r["frames_per_s"] = 1e3 / r["frame_pipeline_ms_median_p95"][0]
    r["max_layup_speed_m_min"] = r["frames_per_s"] * H * PX_MM * 1e-3 * 60
    print(json.dumps(r, indent=1))
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
