"""Calibrate the threshold, evaluate detection quality and benchmark JAX inference.

    python scripts/evaluate.py             # full protocol of the abstract (100 + 570 frames, ~6 min)
    python scripts/evaluate.py --quick     # small smoke run

Protocol
  * threshold tau = max of A over 100 defect-free validation frames (seed 1) -> zero false alarms there
  * test set (seed 2): 150 defect-free frames + 60 frames per defect type (7 types)
  * tile level: tile is defective if it holds >= 20 px of the ground-truth mask; score = max A in tile
  * instance level: a defect is detected if a component (area >= 20 px) of {A > tau} overlaps its mask;
    a false alarm is a component farther than 3 mm (30 px) from any defect
Outputs: results/results.json and models/threshold.json
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from sklearn.metrics import average_precision_score, roc_auc_score

from afp_cae.model import count_params, load_params, macs_per_tile
from afp_cae.pipeline import DEFAULT, anomaly_map, jax_predictor, preprocess, tile_scores, tiles
from afp_cae.synth import DEFECT_TYPES, H, PX_MM, make_frame

TH_LAB = 20       # px of ground truth that make a tile "defective"
NEAR_PX = 30      # 3 mm: components closer than this to a defect are not counted as false alarms


def frames(seed, n_norm, n_per):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_norm):
        f, m = make_frame(rng)
        out.append((None, preprocess(f), m))
    for k in DEFECT_TYPES:
        for _ in range(n_per):
            f, m = make_frame(rng, k)
            out.append((k, preprocess(f), m))
    return out


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


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
    ap.add_argument("--weights", default=str(ROOT / "models" / "afp_cae.npz"))
    ap.add_argument("--quick", action="store_true", help="10 validation, 10 normal + 4 per type test frames")
    ap.add_argument("--out", default=None, help="results JSON (default: results/results.json, not written in --quick)")
    a = ap.parse_args()
    cfg = DEFAULT
    nv, nt, nper = (10, 10, 4) if a.quick else (100, 150, 60)

    params = load_params(a.weights)
    predict = jax_predictor(params)

    # ---- threshold calibration on defect-free validation frames ----
    tau = float(max(anomaly_map(f, predict, cfg).max() for _, f, _ in frames(1, nv, 0)))
    print(f"tau = {tau:.5f}", flush=True)

    # ---- test ----
    test = frames(2, nt, nper)
    y_t, s_t, pix_y, pix_s = [], [], [], []
    det = {k: [] for k in DEFECT_TYPES}
    ious = {k: [] for k in DEFECT_TYPES}
    fp_comp = fp_far = normal_alarms = 0
    prng = np.random.default_rng(3)
    for kind, f, m in test:
        A = anomaly_map(f, predict, cfg)
        _, ys, xs = tiles(f, cfg.tile, cfg.stride)
        sc = tile_scores(A, ys, xs, cfg.tile)
        i = 0
        for y in ys:
            for x in xs:
                y_t.append(int(m[y:y + cfg.tile, x:x + cfg.tile].sum() >= TH_LAB))
                s_t.append(sc[i])
                i += 1
        roi = np.zeros_like(m)
        roi[:, cfg.edge:-cfg.edge] = True
        idx = prng.choice(np.flatnonzero(roi), 4000, replace=False)
        if (m & roi).any():
            mi = np.flatnonzero(m & roi)
            idx = np.concatenate([idx, prng.choice(mi, min(1000, mi.size), replace=False)])
        pix_y += list(m.ravel()[idx].astype(int))
        pix_s += list(A.ravel()[idx])

        lab, n = label(A > tau)
        hit, pred = False, np.zeros_like(m)
        near = distance_transform_edt(~m) <= NEAR_PX if m.any() else np.zeros_like(m)
        for c in range(1, n + 1):
            comp = lab == c
            if comp.sum() < cfg.min_area:
                continue
            if m.any() and (comp & m).any():
                hit = True
                pred |= comp
            else:
                fp_comp += 1
                fp_far += int(not (comp & near).any())
                normal_alarms += int(kind is None)
        if kind is not None:
            det[kind].append(hit)
            if hit:
                ious[kind].append((pred & m).sum() / (pred | m).sum())

    y_t, s_t = np.array(y_t), np.array(s_t)
    pred_t = s_t > tau
    tp = int((pred_t & (y_t == 1)).sum())
    fp = int((pred_t & (y_t == 0)).sum())
    fn = int((~pred_t & (y_t == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    course_m = len(test) * H * PX_MM * 1e-3
    quality = {
        "tau": tau,
        "tile_auroc": float(roc_auc_score(y_t, s_t)),
        "tile_ap": float(average_precision_score(y_t, s_t)),
        "tile_precision": prec, "tile_recall": rec, "tile_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "n_tiles": int(len(y_t)), "n_defective_tiles": int(y_t.sum()),
        "pixel_auroc": float(roc_auc_score(pix_y, pix_s)),
        "instance_recall": float(np.mean(sum(det.values(), []))),
        "recall_by_type": {k: float(np.mean(v)) for k, v in det.items()},
        "iou_by_type": {k: float(np.mean(v)) if v else 0.0 for k, v in ious.items()},
        "mean_iou": float(np.mean(sum(ious.values(), []))) if sum(ious.values(), []) else 0.0,
        "test_frames": len(test), "normal_test_frames": nt, "course_length_m": course_m,
        "false_alarms_far": fp_far, "false_alarms_per_m": fp_far / course_m,
        "components_outside_mask": fp_comp, "false_alarms_on_normal_frames": normal_alarms,
    }
    print(json.dumps(quality, indent=1), flush=True)

    # ---- JAX inference benchmark ----
    f, _ = make_frame(np.random.default_rng(9), "gap")
    t, _, _ = tiles(preprocess(f), cfg.tile, cfg.stride)
    fwd = predict.jitted
    import jax.numpy as jnp
    x1, xb = jnp.asarray(t[:1, ..., None]), jnp.asarray(t[..., None])
    n1, nb, ne = (50, 20, 10) if a.quick else (500, 200, 100)
    e2e = bench(lambda: label(anomaly_map(preprocess(f), predict, cfg) > tau), ne)
    fps = 1e3 / e2e[0]
    n_par = count_params(params)
    perf = {
        "backend": "JAX/XLA, FP32", "cpu": cpu_name(), "cpu_count": os.cpu_count(),
        "params": n_par, "size_MB_fp32": n_par * 4 / 2 ** 20,
        "MMAC_per_tile": macs_per_tile() / 1e6, "GFLOP_per_frame": 2 * macs_per_tile() * len(t) / 1e9,
        "tiles_per_frame": len(t),
        "tile_latency_ms_median_p95": bench(lambda: fwd(x1).block_until_ready(), n1),
        "frame_cae_latency_ms_median_p95": bench(lambda: fwd(xb).block_until_ready(), nb),
        "frame_pipeline_ms_median_p95": e2e, "frames_per_s": fps,
        "max_layup_speed_m_min": fps * H * PX_MM * 1e-3 * 60,
    }
    perf["tiles_per_s"] = len(t) / (perf["frame_cae_latency_ms_median_p95"][0] / 1e3)
    print(json.dumps(perf, indent=1), flush=True)

    out = a.out or (None if a.quick else str(ROOT / "results" / "results.json"))
    if out:
        json.dump({"quality": quality, "benchmark_jax": perf}, open(out, "w"), indent=1)
        json.dump({"tau": tau, "calibration": f"max of A over {nv} defect-free validation frames (seed 1)",
                   "pipeline": cfg.__dict__}, open(ROOT / "models" / "threshold.json", "w"), indent=1)
        print("written", out)


if __name__ == "__main__":
    main()
