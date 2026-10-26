"""Effect of excluding tape boundaries (tow seams, tape edges) from the anomaly map on real Cardoso scans.

    python scripts/eval_boundaries.py --data path/to/cardoso/data                 # CAE + DRAEM-CAE, seeds 0 1 2
    python scripts/eval_boundaries.py --data path/to/cardoso/data --quick         # smoke run

Boundary detectors (`afp_cae.cardoso.BOUNDARIES`, run on the clean scan on the 0.1 mm/px grid, the same edge
slope 0.3 mm/mm and the same +-0.5 mm band around the found boundary):
    none          nothing is excluded (baseline)
    sobel_x       |Sobel_x| of the map smoothed along the layup > slope
    canny         Canny edges (hysteresis slope/2 .. slope) that run along the layup
    canny_hough   straight Hough segments (+-15 deg of the layup, >= 10 mm) on the Canny edges
Exclusion: the anomaly map is set to 0 inside the zone, as the 1.2 mm course-edge band of the CAE. Two thresholds
(protocol of scripts/eval_cardoso.py: auxiliary models that did not see the scored pass, 99.5th percentile over the
passes of a layer):
    fixed    tau of 'none' for every variant: exclusion only removes what lies in the zone (same sensitivity; main)
    recal    tau recalibrated on the kept pixels of each variant: the same 0.5 % of clean kept pixels exceed it,
             so false zones move from the excluded zone to the rest of the tape
Models are trained once per fold / method / seed; all variants are evaluated on the same maps.

Two defect sets per test pass, 10 trials each:
    random   defects of scripts/eval_cardoso.py (identical injections)
    seams    half of the gaps and overlaps centred on a tow seam (`inject(..., seam=, seam_frac=0.5)`)
Seams for placement and for the on/off-seam statistics come from one reference zone (`seam_mask`), not from the
detector under test.

Metrics per set / method / variant: defects detected (all, on seams, off seams, by type), false zones per metre
(all, on seams), tile AUROC, share of the scan area excluded.
Outputs: results/cardoso_boundaries/{results,summary}.json and masks.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from afp_cae.cardoso import BOUNDARIES, DEFECT_TYPES, inject, load_all, seam_mask
from eval_cardoso import THR_PCT, Detector, bootstrap_ci, false_zones, layer_of, tile_scores

VARIANTS = ["none", *BOUNDARIES]
SETS = ("random", "seams")
THRESHOLDS = ("fixed", "recal")
SEAM_SET_SALT = 7919    # the seam set draws from its own random stream


def zones_of(m):
    """Exclusion zone of every variant for a clean pass."""
    z = {v: fn(m.z, m.dx, m.dy) for v, fn in BOUNDARIES.items()}
    z["none"] = np.zeros(m.z.shape, bool)
    return z


def plot_masks(maps, zones, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(BOUNDARIES), len(maps), figsize=(4 * len(maps), 6 * len(BOUNDARIES)), squeeze=False)
    for j, m in enumerate(maps):
        ext = [0, m.z.shape[1] * m.dx, m.z.shape[0] * m.dy, 0]
        for i, v in enumerate(BOUNDARIES):
            zone = zones[m.name][v]
            a = ax[i, j]
            a.imshow(m.z, cmap="gray", vmin=np.percentile(m.z, 1), vmax=np.percentile(m.z, 99), extent=ext, aspect="auto")
            a.imshow(np.ma.masked_where(~zone, zone), cmap="autumn", alpha=0.45, extent=ext, aspect="auto")
            a.set_title(f"{v} | {m.name} | {100 * zone.mean():.1f} %", fontsize=9)
            a.set_xlabel("across, mm")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def summarize(defect_rows, trial_rows):
    """summary[threshold][set][method][variant] -> metrics."""
    out = {}
    for th in THRESHOLDS:
        for s in SETS:
            for meth in sorted({r["method"] for r in trial_rows}):
                d = [r for r in defect_rows if r["set"] == s and r["method"] == meth]
                for v in VARIANTS:
                    t = [r for r in trial_rows if (r["set"], r["method"], r["variant"], r["threshold"]) == (s, meth, v, th)]
                    out.setdefault(th, {}).setdefault(s, {}).setdefault(meth, {})[v] = _metrics(d, t, f"{v}|{th}")
    return out


def _metrics(d, t, key):
    """Metrics of defect rows d and trial rows t; key = 'variant|threshold' of the detected flags."""
    det = [r["detected"][key] for r in d]
    on = [r["detected"][key] for r in d if r["on_seam"]]
    off = [r["detected"][key] for r in d if not r["on_seam"]]
    return {
        "defects": len(d), "defects_on_seams": len(on),
        "detected": float(np.mean(det)), "detected_ci95": bootstrap_ci(det),
        "detected_on_seams": float(np.mean(on)) if on else float("nan"),
        "detected_off_seams": float(np.mean(off)) if off else float("nan"),
        "detected_by_type": {k: float(np.mean([r["detected"][key] for r in d if r["kind"] == k])) for k in DEFECT_TYPES},
        "false_zones_per_m": float(np.mean([r["false_zones_per_m"] for r in t])),
        "false_zones_per_m_on_seams": float(np.mean([r["false_zones_seam_per_m"] for r in t])),
        "tile_auroc": float(np.nanmean([r["tile_auroc"] for r in t])),
        "excluded_share": float(np.mean([r["excluded_share"] for r in t])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Cardoso dataset folder (contains PART-*)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cardoso_cache"), help="cache of parsed passes (.npz)")
    ap.add_argument("--methods", nargs="+", choices=["cae", "draem"], default=["cae", "draem"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=10, help="defect injections per test pass and set")
    ap.add_argument("--seam-frac", type=float, default=0.5, help="share of gaps / overlaps centred on seams")
    ap.add_argument("--quick", action="store_true", help="fold 3, 1 trial, seed 0, 2 / 5 epochs, nothing written")
    ap.add_argument("--out", default=str(ROOT / "results" / "cardoso_boundaries"))
    a = ap.parse_args()
    folds = (3,) if a.quick else (1, 2, 3)
    seeds = [0] if a.quick else a.seeds
    trials = 1 if a.quick else a.trials
    epochs = {"cae": 2, "draem": 5} if a.quick else {"cae": None, "draem": None}
    seam_frac = {"random": 0.0, "seams": a.seam_frac}

    maps = load_all(a.data, a.cache)
    by_name = {m.name: m for m in maps}
    zones = {m.name: zones_of(m) for m in maps}
    seam_ref = {m.name: seam_mask(m.z, m.dx, m.dy) for m in maps}
    for m in maps:
        print(m.name, "excluded:", {v: f"{100 * zones[m.name][v].mean():.1f} %" for v in BOUNDARIES}, flush=True)
    out = Path(a.out)
    if not a.quick:
        out.mkdir(parents=True, exist_ok=True)
        plot_masks(maps, zones, out / "masks.png")

    defect_rows, trial_rows = [], []
    for k in folds:
        test = [by_name[f"Layer_01_Pass_0{k}"], by_name[f"Layer_02_Pass_0{k}"]]
        train_maps = [m for m in maps if m not in test]
        for meth in a.methods:
            for seed in seeds:
                t0 = time.time()
                # threshold per variant and layer from models that did not see the scored pass
                kept = {v: {} for v in VARIANTS}
                for held in train_maps:
                    aux = Detector(meth, seed, epochs[meth]).fit([m for m in train_maps if m is not held])
                    A = aux.score(held.z, held.dx, held.dy)
                    for v in VARIANTS:
                        kept[v].setdefault(layer_of(held), []).append(A[~zones[held.name][v]])
                tau = {v: {L: float(np.percentile(np.concatenate(s), THR_PCT)) for L, s in kept[v].items()} for v in VARIANTS}
                det = Detector(meth, seed, epochs[meth]).fit(train_maps)
                for m in test:
                    length_m = m.z.shape[0] * m.dy / 1000
                    for s in SETS:
                        rng = np.random.default_rng(1000 * k + sum(map(ord, m.name)) + (SEAM_SET_SALT if s == "seams" else 0))
                        for t in range(trials):
                            zd, gt, defs = inject(m.z, m.dx, m.dy, rng, seam=seam_ref[m.name], seam_frac=seam_frac[s])
                            A = det.score(zd, m.dx, m.dy)
                            hit = [{} for _ in defs]
                            for v in VARIANTS:
                                Av = np.where(zones[m.name][v], 0, A)
                                ts, ty = tile_scores(Av, gt)
                                auroc = float(roc_auc_score(ty, ts)) if 0 < ty.sum() < len(ty) else float("nan")
                                for th in THRESHOLDS:
                                    thr = tau["none" if th == "fixed" else v][layer_of(m)]
                                    n_fz, n_fz_seam = false_zones(Av, thr, gt, seam_ref[m.name])
                                    trial_rows.append(dict(
                                        fold=k, method=meth, seed=seed, map=m.name, set=s, trial=t, variant=v,
                                        threshold=th, tau=thr, tile_auroc=auroc,
                                        false_zones_per_m=n_fz / length_m, false_zones_seam_per_m=n_fz_seam / length_m,
                                        excluded_share=float(zones[m.name][v].mean())))
                                    for h, d in zip(hit, defs):
                                        h[f"{v}|{th}"] = bool(Av[d.mask].max() > thr)
                            for h, d in zip(hit, defs):
                                defect_rows.append(dict(fold=k, method=meth, seed=seed, map=m.name, set=s, trial=t,
                                                        kind=d.kind, size=d.size, on_seam=d.on_seam, detected=h))
                cur = [r for r in defect_rows if (r["fold"], r["method"], r["seed"]) == (k, meth, seed)]
                line = "  ".join(f"{v} {np.mean([r['detected'][v + '|fixed'] for r in cur]):.3f}" for v in VARIANTS)
                print(f"fold {k}  {meth:5s} seed {seed}: detected  {line}  ({time.time() - t0:.0f} s)", flush=True)

    summary = summarize(defect_rows, trial_rows)
    print(f"\n{'thr':5s} {'set':7s} {'method':6s} {'variant':12s} {'excl':>5s} {'det':>6s} {'on seam':>8s} "
          f"{'off seam':>8s} {'FZ/m':>6s} {'FZ seam':>8s} {'AUROC':>6s}")
    for th, by_s in summary.items():
        for s, by_m in by_s.items():
            for meth, by_v in by_m.items():
                for v, r in by_v.items():
                    print(f"{th:5s} {s:7s} {meth:6s} {v:12s} {100 * r['excluded_share']:4.1f}% {100 * r['detected']:5.1f}% "
                          f"{100 * r['detected_on_seams']:7.1f}% {100 * r['detected_off_seams']:7.1f}% "
                          f"{r['false_zones_per_m']:6.1f} {r['false_zones_per_m_on_seams']:8.1f} {r['tile_auroc']:6.3f}")
    if not a.quick:
        json.dump({"defects": defect_rows, "trials": trial_rows}, open(out / "results.json", "w"))
        json.dump({"protocol": {"folds": folds, "seeds": seeds, "trials": trials, "thr_pct": THR_PCT,
                                "seam_frac": a.seam_frac, "variants": VARIANTS, "thresholds": THRESHOLDS},
                   "results": summary},
                  open(out / "summary.json", "w"), indent=1)
        print("written", out)


if __name__ == "__main__":
    main()
