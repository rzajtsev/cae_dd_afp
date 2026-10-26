"""Evaluate the CAE and DRAEM-CAE on real AFP scans of Cardoso et al. (2026) with synthetic defects.

    python scripts/eval_cardoso.py --data path/to/cardoso/data                      # both methods, seeds 0 1 2
    python scripts/eval_cardoso.py --data path/to/cardoso/data --methods draem --seeds 0 1 2 3 4
    python scripts/eval_cardoso.py --data path/to/cardoso/data --quick              # smoke run (~2 min)
    python scripts/eval_cardoso.py --data path/to/cardoso/data --methods draem --ensemble 3   # mean map of 3 models

`--data` is the folder with PART-*/Layer_*/Pass_* of Zenodo 10.5281/zenodo.17048470.

Protocol (cross-validation over passes; 2 layers x 3 passes)
  * fold k: train on the 4 passes other than Pass_0k of both layers, test on Layer_01_Pass_0k and Layer_02_Pass_0k
  * threshold: every training pass is scored by an auxiliary model trained on the other 3 training passes;
    tau of a layer = 99.5th percentile of these scores over the passes of that layer (no test data involved)
  * test: 10 trials per test pass, each injects 2 defects of each of 5 types (gap, overlap, foreign object,
    twist, wrinkle; `afp_cae.cardoso.inject`) into the clean scan; the defects depend only on fold and pass,
    so all methods and seeds see the same ones
  * a defect is detected if max A inside its mask > tau; a false zone is a connected component of {A > tau}
    that does not touch the defect masks dilated by 3 px; tile AUROC over non-overlapping 64 x 64 tiles
    of the scan grid (a tile is defective if it holds any ground-truth pixel)
  * tow seams (`afp_cae.cardoso.seam_mask` of the clean pass): a false zone or a defect is "on a seam" if more
    than half of its area lies in the seam zone; reported next to the share of the scan area the zone covers
Maps are computed on the 0.1 mm/px grid of the CAE (`afp_cae.cardoso.Grid`) and resampled back to the scan grid.
`--ensemble N`: every model (auxiliary and final) is a mean of N maps of models trained with seeds
seed, seed + 1000, ..., seed + 1000 (N - 1); this averages out run-to-run instability of GPU training.
Outputs: results/cardoso/results.json (all rows) and results/cardoso/summary.json
         (with --ensemble N > 1: results/cardoso_ens<N>/, methods named e.g. 'draem_ens3')
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from afp_cae import draem, model
from afp_cae.cardoso import DEFECT_TYPES, Grid, inject, load_all, seam_mask, tiles_dataset
from afp_cae.pipeline import DEFAULT, anomaly_map, jax_predictor
import train
import train_draem

THR_PCT = 99.5
TILE_EVAL = 64      # px of the scan grid, tile-level AUROC
GROW_PX = 3         # false zones must not touch the defect masks dilated by this
N_TILES = 16000


class Detector:
    """CAE or DRAEM-CAE on real scans: fit(list[HeightMap]), score(HeightMap or z, dx, dy) -> map on the scan grid."""

    def __init__(self, method, seed=0, epochs=None, sigma=None):
        self.method, self.seed = method, seed
        self.epochs = epochs or (100 if method == "draem" else 25)
        # the course edge is masked after removing the reflect padding (see score)
        self.cfg = replace(DEFAULT, edge=0, sigma=DEFAULT.sigma if sigma is None else sigma)

    def fit(self, maps):
        self.grid = Grid(maps)
        rng = np.random.default_rng(self.seed)
        X = tiles_dataset(rng, [self.grid.to_grid(m.z, m.dx, m.dy) for m in maps], N_TILES)
        if self.method == "draem":
            params, _ = train_draem.fit(X, rng, self.epochs, seed=self.seed, verbose=False)
            self.predict, self.kind = draem.jax_predictor(params), "map"
        else:
            params, _ = train.fit(X, rng, self.epochs, seed=self.seed, verbose=False)
            self.predict, self.kind = jax_predictor(params), "residual"
        return self

    def score(self, z, dx, dy):
        x = self.grid.to_grid(z, dx, dy)
        # tiles() skips a tail that is not a multiple of the stride: pad by reflection, crop afterwards
        H, W = x.shape
        t, s = self.cfg.tile, self.cfg.stride
        ph = (-(max(H, t) - t)) % s + max(t - H, 0)
        pw = (-(max(W, t) - t)) % s + max(t - W, 0)
        A = anomaly_map(np.pad(x, ((0, ph), (0, pw)), mode="reflect"), self.predict, self.cfg, kind=self.kind)[:H, :W]
        A[:, :DEFAULT.edge] = 0
        A[:, -DEFAULT.edge:] = 0
        return Grid.from_grid(A, z.shape, dx, dy)


class Ensemble:
    """Mean anomaly map of n Detectors that differ only in the training seed."""

    def __init__(self, method, seed=0, epochs=None, sigma=None, n=1):
        self.members = [Detector(method, seed + 1000 * i, epochs, sigma) for i in range(n)]

    def fit(self, maps):
        for d in self.members:
            d.fit(maps)
        return self

    def score(self, z, dx, dy):
        return np.mean([d.score(z, dx, dy) for d in self.members], axis=0)


def layer_of(m):
    return m.name.split("_Pass")[0]


def tile_scores(A, gt):
    s, y = [], []
    for i in range(0, A.shape[0] - TILE_EVAL + 1, TILE_EVAL):
        for j in range(0, A.shape[1] - TILE_EVAL + 1, TILE_EVAL):
            s.append(A[i:i + TILE_EVAL, j:j + TILE_EVAL].max())
            y.append(gt[i:i + TILE_EVAL, j:j + TILE_EVAL].any())
    return np.array(s), np.array(y)


def false_zones(A, tau, gt, seam):
    """(number of false zones, how many of them lie on a seam)."""
    above = A > tau
    lab, n = ndimage.label(above)
    grown = ndimage.binary_dilation(gt, iterations=GROW_PX)
    ids = sorted(set(np.unique(lab[above & ~grown])) - set(np.unique(lab[grown & above])) - {0})
    if not ids:
        return 0, 0
    on = np.asarray(ndimage.mean(seam.astype(np.float32), lab, ids)) > 0.5
    return len(ids), int(on.sum())


def calibrate(make, train_maps):
    """tau per layer from scores of models that did not see the scored pass."""
    scores = {}
    for held in train_maps:
        aux = make().fit([m for m in train_maps if m is not held])
        scores.setdefault(layer_of(held), []).append(aux.score(held.z, held.dx, held.dy).ravel())
    return {k: float(np.percentile(np.concatenate(v), THR_PCT)) for k, v in scores.items()}


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(defect_rows, trial_rows):
    out = {}
    for meth in sorted({r["method"] for r in trial_rows}):
        d = [r for r in defect_rows if r["method"] == meth]
        t = [r for r in trial_rows if r["method"] == meth]
        seeds = sorted({r["seed"] for r in t})
        det = [r["detected"] for r in d]
        out[meth] = {
            "seeds": seeds, "defects": len(d),
            "tile_auroc": float(np.nanmean([r["tile_auroc"] for r in t])),
            "pixel_auroc": float(np.mean([r["pixel_auroc"] for r in t])),
            "detected": float(np.mean(det)), "detected_ci95": bootstrap_ci(det),
            "detected_by_seed": [float(np.mean([r["detected"] for r in d if r["seed"] == s])) for s in seeds],
            "false_zones_per_m": float(np.mean([r["false_zones_per_m"] for r in t])),
            "false_zones_per_m_by_seed": [float(np.mean([r["false_zones_per_m"] for r in t if r["seed"] == s]))
                                          for s in seeds],
            "detected_by_type": {k: float(np.mean([r["detected"] for r in d if r["kind"] == k])) for k in DEFECT_TYPES},
            "seams": {
                "area_share": float(np.mean([r["seam_share"] for r in t])),
                "false_zones_per_m_on_seams": float(np.mean([r["false_zones_seam_per_m"] for r in t])),
                "false_zone_share_on_seams": float(sum(r["false_zones_seam_per_m"] for r in t)
                                                   / max(sum(r["false_zones_per_m"] for r in t), 1e-9)),
                "defects_on_seams": int(sum(r["on_seam"] for r in d)),
                "detected_on_seams": float(np.mean([r["detected"] for r in d if r["on_seam"]] or [np.nan])),
                "detected_off_seams": float(np.mean([r["detected"] for r in d if not r["on_seam"]] or [np.nan])),
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Cardoso dataset folder (contains PART-*)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cardoso_cache"), help="cache of parsed passes (.npz)")
    ap.add_argument("--methods", nargs="+", choices=["cae", "draem"], default=["cae", "draem"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="training seeds")
    ap.add_argument("--trials", type=int, default=10, help="defect injections per test pass")
    ap.add_argument("--sigma", type=float, default=None, help="override the smoothing sigma of the map, px")
    ap.add_argument("--ensemble", type=int, default=1, help="models per detector (mean map), 1 = single model")
    ap.add_argument("--quick", action="store_true", help="fold 3, 1 trial, seed 0, 2 / 5 epochs")
    ap.add_argument("--out", default=None, help="default: results/cardoso or results/cardoso_ens<N>")
    a = ap.parse_args()
    a.out = a.out or str(ROOT / "results" / ("cardoso" if a.ensemble == 1 else f"cardoso_ens{a.ensemble}"))
    folds = (3,) if a.quick else (1, 2, 3)
    seeds = [0] if a.quick else a.seeds
    trials = 1 if a.quick else a.trials
    epochs = {"cae": 2, "draem": 5} if a.quick else {"cae": None, "draem": None}

    maps = load_all(a.data, a.cache)
    by_name = {m.name: m for m in maps}
    print({m.name: m.z.shape for m in maps}, flush=True)
    defect_rows, trial_rows = [], []
    for k in folds:
        test = [by_name[f"Layer_01_Pass_0{k}"], by_name[f"Layer_02_Pass_0{k}"]]
        train_maps = [m for m in maps if m not in test]
        for base in a.methods:
            meth = base if a.ensemble == 1 else f"{base}_ens{a.ensemble}"
            for seed in seeds:
                make = lambda: Ensemble(base, seed, epochs[base], a.sigma, a.ensemble)
                t0 = time.time()
                tau = calibrate(make, train_maps)
                det = make().fit(train_maps)
                for m in test:
                    rng = np.random.default_rng(1000 * k + sum(map(ord, m.name)))
                    length_m = m.z.shape[0] * m.dy / 1000
                    seam = seam_mask(m.z, m.dx, m.dy)
                    for t in range(trials):
                        zd, gt, defs = inject(m.z, m.dx, m.dy, rng)
                        A = det.score(zd, m.dx, m.dy)
                        ts, ty = tile_scores(A, gt)
                        n_fz, n_fz_seam = false_zones(A, tau[layer_of(m)], gt, seam)
                        trial_rows.append(dict(
                            fold=k, method=meth, seed=seed, map=m.name, trial=t, tau=tau[layer_of(m)],
                            pixel_auroc=float(roc_auc_score(gt.ravel(), A.ravel())),
                            tile_auroc=float(roc_auc_score(ty, ts)) if 0 < ty.sum() < len(ty) else float("nan"),
                            false_zones_per_m=n_fz / length_m, false_zones_seam_per_m=n_fz_seam / length_m,
                            seam_share=float(seam.mean())))
                        for d in defs:
                            defect_rows.append(dict(fold=k, method=meth, seed=seed, map=m.name, trial=t, kind=d.kind,
                                                    size=d.size, detected=bool(A[d.mask].max() > tau[layer_of(m)]),
                                                    on_seam=bool(seam[d.mask].mean() > 0.5)))
                cur = [r for r in defect_rows if (r["fold"], r["method"], r["seed"]) == (k, meth, seed)]
                cur_t = [r for r in trial_rows if (r["fold"], r["method"], r["seed"]) == (k, meth, seed)]
                print(f"fold {k}  {meth:5s} seed {seed}:  tile AUROC {np.nanmean([r['tile_auroc'] for r in cur_t]):.3f}"
                      f"  detected {np.mean([r['detected'] for r in cur]):.3f}"
                      f"  false zones/m {np.mean([r['false_zones_per_m'] for r in cur_t]):.1f}"
                      f" (on seams {np.mean([r['false_zones_seam_per_m'] for r in cur_t]):.1f},"
                      f" seams {100 * np.mean([r['seam_share'] for r in cur_t]):.0f} % of area)"
                      f"  ({time.time() - t0:.0f} s)", flush=True)

    summary = summarize(defect_rows, trial_rows)
    print(json.dumps(summary, indent=1))
    if not a.quick:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"defects": defect_rows, "trials": trial_rows}, open(out / "results.json", "w"), indent=1)
        json.dump({"protocol": {"folds": folds, "seeds": seeds, "trials": trials, "thr_pct": THR_PCT,
                                "sigma": a.sigma or DEFAULT.sigma}, "methods": summary},
                  open(out / "summary.json", "w"), indent=1)
        print("written", out)


if __name__ == "__main__":
    main()
