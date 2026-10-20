"""Train the convolutional autoencoder on defect-free synthetic AFP scans.

    python scripts/train.py                       # full run used for the abstract (~45 min on 2 CPU cores)
    python scripts/train.py --frames 20 --epochs 1 --out models/smoke.npz   # quick smoke test
    python scripts/train.py --data scans/defect_free --normalize --crops 200  # real scans (*.npy)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from afp_cae.model import TILE, count_params, init_params, loss_fn, save_params
from afp_cae.pipeline import load_scans, preprocess
from afp_cae.synth import make_frame


def build_dataset(rng, n_frames, crops, scans=None):
    """Random 64x64 crops of pre-processed defect-free frames (synthetic, or real `scans`)."""
    if scans is not None:
        n_frames = len(scans)
    X = np.empty((n_frames * crops, TILE, TILE, 1), np.float32)
    i = 0
    for j in range(n_frames):
        f = preprocess(scans[j] if scans is not None else make_frame(rng)[0])
        if min(f.shape) < TILE:
            raise ValueError(f"scan {j} is smaller than {TILE} px")
        for _ in range(crops):
            y = rng.integers(0, f.shape[0] - TILE + 1)
            x = rng.integers(0, f.shape[1] - TILE + 1)
            X[i, :, :, 0] = f[y:y + TILE, x:x + TILE]
            i += 1
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=400, help="defect-free frames to generate")
    ap.add_argument("--crops", type=int, default=40, help="random tiles per frame")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (cosine decay to 5 %)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", help="folder with real defect-free height maps (*.npy); default: synthetic")
    ap.add_argument("--normalize", action="store_true", help="min-max scale real scans to [0, 1]")
    ap.add_argument("--out", default=str(ROOT / "models" / "afp_cae.npz"))
    ap.add_argument("--log", default=str(ROOT / "results" / "train_log.json"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    scans = load_scans(a.data, a.normalize) if a.data else None
    X = build_dataset(rng, a.frames, a.crops, scans)
    print(f"dataset {X.shape} built in {time.time() - t0:.1f} s", flush=True)

    params = init_params(jax.random.PRNGKey(a.seed))
    print("parameters:", count_params(params), flush=True)
    steps = a.epochs * (len(X) // a.batch)
    opt = optax.adam(optax.cosine_decay_schedule(a.lr, steps, alpha=0.05))
    state = opt.init(params)

    @jax.jit
    def step(p, s, xb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    history = []
    for ep in range(a.epochs):
        perm = rng.permutation(len(X))
        losses, te = [], time.time()
        for b in range(len(X) // a.batch):
            params, state, loss = step(params, state, jnp.asarray(X[perm[b * a.batch:(b + 1) * a.batch]]))
            losses.append(float(loss))
        history.append(float(np.mean(losses)))
        print(f"epoch {ep + 1:3d}/{a.epochs}  loss {history[-1]:.5f}  ({time.time() - te:.1f} s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        save_params(params, a.out)   # checkpoint every epoch

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"loss": history, "params": count_params(params), "train_tiles": int(len(X)),
               "data": a.data or "synthetic",
               "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed,
               "train_time_s": time.time() - t0}, open(a.log, "w"), indent=1)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
