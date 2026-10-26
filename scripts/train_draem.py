"""Train DRAEM-CAE (CAE reconstructor + U-Net discriminator) on defect-free synthetic AFP scans.

    python scripts/train_draem.py                                   # same 16 000 tiles as scripts/train.py
    python scripts/train_draem.py --init models/afp_cae.npz         # start from the trained CAE
    python scripts/train_draem.py --frames 20 --epochs 1 --out models/smoke_draem.npz   # quick smoke test
    python scripts/train_draem.py --data scans/defect_free --normalize --crops 200    # real scans (*.npy)

Synthetic anomalies are drawn on the fly on the device by `afp_cae.draem.augment` (Perlin masks), never
by the defect generator of `afp_cae.synth`.
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

from afp_cae import model as cae
from afp_cae.draem import augment, count_params, init_params, loss_fn, save_params
from afp_cae.pipeline import load_scans
from train import build_dataset


def fit(X, rng, epochs=100, batch=64, lr=1e-3, seed=0, init=None, out=None, verbose=True):
    """Adam with cosine LR decay to 5 % over tiles X[N, 64, 64, 1]; `init`: CAE parameters for the
    reconstructor. Returns (params, [total, reconstruction, segmentation] loss per epoch)."""
    params = init_params(jax.random.PRNGKey(seed), init)
    if verbose:
        print("parameters:", count_params(params), flush=True)
    steps = epochs * (len(X) // batch)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05))
    state = opt.init(params)

    @jax.jit
    def step(p, s, xb, key):
        xa, mb = augment(key, xb)
        (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, xb, xa, mb)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, jnp.stack([loss, *parts])

    Xd = jnp.asarray(X)
    key = jax.random.PRNGKey(seed + 1)
    history = []
    for ep in range(epochs):
        perm = jnp.asarray(rng.permutation(len(X)))
        losses, te = [], time.time()
        for b in range(len(X) // batch):
            key, kb = jax.random.split(key)
            params, state, loss = step(params, state, Xd[perm[b * batch:(b + 1) * batch]], kb)
            losses.append(loss)
        history.append(np.asarray(jnp.stack(losses).mean(0)).tolist())
        if verbose:
            print(f"epoch {ep + 1:3d}/{epochs}  loss {history[-1][0]:.5f}  (rec {history[-1][1]:.5f}, "
                  f"seg {history[-1][2]:.5f}, {time.time() - te:.1f} s)", flush=True)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            save_params(params, out)   # checkpoint every epoch
    return params, history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=400, help="defect-free frames to generate")
    ap.add_argument("--crops", type=int, default=40, help="random tiles per frame")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (cosine decay to 5 %)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="pre-trained CAE weights (.npz of scripts/train.py) for the reconstructor")
    ap.add_argument("--data", help="folder with real defect-free height maps (*.npy); default: synthetic")
    ap.add_argument("--normalize", action="store_true", help="min-max scale real scans to [0, 1]")
    ap.add_argument("--out", default=str(ROOT / "models" / "draem_cae.npz"))
    ap.add_argument("--log", default=str(ROOT / "results" / "train_log_draem.json"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    scans = load_scans(a.data, a.normalize) if a.data else None
    X = build_dataset(rng, a.frames, a.crops, scans)
    print(f"dataset {X.shape} built in {time.time() - t0:.1f} s", flush=True)

    params, history = fit(X, rng, a.epochs, a.batch, a.lr, a.seed,
                          cae.load_params(a.init) if a.init else None, out=a.out)

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"loss_total_rec_seg": history, "params": count_params(params), "train_tiles": int(len(X)),
               "data": a.data or "synthetic", "init": a.init,
               "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed,
               "train_time_s": time.time() - t0}, open(a.log, "w"), indent=1)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
