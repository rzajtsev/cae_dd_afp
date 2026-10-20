"""Render one synthetic frame per defect type (results/samples.png)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afp_cae.synth import DEFECT_TYPES, make_frame


def main(out=ROOT / "results" / "samples.png", seed=0):
    rng = np.random.default_rng(seed)
    kinds = [None] + DEFECT_TYPES
    fig, ax = plt.subplots(4, 2, figsize=(12, 8.2))
    for a, k in zip(ax.T.ravel(), kinds):
        f, m = make_frame(rng, k)
        a.imshow(f, cmap="gray", vmin=0, vmax=1)
        if m.any():
            a.contour(m, [0.5], colors=["#c2481c"], linewidths=0.7)
        a.set_title(k.replace("_", " ") if k else "defect-free", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Synthetic 12-tow AFP height maps, 76.8 x 25.6 mm, 0.1 mm/px (outline: ground-truth mask)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
