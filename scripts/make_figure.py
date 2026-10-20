"""Figure 1 of the abstract: CAE pipeline scheme with real outputs of the trained model.

    python scripts/make_figure.py          # -> results/fig1.png (600 dpi)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import find_objects, label

from afp_cae.model import load_params
from afp_cae.pipeline import DEFAULT, anomaly_map, jax_predictor, preprocess
from afp_cae.synth import add_defect, finish, normal_frame

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 6.6, "mathtext.fontset": "stix"})
INK, INK2 = "#0b0b0b", "#52514e"
BLUE_F, BLUE_E = "#e4effc", "#1c5cab"
GRAY_F, GRAY_E = "#f0efec", "#52514e"
ORNG_F, ORNG_E = "#fdebe3", "#c2481c"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

predict = jax_predictor(load_params(ROOT / "models" / "afp_cae.npz"))
tau = json.load(open(ROOT / "models" / "threshold.json"))["tau"]

# ---- illustrative frame with two defects (foreign object + overlap) in one 25.6 x 25.6 mm window ----
for seed in range(200):
    rng = np.random.default_rng(1000 + seed)
    h, edges = normal_frame(rng)
    h, m1 = add_defect(rng, h, edges, "fod")
    h, m2 = add_defect(rng, h, edges, "overlap")
    c1, c2 = np.nonzero(m1)[1].mean(), np.nonzero(m2)[1].mean()
    if abs(c1 - c2) < 150 and m1.sum() > 300 and m2.sum() > 600:
        frame = preprocess(finish(rng, h)); mask = m1 | m2
        A, recon, _ = anomaly_map(frame, predict, DEFAULT, return_recon=True)
        if (A[m1] > tau).sum() >= 20 and (A[m2] > tau).sum() >= 20:
            break
print("illustration seed", 1000 + seed)
cx = int(np.clip((c1 + c2) / 2 - 128, 0, frame.shape[1] - 256))
sl = np.s_[:, cx:cx + 256]
x_c, A_c, m_c, xh_c = frame[sl], A[sl], mask[sl], recon[sl]

FH = 1.84
fig = plt.figure(figsize=(6.26, FH), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 6.26); ax.set_ylim(0, FH); ax.axis("off")

# ---- top row: pipeline blocks ----
bw, bh, gap, y0 = 0.905, 0.62, 0.148, FH - 0.64
blocks = [
    ("Data acquisition", "laser profilometer\non the layup head,\nheight map 0.1 mm/px", GRAY_F, GRAY_E),
    ("Pre-processing", "3×3 median filter,\ntiles $x$: 64×64 px,\nstride 32 px", GRAY_F, GRAY_E),
    ("Encoder $f_\\theta$", "4 × Conv 3×3 / 2\n+ LeakyReLU, FC\n$z=f_\\theta(x)\\in\\mathbb{R}^{32}$", BLUE_F, BLUE_E),
    ("Decoder $g_\\varphi$", "FC, 4 × (Up×2\n+ Conv 3×3), sigmoid\n$\\hat{x}=g_\\varphi(z)$", BLUE_F, BLUE_E),
    ("Anomaly map", "tile fusion, edge mask,\n$A=G_\\sigma\\ast$ |$x-\\hat{x}$|,\n$\\sigma$ = 3 px", GRAY_F, GRAY_E),
    ("Decision", "$A(u,v)>\\tau$ → defect,\nconnected components\n→ HMI / CNC stop", ORNG_F, ORNG_E),
]
xs_b = [0.03 + k * (bw + gap) for k in range(6)]
for (ttl, body, fc, ec), xb in zip(blocks, xs_b):
    ax.add_patch(FancyBboxPatch((xb, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(xb + bw / 2, y0 + bh - 0.095, ttl, ha="center", va="center", fontsize=7.0, weight="bold", color=INK)
    ax.text(xb + bw / 2, y0 + bh / 2 - 0.075, body, ha="center", va="center", fontsize=6.1,
            color=INK, linespacing=1.15)
for k in range(5):
    ax.add_patch(FancyArrowPatch((xs_b[k] + bw + 0.012, y0 + bh / 2), (xs_b[k + 1] - 0.012, y0 + bh / 2),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK2))
# latent bottleneck marker between encoder and decoder
ax.text((xs_b[2] + bw + xs_b[3]) / 2, y0 + bh / 2 + 0.085, "$z$", ha="center", va="bottom", fontsize=6.6, color=BLUE_E)

# ---- bottom-left: offline training branch ----
tx, ty, th = 0.03, 0.15, (y0 - 0.19) - 0.15
tw = (6.20 - (4 * ((y0 - 0.19) - 0.15) + 3 * 0.08)) - 0.03 - 0.14
ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=0.05",
                            fc="white", ec=BLUE_E, lw=0.7, ls=(0, (3, 2))))
ax.text(tx + tw / 2, ty + th - 0.11, "Offline training (defect-free scans only)", ha="center",
        va="center", fontsize=6.8, weight="bold", color=INK)
ax.text(tx + 0.07, ty + th - 0.21,
        "loss (1): $\\alpha\\,(1-\\mathrm{SSIM})+(1-\\alpha)\\,L_1$, $\\alpha=0.85$\n"
        "Adam, cosine LR, 16 000 tiles, 25 epochs\n"
        "$\\tau$ = max $A$ on defect-free validation\n"
        "scans (zero false alarms on calibration)\n"
        "deployment: ONNX Runtime, CPU, FP32",
        ha="left", va="top", fontsize=6.2, color=INK, linespacing=1.32)
# dashed arrow: trained weights -> encoder
ax.add_patch(FancyArrowPatch((tx + tw * 0.55, ty + th), (xs_b[2] + bw / 2, y0 - 0.01),
                             arrowstyle="-|>", mutation_scale=5, lw=0.6, color=BLUE_E, ls=(0, (3, 2)),
                             connectionstyle="arc3,rad=-0.15"))
ax.text(xs_b[2] + bw / 2 + 0.06, y0 - 0.11, "$\\theta^*,\\varphi^*$", fontsize=6.4, color=BLUE_E, ha="left", va="center")

# ---- bottom-right: real model outputs ----
pg = 0.08
ptop = y0 - 0.19
ph = pw = ptop - 0.15
px0 = 6.20 - (4 * pw + 3 * pg)
panels = [(x_c, "gray", (0, 1), "(a) input height map $x$"),
          (xh_c, "gray", (0, 1), "(b) reconstruction $\\hat{x}$"),
          (A_c, SEQ, (0, max(A_c.max(), tau * 1.2)), "(c) anomaly map $A$"),
          (x_c, "gray", (0, 1), "(d) detected defects")]
for k, (img, cm, (vmin, vmax), cap) in enumerate(panels):
    a = fig.add_axes([(px0 + k * (pw + pg)) / 6.26, (ptop - ph) / FH, pw / 6.26, ph / FH])
    a.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5); s.set_color(INK2)
    if k == 2:
        a.contour(A_c > tau, [0.5], colors=[ORNG_E], linewidths=0.5)
    if k == 3:
        lab, n = label(A_c > tau)
        for c, sl_ in enumerate(find_objects(lab), 1):
            if (lab[sl_] == c).sum() < 20:
                continue
            ys_, xs_ = sl_
            a.add_patch(Rectangle((xs_.start - 2, ys_.start - 2), xs_.stop - xs_.start + 3,
                                  ys_.stop - ys_.start + 3, fill=False, ec=ORNG_E, lw=0.9))
        a.contour(m_c, [0.5], colors=["#1baf7a"], linewidths=0.45, linestyles="dotted")
    ax.text(px0 + k * (pw + pg) + pw / 2, ptop - ph - 0.075, cap, ha="center", va="center", fontsize=6.3, color=INK)
ax.text(px0 + 4 * (pw + pg) - pg, ptop + 0.025, "25.6 × 25.6 mm window, 4 of 12 tows",
        ha="right", va="bottom", fontsize=5.8, color=INK2, style="italic")
out = ROOT / "results" / "fig1.png"
fig.savefig(out, dpi=600)
print("saved", out)
