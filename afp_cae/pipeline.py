"""Inference pipeline: pre-processing, tiling, CAE reconstruction, anomaly map and decision rule.

    A = G_sigma * |x - x_hat|                    (residuals of overlapping tiles averaged)
    D(u, v) = 1[A(u, v) > tau]                   (eq. 2 of the abstract)
    tau = max_{x in V0, (u, v)} A(u, v)          (V0: defect-free validation frames)

Connected components of D with area >= min_area are reported as defects.

The reconstruction backend is any callable `predict(tiles) -> recon`, where both arrays have
shape (N, tile, tile); `jax_predictor` and `onnx_predictor` provide the two backends used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label, median_filter


@dataclass(frozen=True)
class PipelineConfig:
    tile: int = 64          # px, CAE input size
    stride: int = 32        # px, tiling stride (50 % overlap)
    sigma: float = 3.0      # px, Gaussian smoothing of the fused residual map
    edge: int = 12          # px, excluded band at the outer course edges (1.2 mm)
    min_area: int = 20      # px, minimal connected-component area (0.2 mm^2)
    px_mm: float = 0.1      # mm per pixel


DEFAULT = PipelineConfig()


def normalize_scan(arr, lo_pct=0.1, hi_pct=99.9):
    """Robust min-max scaling of a raw height map to [0, 1] (percentiles ignore outliers/dropouts)."""
    arr = np.asarray(arr, np.float32)
    lo, hi = np.nanpercentile(arr, [lo_pct, hi_pct])
    return np.clip((np.nan_to_num(arr, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def load_scans(folder, normalize=False):
    """Load all 2-D height maps (*.npy, 0.1 mm/px, rows along the layup direction) from a folder."""
    from pathlib import Path
    files = sorted(Path(folder).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no *.npy height maps in {folder}")
    scans = []
    for f in files:
        a = np.load(f)
        if a.ndim != 2:
            raise ValueError(f"{f}: expected a 2-D height map, got shape {a.shape}")
        scans.append(normalize_scan(a) if normalize else a.astype(np.float32))
    return scans


def preprocess(frame):
    """3x3 median filter: removes isolated dropout pixels of laser triangulation."""
    return median_filter(frame, size=3)


def tiles(frame, size=64, stride=32):
    """Cut a frame into overlapping square tiles. Returns (tiles[N, size, size], ys, xs)."""
    h, w = frame.shape
    ys = list(range(0, h - size + 1, stride))
    xs = list(range(0, w - size + 1, stride))
    out = np.stack([frame[y:y + size, x:x + size] for y in ys for x in xs])
    return out, ys, xs


def fuse(per_tile, shape, ys, xs, size):
    """Average per-tile maps back into a full-frame map (overlaps are averaged)."""
    acc = np.zeros(shape, np.float32)
    cnt = np.zeros(shape, np.float32)
    i = 0
    for y in ys:
        for x in xs:
            acc[y:y + size, x:x + size] += per_tile[i]
            cnt[y:y + size, x:x + size] += 1
            i += 1
    return acc / np.maximum(cnt, 1)


def anomaly_map(frame, predict, cfg: PipelineConfig = DEFAULT, return_recon=False):
    """Smoothed residual map A for a pre-processed frame (float32 [H, W] in [0, 1])."""
    t, ys, xs = tiles(frame, cfg.tile, cfg.stride)
    xh = predict(t)
    A = gaussian_filter(fuse(np.abs(t - xh), frame.shape, ys, xs, cfg.tile), cfg.sigma)
    if cfg.edge:
        A[:, :cfg.edge] = 0
        A[:, -cfg.edge:] = 0
    if return_recon:
        return A, fuse(xh, frame.shape, ys, xs, cfg.tile), (t, ys, xs)
    return A


def tile_scores(A, ys, xs, size=64):
    """Tile-level anomaly score: maximum of A inside each tile window."""
    return np.array([A[y:y + size, x:x + size].max() for y in ys for x in xs])


def detect(A, tau, cfg: PipelineConfig = DEFAULT):
    """Apply the decision rule and return a list of detected defects (largest first)."""
    lab, _ = label(A > tau)
    found = []
    for c, sl in enumerate(find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == c
        area = int(comp.sum())
        if area < cfg.min_area:
            continue
        ys, xs = sl
        found.append({
            "bbox_px": [int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)],   # x0, y0, x1, y1
            "bbox_mm": [round(v * cfg.px_mm, 2) for v in (xs.start, ys.start, xs.stop, ys.stop)],
            "area_mm2": round(area * cfg.px_mm ** 2, 3),
            "peak_score": float(A[sl][comp].max()),
        })
    return sorted(found, key=lambda d: -d["area_mm2"])


# ----------------------------------------------------------------------------- backends
def jax_predictor(params):
    """JAX/XLA backend (CPU or GPU, whichever JAX is installed for)."""
    import jax
    import jax.numpy as jnp
    from .model import forward

    fwd = jax.jit(lambda x: forward(params, x))

    def predict(t):
        return np.asarray(fwd(jnp.asarray(t[..., None])))[..., 0]

    predict.jitted = fwd
    return predict


def onnx_predictor(path, threads=None):
    """ONNX Runtime backend (CPUExecutionProvider)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def predict(t):
        return sess.run(None, {"x": np.ascontiguousarray(t[:, None], np.float32)})[0][:, 0]

    predict.session = sess
    return predict
