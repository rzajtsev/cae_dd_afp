"""Real laser-profilometer scans of Cardoso et al. (2026), Zenodo 10.5281/zenodo.17048470 (CC BY 4.0):
loading, conversion to the 0.1 mm/px grid of the CAE, and synthetic defects injected in millimetres.

Dataset: 2 layers x 3 passes of a KUKA AFP cell; each pass has laser_scan_x_*.csv (2048 points across),
laser_scan_z_*.csv (one profile per row) and robot_path_*.csv. A pass becomes a height map z in mm
relative to the fitted substrate plane, with pitch dx (~0.104 mm) across and dy (~0.35 mm) along the layup.

The CAE works on [0, 1] heights at 0.1 mm/px. `Grid` resamples a scan to that grid and scales heights with
one range for all passes: the (0.1, 99.9) percentiles of the training heights widened by 25 % on each
side (room for defects above and below); per-scan `normalize_scan` would rescale every scan differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .pipeline import preprocess

COL_STEP = 8          # 2048 points -> 256 columns (~0.1 mm/px across)
DROPOUT_Z = 50.0      # z below this is a sensor dropout
COURSE_START_MM = {"Layer_01": 22.0, "Layer_02": 3.0}   # cut / feed zone at the course start, not inspected
END_MARGIN_MM = 10.0
PX_MM = 0.1
RANGE_PCT = (0.1, 99.9)
RANGE_MARGIN = 0.25


@dataclass
class HeightMap:
    name: str
    z: np.ndarray       # (profiles, W) height relative to the substrate plane, mm
    valid: np.ndarray   # pixels with a sensor reading
    dx: float           # mm/px across the course
    dy: float           # mm/profile along the layup


# ----------------------------------------------------------------------------- loading
def pass_dirs(root):
    return sorted(Path(root).glob("PART-*/Layer_*/Pass_*"))


def load_heightmap(pass_dir: Path, cache=None) -> HeightMap:
    """CSV of one pass -> height map (column-binned, gaps filled, plane removed). Cached as .npz."""
    import pandas as pd

    name = f"{pass_dir.parent.name}_{pass_dir.name}"
    cf = Path(cache) / f"{name}.npz" if cache else None
    if cf is not None and cf.exists():
        d = np.load(cf)
        return HeightMap(name, d["z"], d["valid"], float(d["dx"]), float(d["dy"]))

    x = pd.read_csv(next(pass_dir.glob("laser_scan_x_*.csv"))).to_numpy(float).ravel()
    zdf = pd.read_csv(next(pass_dir.glob("laser_scan_z_*.csv")))
    zdf.pop("timestamp_iso")
    z = zdf.to_numpy(float)
    robot = pd.read_csv(next(pass_dir.glob("robot_path_*.csv")))
    n_profiles = len(z)

    z = np.where(z < DROPOUT_Z, np.nan, z)
    if x[1] < x[0]:   # the scanner X axis decreases
        x, z = x[::-1], z[:, ::-1]
    rows = np.where(np.mean(np.isfinite(z), axis=1) > 0.9)[0]   # drop profiles past the end of the course
    z = z[rows.min():rows.max() + 1]

    n = z.shape[1] // COL_STEP * COL_STEP
    with np.errstate(all="ignore"):
        zc = np.nanmedian(z[:, :n].reshape(z.shape[0], -1, COL_STEP), axis=2)
    valid = np.isfinite(zc)
    if (~valid).any():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        zc = zc[tuple(idx)]
    zc = ndimage.median_filter(zc, size=3)

    # robust plane fit (scanner / tool tilt)
    yy, xx = np.mgrid[:zc.shape[0], :zc.shape[1]]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    b = zc.ravel()
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(3):
        r = b - A @ coef
        keep = np.abs(r) < 2.5 * np.median(np.abs(r)) + 1e-6
        coef = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
    zc = (b - A @ coef).reshape(zc.shape)
    zc -= np.median(zc)

    travel = robot[["robot_X", "robot_Y"]].to_numpy()
    dy = float(np.sum(np.linalg.norm(np.diff(travel, axis=0), axis=1))) / n_profiles
    dx = float(abs(np.median(np.diff(x))) * COL_STEP)
    m = HeightMap(name, zc.astype(np.float32), valid, dx, dy)
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, z=m.z, valid=valid, dx=dx, dy=dy)
    return m


def crop_to_tape(m: HeightMap) -> HeightMap:
    """Keep the part of the pass covered by tape (without the course start and end zones)."""
    on_tape = np.where(np.median(m.z, axis=1) > -1.0)[0]
    start = int(COURSE_START_MM[m.name.split("_Pass")[0]] / m.dy)
    stop = on_tape.max() - int(END_MARGIN_MM / m.dy)
    return HeightMap(m.name, m.z[start:stop].copy(), m.valid[start:stop].copy(), m.dx, m.dy)


def load_all(root, cache=None):
    """All passes of the dataset, cropped to the tape. Names: 'Layer_01_Pass_01', ..."""
    return [crop_to_tape(load_heightmap(p, cache)) for p in pass_dirs(root)]


# ----------------------------------------------------------------------------- tow seams
SEAM_SLOPE = 0.3          # mm/mm: across-course slope that marks a seam edge (flat tape ~0.01, seam edge 0.5-1)
SEAM_SIGMA_MM = (3.0, 0.15)   # smoothing (along, across) before the slope, mm
SEAM_GROW_MM = 0.5        # the seam zone extends this far across on both sides


def seam_mask(z, dx, dy, slope=SEAM_SLOPE, sigma_mm=SEAM_SIGMA_MM, grow_mm=SEAM_GROW_MM):
    """Tow seams and tape edges of a defect-free height map z (mm): pixels where the across-course slope of z,
    smoothed along the layup, exceeds `slope`, grown by `grow_mm` across. Follows slanted seams row by row."""
    zs = ndimage.gaussian_filter(np.asarray(z, np.float64), (sigma_mm[0] / dy, sigma_mm[1] / dx))
    edge = np.abs(np.gradient(zs, dx, axis=1)) > slope
    r = max(int(round(grow_mm / dx)), 0)
    return ndimage.binary_dilation(edge, np.ones((1, 2 * r + 1), bool)) if r else edge


# ----------------------------------------------------------------------------- tape boundaries for exclusion
# Three detectors of tow seams / tape edges on a defect-free scan, compared in scripts/eval_boundaries.py.
# All work on the isotropic 0.1 mm/px grid (on the scan grid a pixel is ~0.104 x 0.35 mm, which would distort
# gradients and Hough angles), mark edges with the same slope SEAM_SLOPE, grow them by SEAM_GROW_MM across
# and map the zone back to the scan grid.
CANNY_SIGMA_PX = 2.0      # Gaussian of the Canny detector, px of 0.1 mm
EDGE_ALONG = 2.0          # keep edges whose across-course gradient is this many times the along-course one
HOUGH_ANGLE_DEG = 15.0    # Hough lines may deviate this much from the layup direction
HOUGH_MIN_LEN_MM = 10.0   # minimal Hough segment length
HOUGH_GAP_MM = 3.0        # maximal gap inside a Hough segment


def _iso(z, dx, dy):
    """Heights in mm on the scan grid -> the same heights on the 0.1 mm/px grid."""
    h, w = z.shape
    H, W = round(h * dy / PX_MM), round(w * dx / PX_MM)
    yy, xx = np.meshgrid(np.arange(H) * PX_MM / dy, np.arange(W) * PX_MM / dx, indexing="ij")
    return ndimage.map_coordinates(np.asarray(z, np.float64), [yy, xx], order=1, mode="nearest")


def _zone(edges, shape, dx, dy, grow_mm=SEAM_GROW_MM):
    """Edges on the 0.1 mm/px grid -> zone grown by grow_mm across, on the scan grid of `shape`."""
    r = max(int(round(grow_mm / PX_MM)), 0)
    zone = ndimage.binary_dilation(edges, np.ones((1, 2 * r + 1), bool)) if r else edges
    return Grid.from_grid(zone.astype(np.float32), shape, dx, dy) > 0.5


def boundary_sobel(z, dx, dy, slope=SEAM_SLOPE, sigma_mm=SEAM_SIGMA_MM):
    """Sobel along x: |Sobel_x| of the map smoothed along the layup > slope (mm/mm)."""
    zi = ndimage.gaussian_filter(_iso(z, dx, dy), (sigma_mm[0] / PX_MM, sigma_mm[1] / PX_MM))
    gx = ndimage.sobel(zi, axis=1) / (8 * PX_MM)          # Sobel kernel weights sum to 8 -> mm per mm
    return _zone(np.abs(gx) > slope, z.shape, dx, dy)


def _canny_edges(zi, slope=SEAM_SLOPE):
    """Canny edges of the 0.1 mm/px map that run along the layup (thin, non-maximum suppressed)."""
    from skimage.feature import canny

    t = 8 * PX_MM * slope       # canny thresholds the raw ndimage.sobel magnitude: ~8 x height step per px
    edges = canny(zi, sigma=CANNY_SIGMA_PX, low_threshold=t / 2, high_threshold=t)
    zs = ndimage.gaussian_filter(zi, CANNY_SIGMA_PX)
    return edges & (np.abs(ndimage.sobel(zs, axis=1)) > EDGE_ALONG * np.abs(ndimage.sobel(zs, axis=0)))


def boundary_canny(z, dx, dy, slope=SEAM_SLOPE):
    """Canny edges running along the layup (hysteresis between slope / 2 and slope)."""
    return _zone(_canny_edges(_iso(z, dx, dy), slope), z.shape, dx, dy)


def boundary_hough(z, dx, dy, slope=SEAM_SLOPE):
    """Straight segments found by the probabilistic Hough transform on the Canny edges, within
    +-HOUGH_ANGLE_DEG of the layup direction and at least HOUGH_MIN_LEN_MM long."""
    from skimage.draw import line
    from skimage.transform import probabilistic_hough_line

    edges = _canny_edges(_iso(z, dx, dy), slope)
    theta = np.deg2rad(np.linspace(-HOUGH_ANGLE_DEG, HOUGH_ANGLE_DEG, 61))   # theta = 0: line along the layup
    segs = probabilistic_hough_line(edges, threshold=10, line_length=int(HOUGH_MIN_LEN_MM / PX_MM),
                                    line_gap=int(HOUGH_GAP_MM / PX_MM), theta=theta, rng=0)
    out = np.zeros_like(edges)
    for (x0, y0), (x1, y1) in segs:
        rr, cc = line(y0, x0, y1, x1)
        out[rr, cc] = True
    return _zone(out, z.shape, dx, dy)


BOUNDARIES = {"sobel_x": boundary_sobel, "canny": boundary_canny, "canny_hough": boundary_hough}


# ----------------------------------------------------------------------------- 0.1 mm/px grid
class Grid:
    """Height range and resampling shared by all passes of one training set."""

    def __init__(self, maps):
        heights = np.concatenate([m.z[m.valid & np.isfinite(m.z)] for m in maps])
        lo, hi = np.percentile(heights, RANGE_PCT)
        pad = RANGE_MARGIN * (hi - lo)
        self.lo, self.hi = float(lo - pad), float(hi + pad)

    def to_grid(self, z, dx, dy):
        """Heights in mm on the (dy, dx) grid -> median-filtered [0, 1] map at 0.1 mm/px."""
        z = np.asarray(z, np.float32)
        h, w = z.shape
        H, W = round(h * dy / PX_MM), round(w * dx / PX_MM)
        yy, xx = np.meshgrid(np.arange(H) * PX_MM / dy, np.arange(W) * PX_MM / dx, indexing="ij")
        zr = ndimage.map_coordinates(z, [yy, xx], order=1, mode="nearest")
        return preprocess(np.clip((zr - self.lo) / (self.hi - self.lo), 0, 1).astype(np.float32))

    @staticmethod
    def from_grid(A, shape, dx, dy):
        """Map on the 0.1 mm/px grid -> the scan grid: pixel (i, j) lies at (i dy, j dx) mm."""
        h, w = shape
        yy, xx = np.meshgrid(np.arange(h) * dy / PX_MM, np.arange(w) * dx / PX_MM, indexing="ij")
        return ndimage.map_coordinates(A, [yy, xx], order=1, mode="nearest").astype(np.float32)


def tiles_dataset(rng, grids, n_tiles, tile=64):
    """Random tiles, an equal number from each 0.1 mm/px map (as scripts/train.py). Returns X[N, t, t, 1]."""
    per = n_tiles // len(grids)
    X = np.empty((per * len(grids), tile, tile, 1), np.float32)
    i = 0
    for g in grids:
        for _ in range(per):
            y = rng.integers(0, g.shape[0] - tile + 1)
            x = rng.integers(0, g.shape[1] - tile + 1)
            X[i, :, :, 0] = g[y:y + tile, x:x + tile]
            i += 1
    return X


# ----------------------------------------------------------------------------- defects in millimetres
MASK_EPS = 0.03   # mm: ground truth is |height increment| > MASK_EPS
DEFECT_TYPES = ("gap", "overlap", "fod", "twist", "wrinkle")
SEAM_TYPES = ("gap", "overlap")   # types that `inject` may centre on a tow seam


@dataclass
class Defect:
    kind: str
    size: float         # main size parameter: width (gap, overlap) or height / amplitude, mm
    mask: np.ndarray
    on_seam: bool = False   # more than half of the mask lies in the seam zone (when the zone is given)


def _smoothstep(d, edge):
    """1 inside (d < 0), 0 outside, smooth edge of width `edge` mm."""
    return 1.0 / (1.0 + np.exp(np.clip(d / max(edge, 1e-3) * 4, -30, 30)))


def make_defect(kind, shape, dx, dy, rng, center=None):
    """Height increment (mm) of one defect; geometry in mm; center = (row, column) or None for a random one."""
    h, w = shape
    if center is None:
        cy, cx = rng.uniform(0.12, 0.88) * h, rng.uniform(0.1, 0.9) * w
    else:
        cy, cx = center
    yy, xx = np.mgrid[:h, :w]
    X, Y = (xx - cx) * dx, (yy - cy) * dy
    if kind == "gap":
        width, depth, length = rng.choice([0.3, 0.5, 0.8, 1.2, 2.0]), rng.uniform(0.15, 0.3), rng.uniform(8, 25)
        prof = _smoothstep(np.abs(X) - width / 2, 0.05) * _smoothstep(np.abs(Y) - length / 2, 0.5)
        dz, size = -depth * prof, width
    elif kind == "overlap":
        width, height, length = rng.choice([0.3, 0.5, 1.0, 2.0, 3.0]), rng.uniform(0.15, 0.3), rng.uniform(8, 25)
        prof = _smoothstep(np.abs(X) - width / 2, 0.05) * _smoothstep(np.abs(Y) - length / 2, 0.5)
        dz, size = height * prof, width
    elif kind == "fod":
        height, diam = rng.choice([0.05, 0.1, 0.15, 0.25, 0.4]), rng.uniform(1.0, 4.0)
        r = np.hypot(X, Y * rng.uniform(0.6, 1.4))
        dz, size = height * _smoothstep(r - diam / 2, 0.3), height
    elif kind == "twist":
        height = rng.choice([0.05, 0.1, 0.15, 0.25, 0.4])
        length, width = rng.uniform(6, 15), rng.uniform(1.5, 3.5)
        a = np.deg2rad(rng.uniform(15, 45) * rng.choice([-1, 1]))
        u, v = X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)
        core = np.exp(-(u / (length / 2)) ** 4 - (v / (width / 2)) ** 2)
        flank = np.exp(-(u / (length / 2)) ** 4 - ((np.abs(v) - width) / (width / 2)) ** 2)
        dz, size = height * (core - 0.4 * flank), height
    elif kind == "wrinkle":
        amp, span, along = rng.choice([0.05, 0.1, 0.15, 0.25, 0.4]), rng.uniform(4, 12), rng.uniform(2, 5)
        prof = np.exp(-(Y / (along / 2)) ** 2) * _smoothstep(np.abs(X) - span / 2, 0.8)
        dz, size = amp * prof, amp
    else:
        raise ValueError(kind)
    return dz.astype(np.float32), float(size)


def _seam_centers(seam):
    """Seam-zone pixels allowed as a defect centre (the same bounds as for a random centre)."""
    h, w = seam.shape
    ys, xs = np.nonzero(seam)
    keep = (ys >= 0.12 * h) & (ys <= 0.88 * h) & (xs >= 0.1 * w) & (xs <= 0.9 * w)
    return ys[keep], xs[keep]


def inject(z, dx, dy, rng, per_type=2, types=DEFECT_TYPES, max_tries=50, seam=None, seam_frac=0.5,
           seam_types=SEAM_TYPES):
    """Add per_type non-overlapping defects of every type. Returns (z, ground-truth mask, defects).

    seam: seam zone of the clean map (`seam_mask`); a defect of a type in seam_types is then centred on a random
    seam pixel with probability seam_frac, and every defect gets on_seam. Without seam (or with seam_frac = 0)
    the defects and the random-number sequence are exactly those of the plain call."""
    z = z.copy()
    occupied = np.zeros(z.shape, bool)
    defects = []
    sy, sx = _seam_centers(seam) if seam is not None else (None, None)
    for kind in types:
        for _ in range(per_type):
            to_seam = (seam is not None and seam_frac > 0 and kind in seam_types and len(sy) > 0
                       and rng.random() < seam_frac)
            for _ in range(max_tries):
                center = None
                if to_seam:
                    i = rng.integers(len(sy))
                    center = (float(sy[i]), float(sx[i]))
                dz, size = make_defect(kind, z.shape, dx, dy, rng, center)
                mask = np.abs(dz) > MASK_EPS
                if mask.sum() < 3:
                    continue
                ys, xs = np.where(mask)
                grown = np.zeros_like(mask)
                grown[max(ys.min() - 15, 0):ys.max() + 16, max(xs.min() - 15, 0):xs.max() + 16] = True
                if (grown & occupied).any():
                    continue
                occupied |= grown
                z += dz
                defects.append(Defect(kind, size, mask, bool(seam[mask].mean() > 0.5) if seam is not None else False))
                break
    gt = np.zeros(z.shape, bool)
    for d in defects:
        gt |= d.mask
    return z, gt, defects
