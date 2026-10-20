"""Synthetic height-map generator emulating laser-line profilometer scans of a
12-tow dry carbon-tape AFP (automated fiber placement) course.

The generator is used both for training (defect-free frames only) and for testing
(frames with exactly one injected defect of a known type and a pixel-exact mask).

Geometry (pixel pitch 0.1 mm):
  * frame: W = 768 px across the course (12 tows x 6.35 mm + margins), H = 256 px
    along the layup direction (25.6 mm of material per frame)
  * tow width 63.5 px, nominal inter-tow gap 0-3 px
Output: float32 height map normalised to [0, 1] and a binary ground-truth defect mask.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, rotate

PX_MM = 0.1
TOW_W = 63.5
N_TOWS = 12
W, H = 768, 256
DEFECT_TYPES = ["gap", "overlap", "twist", "fod", "fuzzball", "wrinkle", "missing_tow"]


def fiber_texture(rng, h, w):
    """Anisotropic noise elongated along the layup (row) direction."""
    n = rng.standard_normal((h, w)).astype(np.float32)
    t = gaussian_filter(n, sigma=(9.0, 0.7))
    t /= t.std() + 1e-6
    return t


def normal_frame(rng):
    """Defect-free course: 12 crowned tows with steering drift, fibre texture and substrate level."""
    hmap = np.zeros((H, W), np.float32)
    x0 = (W - N_TOWS * TOW_W) / 2 + rng.uniform(-3, 3)
    cols = np.arange(W, dtype=np.float32)[None, :]
    rows = np.arange(H, dtype=np.float32)[:, None]
    edges = []
    for k in range(N_TOWS):
        # slight steering / lateral wander and per-tow thickness variation
        drift = rng.uniform(-0.004, 0.004) * rows + rng.uniform(-0.6, 0.6) * np.sin(
            rows / rng.uniform(60, 140) + rng.uniform(0, 6.28))
        gap = rng.uniform(0.0, 3.0)
        left = x0 + k * TOW_W + gap / 2 + drift
        right = x0 + (k + 1) * TOW_W - gap / 2 + drift
        edges.append((left, right))
        inside = 1.0 / (1 + np.exp(np.clip(-(cols - left) * 2.5, -50, 50))) * 1.0 / (1 + np.exp(np.clip((cols - right) * 2.5, -50, 50)))
        thick = 0.62 + rng.uniform(-0.03, 0.03)
        # crowned cross-section of a tow
        u = np.clip((cols - left) / (right - left), 0, 1)
        crown = 0.05 * (1 - (2 * u - 1) ** 2)
        hmap += inside * (thick + crown)
    tex = fiber_texture(rng, H, W)
    hmap += 0.035 * tex * (hmap > 0.3)
    hmap += 0.12  # substrate / previous ply level
    return hmap, edges


def add_defect(rng, hmap, edges, kind):
    """Inject one defect of type `kind` (see DEFECT_TYPES) in place; returns (hmap, mask)."""
    mask = np.zeros_like(hmap, dtype=bool)
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    L = int(rng.uniform(40, 200))
    r0 = int(rng.uniform(0, H - L))
    r1 = r0 + L
    k = int(rng.integers(1, N_TOWS - 1))
    left, right = edges[k]
    lc = float(np.mean(left[r0:r1]))
    rc = float(np.mean(right[r0:r1]))
    if kind == "gap":  # widened inter-tow gap 0.8-2.0 mm
        g = rng.uniform(8, 20)
        band = (cols > rc - g / 2) & (cols < rc + g / 2)
        sel = np.zeros_like(mask); sel[r0:r1] = True
        reg = band & sel
        hmap[reg] = 0.12 + 0.01 * rng.standard_normal(reg.sum())
        mask |= reg
    elif kind == "overlap":  # tow edge riding on neighbour, 1-3 mm
        g = rng.uniform(10, 30)
        band = (cols > rc - g) & (cols < rc)
        sel = np.zeros_like(mask); sel[r0:r1] = True
        reg = band & sel
        hmap[reg] += rng.uniform(0.15, 0.30)
        mask |= reg
    elif kind == "twist":  # rotated texture + local bulge over a tow
        Lt = int(rng.uniform(40, 90)); r0 = int(rng.uniform(0, H - Lt)); r1 = r0 + Lt
        c0, c1 = int(lc), int(rc)
        patch = hmap[r0:r1, c0:c1].copy()
        rot = rotate(patch, angle=rng.uniform(25, 70), reshape=False, mode="reflect")
        yy, xx = np.mgrid[0:r1 - r0, 0:c1 - c0]
        bump = rng.uniform(0.08, 0.2) * np.exp(-(((yy - (r1 - r0) / 2) / ((r1 - r0) / 3)) ** 2))
        hmap[r0:r1, c0:c1] = rot + bump
        mask[r0:r1, c0:c1] = True
    elif kind == "fod":  # foreign object / backing-paper fragment
        cy, cx = rng.uniform(20, H - 20), rng.uniform(lc, rc)
        ry, rx = rng.uniform(6, 25), rng.uniform(6, 25)
        ang = rng.uniform(0, np.pi)
        dy, dx = rows - cy, cols - cx
        u = (dx * np.cos(ang) + dy * np.sin(ang)) / rx
        v = (-dx * np.sin(ang) + dy * np.cos(ang)) / ry
        reg = (u ** 2 + v ** 2) < 1
        hmap[reg] = hmap[reg].mean() + rng.uniform(0.06, 0.20)  # flat, no fibre texture
        mask |= reg
    elif kind == "fuzzball":  # accumulations of broken filaments
        n = int(rng.integers(3, 8))
        cy, cx = rng.uniform(20, H - 20), rng.uniform(lc, rc)
        blob = np.zeros_like(hmap)
        for _ in range(n):
            y, x = int(np.clip(cy + rng.normal(0, 6), 0, H - 1)), int(np.clip(cx + rng.normal(0, 6), 0, W - 1))
            blob[y, x] = 1
        blob = gaussian_filter(blob, 2.5)
        blob /= blob.max()
        hmap += rng.uniform(0.15, 0.35) * blob
        mask |= blob > 0.15
    elif kind == "wrinkle":  # transverse ridge (pucker) across one or two tows
        cy = rng.uniform(30, H - 30)
        span = int(rng.integers(1, 3))
        c0, c1 = lc, edges[min(k + span - 1, N_TOWS - 1)][1][int(cy)]
        ridge = rng.uniform(0.10, 0.25) * np.exp(-((rows - cy) / rng.uniform(3, 6)) ** 2) * ((cols > c0) & (cols < c1))
        hmap += ridge.astype(np.float32)
        mask |= ridge > 0.05
    elif kind == "missing_tow":  # tow cut early / not fed
        sel = np.zeros_like(mask); sel[r0:r1] = True
        reg = sel & (cols > lc) & (cols < rc)
        hmap[reg] = 0.12 + 0.01 * rng.standard_normal(reg.sum())
        mask |= reg
    return hmap, mask


def finish(rng, hmap):
    """Add sensor noise and laser-triangulation dropouts, clip to [0, 1]."""
    hmap = hmap + 0.012 * rng.standard_normal(hmap.shape).astype(np.float32)  # sensor noise
    # occasional dropout pixels typical for laser triangulation
    drop = rng.random(hmap.shape) < 0.0008
    hmap[drop] = 0.0
    return np.clip(hmap, 0, 1).astype(np.float32)


def make_frame(rng, defect=None):
    """Return (height_map float32 [H, W] in [0, 1], defect_mask bool [H, W])."""
    hmap, edges = normal_frame(rng)
    mask = np.zeros_like(hmap, dtype=bool)
    if defect is not None:
        hmap, mask = add_defect(rng, hmap, edges, defect)
    return finish(rng, hmap), mask
