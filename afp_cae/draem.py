"""DRAEM-CAE: discriminatively trained reconstruction anomaly embedding (Zavrtanik et al., ICCV 2021)
built on the CAE of `afp_cae.model`, written in plain JAX.

Reconstructive sub-network  g_phi(f_theta(.)) : the CAE of `afp_cae.model`, unchanged architecture.
Discriminative sub-network  h_psi : (x_a, x_hat) in [0,1]^{64x64x2} -> anomaly logits s in R^{64x64}
    U-Net with skip connections: 3 x 3 convolutions + LeakyReLU, 2 x 2 max pooling, nearest Up x2
    64 (DISC[0] ch.) -> 32 (DISC[1]) -> 16 (DISC[2]) -> 32 -> 64 -> Conv 1x1 -> 1 channel

Training: a defect-free tile x is corrupted by a synthetic anomaly (x_a, M) = T(x) (Perlin-noise mask M,
see `augment`); the CAE learns to restore x from x_a and the U-Net learns to segment M from (x_a, x_hat):
    L = alpha (1 - SSIM(x, x_hat)) + (1 - alpha) ||x - x_hat||_1 + L_focal(sigmoid(s), M),
    x_hat = g_phi(f_theta(x_a)), alpha = 0.85, focal loss with gamma = 2.
T never uses the defect generator of `afp_cae.synth`, so the test defects stay unseen.

Inference: the anomaly map of a tile is sigmoid(s); maps of overlapping tiles are fused, smoothed and
thresholded exactly as the CAE residual map (`afp_cae.pipeline`, eq. 2).
Tensors use the NHWC layout.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import model as cae
from .model import ALPHA, NEG_SLOPE, TILE, ssim_map

DISC = (16, 32, 64)     # U-Net channels at 64, 32 and 16 px
GAMMA = 2.0             # focal-loss focusing parameter
P_ANOM = 0.5            # share of training tiles that receive a synthetic anomaly


# ----------------------------------------------------------------------------- synthetic anomalies
R_MAX = 8   # finest Perlin lattice (cells per tile side); resolutions are 1, 2, 4 or 8


def perlin(key, ry, rx, size=TILE):
    """2-D Perlin noise on a size x size grid with (ry, rx) lattice cells, size % r == 0.

    ry, rx may be traced: gradients are drawn on the finest (R_MAX + 1)^2 lattice and only the
    first (ry + 1) x (rx + 1) of them are used, so the function vmaps over resolutions."""
    ang = jax.random.uniform(key, (R_MAX + 1, R_MAX + 1), maxval=2 * jnp.pi)
    yy = jnp.arange(size)[:, None] * ry / size + jnp.zeros((1, size))
    xx = jnp.arange(size)[None, :] * rx / size + jnp.zeros((size, 1))
    y0, x0 = jnp.floor(yy).astype(int), jnp.floor(xx).astype(int)
    fy, fx = yy - y0, xx - x0

    def dot(dy, dx):
        a = ang[y0 + dy, x0 + dx]
        return jnp.cos(a) * (fy - dy) + jnp.sin(a) * (fx - dx)

    fade = lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    uy, ux = fade(fy), fade(fx)
    top = dot(0, 0) * (1 - ux) + dot(0, 1) * ux
    bot = dot(1, 0) * (1 - ux) + dot(1, 1) * ux
    return jnp.sqrt(2.0) * (top * (1 - uy) + bot * uy)


def _augment_one(key, x, other, p):
    """One tile x[T, T]; `other` is another tile of the batch (source of foreign texture)."""
    k = jax.random.split(key, 10)
    ry, rx = 2 ** jax.random.randint(k[0], (2,), 0, 4)
    reg = perlin(k[1], ry, rx) > 0.5
    r2 = 2 ** jax.random.randint(k[2], (), 1, 4)
    step = x + jax.random.choice(k[3], jnp.array([-1.0, 1.0])) * jax.random.uniform(k[4], (), minval=0.03, maxval=0.30)
    relief = x + jax.random.uniform(k[5], (), minval=0.05, maxval=0.30) * perlin(k[6], r2, r2)
    y = jnp.stack([step, relief, other])[jax.random.randint(k[7], (), 0, 3)]
    beta = jax.random.uniform(k[8], (), maxval=0.8)
    on = (jax.random.uniform(k[9], ()) < p) & (reg.sum() >= 4)
    reg = reg & on
    return jnp.where(reg, jnp.clip(beta * x + (1 - beta) * y, 0, 1), x), reg.astype(x.dtype)


def augment(key, x, p=P_ANOM):
    """Corrupt a batch of defect-free tiles x[N, T, T, 1]. Returns (x_a, M), both [N, T, T, 1].

    For a share p of the tiles the anomaly region is a thresholded Perlin field (resolution 1-8 cells
    per side and axis); inside it the height becomes beta * x + (1 - beta) * y with beta in [0, 0.8]
    and y one of: x shifted by +-0.03-0.30, x plus a smooth Perlin relief of amplitude 0.05-0.30,
    or another tile of the batch (foreign texture). Runs inside jit, on the device."""
    n = x.shape[0]
    k1, k2 = jax.random.split(key)
    other = x[(jnp.arange(n) + jax.random.randint(k1, (n,), 1, n)) % n]
    xa, m = jax.vmap(_augment_one, (0, 0, 0, None))(jax.random.split(k2, n), x[..., 0], other[..., 0], p)
    return xa[..., None], m[..., None]


# ----------------------------------------------------------------------------- parameters
def _he(key, shape, fan_in):
    return jax.random.normal(key, shape) * np.sqrt(2.0 / fan_in)


def _conv_p(key, ci, co, k=3):
    return {"w": _he(key, (k, k, ci, co), k * k * ci), "b": jnp.zeros(co)}


def init_disc(key):
    """He-normal initialisation of the discriminative U-Net."""
    c1, c2, c3 = DISC
    ks = jax.random.split(key, 6)
    return {"e1": _conv_p(ks[0], 2, c1), "e2": _conv_p(ks[1], c1, c2), "e3": _conv_p(ks[2], c2, c3),
            "d2": _conv_p(ks[3], c3 + c2, c2), "d1": _conv_p(ks[4], c2 + c1, c1),
            "out": _conv_p(ks[5], c1, 1, k=1)}


def init_params(key, rec=None):
    """{'rec': CAE parameters, 'disc': U-Net parameters}; `rec` may be a pre-trained CAE."""
    k1, k2 = jax.random.split(key)
    return {"rec": cae.init_params(k1) if rec is None else rec, "disc": init_disc(k2)}


def count_params(p) -> int:
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(p)))


def macs_per_tile() -> int:
    """Multiply-accumulate operations of CAE + U-Net for one 64x64 tile (analytic)."""
    c1, c2, c3 = DISC
    s = TILE
    disc = (s * s * 9 * 2 * c1 + (s // 2) ** 2 * 9 * c1 * c2 + (s // 4) ** 2 * 9 * c2 * c3
            + (s // 2) ** 2 * 9 * (c3 + c2) * c2 + s * s * 9 * (c2 + c1) * c1 + s * s * c1)
    return cae.macs_per_tile() + disc


def save_params(p, path):
    """Save {'rec', 'disc'} to a flat .npz file (no pickle); keys are 'rec.enc.0.w', 'disc.e1.w', ..."""
    flat = {}
    for path_k, v in jax.tree_util.tree_flatten_with_path(p)[0]:
        key = ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path_k)
        flat[key] = np.asarray(v, np.float32)
    np.savez(path, **flat)


def load_params(path, as_jax=True):
    """Load parameters saved by `save_params`."""
    z = np.load(path)
    conv = jnp.asarray if as_jax else np.asarray
    template = init_params(jax.random.PRNGKey(0))
    leaves = []
    for path_k, _ in jax.tree_util.tree_flatten_with_path(template)[0]:
        leaves.append(conv(z[".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path_k)]))
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), leaves)


# ----------------------------------------------------------------------------- network
def _conv(x, layer):
    y = lax.conv_general_dilated(x, layer["w"], (1, 1), "SAME",
                                 dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + layer["b"]


def _act(x):
    return jax.nn.leaky_relu(x, NEG_SLOPE)


def _pool(x):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


def _up(x):
    return jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)


def discriminate(d, xa, xh):
    """h_psi: (N, 64, 64, 1) x 2 -> anomaly logits (N, 64, 64, 1)."""
    e1 = _act(_conv(jnp.concatenate([xa, xh], -1), d["e1"]))
    e2 = _act(_conv(_pool(e1), d["e2"]))
    e3 = _act(_conv(_pool(e2), d["e3"]))
    d2 = _act(_conv(jnp.concatenate([_up(e3), e2], -1), d["d2"]))
    d1 = _act(_conv(jnp.concatenate([_up(d2), e1], -1), d["d1"]))
    return _conv(d1, d["out"])


def forward(p, x):
    """(x_hat, anomaly probability) for tiles x."""
    xh = cae.forward(p["rec"], x)
    return xh, jax.nn.sigmoid(discriminate(p["disc"], x, xh))


# ----------------------------------------------------------------------------- loss
def focal_loss(logits, m, gamma=GAMMA):
    """Binary focal loss (Lin et al., 2017), averaged over pixels."""
    pr = jax.nn.sigmoid(logits)
    pt = jnp.where(m > 0.5, pr, 1 - pr)
    ce = jnp.where(m > 0.5, -jax.nn.log_sigmoid(logits), -jax.nn.log_sigmoid(-logits))
    return ((1 - pt) ** gamma * ce).mean()


def loss_fn(p, x, xa, m, alpha=ALPHA):
    """Reconstruction loss of eq. 1 (restoring clean x from corrupted xa) + focal segmentation loss."""
    xh = cae.forward(p["rec"], xa)
    rec = alpha * (1 - ssim_map(x, xh).mean()) + (1 - alpha) * jnp.abs(x - xh).mean()
    seg = focal_loss(discriminate(p["disc"], xa, xh), m)
    return rec + seg, (rec, seg)


# ----------------------------------------------------------------------------- inference backend
def jax_predictor(params):
    """Backend for `afp_cae.pipeline.anomaly_map(..., kind='map')`: tiles -> anomaly probability."""
    fwd = jax.jit(lambda x: forward(params, x)[1])

    def predict(t):
        return np.asarray(fwd(jnp.asarray(t[..., None])))[..., 0]

    predict.jitted = fwd
    return predict
