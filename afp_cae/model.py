"""Convolutional autoencoder (CAE) for unsupervised AFP defect detection, written in plain JAX.

Encoder  f_theta : x in [0,1]^{64x64}  ->  z in R^32
    4 x (Conv 3x3, stride 2, LeakyReLU)   64 -> 32 -> 16 -> 8 -> 4
    fully connected layer                 4*4*64 -> 32
Decoder  g_phi   : z -> x_hat
    fully connected layer + LeakyReLU     32 -> 4*4*64
    4 x (nearest Up x2, Conv 3x3, LeakyReLU)  4 -> 8 -> 16 -> 32 -> 64
    Conv 3x3 -> 1 channel, sigmoid

Training objective (eq. 1 of the abstract), over N defect-free tiles:
    (theta*, phi*) = argmin 1/N sum_i [ alpha (1 - SSIM(x_i, x_hat_i)) + (1 - alpha) ||x_i - x_hat_i||_1 ]
with alpha = 0.85.

Tensors use the NHWC layout: x has shape (N, 64, 64, 1).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

TILE = 64
LATENT = 32
ENC = ((1, 16), (16, 32), (32, 64), (64, 64))   # (in, out) channels of the stride-2 convolutions
DEC = ((64, 64), (64, 32), (32, 16), (16, 16))   # (in, out) channels of the up-sampling stages
BOTTOM = TILE // 2 ** len(ENC)                   # spatial size at the bottleneck (4)
ALPHA = 0.85                                     # SSIM weight in the loss
NEG_SLOPE = 0.1                                  # LeakyReLU slope


# ----------------------------------------------------------------------------- parameters
def _he(key, shape, fan_in):
    return jax.random.normal(key, shape) * np.sqrt(2.0 / fan_in)


def init_params(key):
    """He-normal initialisation. Returns a pytree (dict of lists of dicts of arrays)."""
    ks = iter(jax.random.split(key, 32))
    c_bot = ENC[-1][1]
    flat = BOTTOM * BOTTOM * c_bot
    p = {"enc": [], "dec": []}
    for ci, co in ENC:
        p["enc"].append({"w": _he(next(ks), (3, 3, ci, co), 9 * ci), "b": jnp.zeros(co)})
    p["fc_e"] = {"w": _he(next(ks), (flat, LATENT), flat), "b": jnp.zeros(LATENT)}
    p["fc_d"] = {"w": _he(next(ks), (LATENT, flat), LATENT), "b": jnp.zeros(flat)}
    for ci, co in DEC:
        p["dec"].append({"w": _he(next(ks), (3, 3, ci, co), 9 * ci), "b": jnp.zeros(co)})
    p["out"] = {"w": _he(next(ks), (3, 3, DEC[-1][1], 1), 9 * DEC[-1][1]) * 0.1, "b": jnp.zeros(1)}
    return p


def count_params(p) -> int:
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(p)))


def macs_per_tile() -> int:
    """Multiply-accumulate operations for one 64x64 tile (analytic)."""
    macs, s = 0, TILE
    for ci, co in ENC:
        s //= 2
        macs += s * s * 9 * ci * co
    macs += 2 * (BOTTOM * BOTTOM * ENC[-1][1] * LATENT)
    s = BOTTOM
    for ci, co in DEC:
        s *= 2
        macs += s * s * 9 * ci * co
    macs += TILE * TILE * 9 * DEC[-1][1]
    return macs


def save_params(p, path):
    """Save the parameter pytree to a flat .npz file (no pickle)."""
    flat = {}
    for part in ("enc", "dec"):
        for i, layer in enumerate(p[part]):
            for k, v in layer.items():
                flat[f"{part}.{i}.{k}"] = np.asarray(v, np.float32)
    for part in ("fc_e", "fc_d", "out"):
        for k, v in p[part].items():
            flat[f"{part}.{k}"] = np.asarray(v, np.float32)
    np.savez(path, **flat)


def load_params(path, as_jax=True):
    """Load parameters saved by `save_params`."""
    z = np.load(path)
    conv = jnp.asarray if as_jax else np.asarray
    p = {"enc": [{} for _ in ENC], "dec": [{} for _ in DEC]}
    for key in z.files:
        parts = key.split(".")
        if parts[0] in ("enc", "dec"):
            p[parts[0]][int(parts[1])][parts[2]] = conv(z[key])
        else:
            p.setdefault(parts[0], {})[parts[1]] = conv(z[key])
    return p


# ----------------------------------------------------------------------------- network
def _conv(x, layer, stride=1):
    y = lax.conv_general_dilated(x, layer["w"], (stride, stride), "SAME",
                                 dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + layer["b"]


def _act(x):
    return jax.nn.leaky_relu(x, NEG_SLOPE)


def encode(p, x):
    """f_theta: (N, 64, 64, 1) -> (N, LATENT)."""
    for layer in p["enc"]:
        x = _act(_conv(x, layer, 2))
    x = x.reshape(x.shape[0], -1)
    return x @ p["fc_e"]["w"] + p["fc_e"]["b"]


def decode(p, z):
    """g_phi: (N, LATENT) -> (N, 64, 64, 1)."""
    x = _act(z @ p["fc_d"]["w"] + p["fc_d"]["b"]).reshape(-1, BOTTOM, BOTTOM, ENC[-1][1])
    for layer in p["dec"]:
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = _act(_conv(x, layer))
    return jax.nn.sigmoid(_conv(x, p["out"]))


def forward(p, x):
    """x_hat = g_phi(f_theta(x))."""
    return decode(p, encode(p, x))


# ----------------------------------------------------------------------------- loss
def _gauss_kernel(size=7, sigma=1.5):
    g = np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2))
    k = np.outer(g, g)
    k /= k.sum()
    return jnp.asarray(k[:, :, None, None], jnp.float32)


_GK = _gauss_kernel()


def ssim_map(a, b):
    """Local SSIM (Gaussian 7x7 window, sigma 1.5) for NHWC single-channel images in [0, 1]."""
    f = lambda t: lax.conv_general_dilated(t, _GK, (1, 1), "SAME",
                                           dimension_numbers=("NHWC", "HWIO", "NHWC"))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def loss_fn(p, x, alpha=ALPHA):
    """alpha * (1 - SSIM) + (1 - alpha) * L1, averaged over the batch (eq. 1)."""
    xh = forward(p, x)
    return alpha * (1 - ssim_map(x, xh).mean()) + (1 - alpha) * jnp.abs(x - xh).mean()
