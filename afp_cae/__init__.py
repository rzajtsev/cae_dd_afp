"""afp_cae: convolutional-autoencoder defect detection for automated fiber placement (AFP)."""
from .model import ALPHA, LATENT, TILE, forward, init_params, load_params, loss_fn, save_params
from .pipeline import (DEFAULT, PipelineConfig, anomaly_map, detect, jax_predictor, onnx_predictor,
                       preprocess, tiles)

__version__ = "0.1.0"
