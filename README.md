# CAE_DD_AFP: Convolutional Autoencoder for Defect Detection in Automated Fiber Placement

Unsupervised in-process inspection of automated fiber placement (AFP) of polymer composites.
A lightweight convolutional autoencoder (CAE) is trained on defect-free laser-profilometer
height maps only; regions it fails to reconstruct are reported as defects (gaps, overlaps,
twisted and missing tows, wrinkles, fuzzballs, foreign objects).

Source code for the conference abstract
*"Machine Learning Models for Defect Detection: Convolutional Autoencoders for In-Process Inspection
of Automated Fiber Placement"* — V.I. Marinin, R.G. Zaytsev, A.G. Savin, Platov South-Russian State
Polytechnic University (NPI), Novocherkassk, Russia.


![Pipeline](results/fig1.png)

## Method

**Input.** 
Height map from a laser profilometer on the layup head, 0.1 mm/px, columns across the
course and rows along the layup direction; a frame is 768 × 256 px (76.8 × 25.6 mm, 12 tows of 6.35 mm).
It is median-filtered (3 × 3) and cut into 64 × 64 tiles with stride 32.

**Model.** 
The encoder $f_\theta$ (four stride-2 3 × 3 convolutions with LeakyReLU and a dense layer)
maps a tile $x \in [0,1]^{64\times 64}$ to a latent code $z = f_\theta(x) \in \mathbb{R}^{32}$; the decoder
$g_\varphi$ (dense layer, four nearest-neighbour ×2 upsampling + 3 × 3 convolution stages, sigmoid) restores
$\hat{x} = g_\varphi(z)$. The parameters minimise the SSIM/L1 loss over $N$ defect-free tiles:

$$
(\theta^*, \varphi^*) = \arg\min_{\theta,\varphi} \frac{1}{N}\sum_{i=1}^{N}
\Big[\alpha\big(1-\mathrm{SSIM}(x_i,\hat{x}_i)\big) + (1-\alpha)\,\lVert x_i-\hat{x}_i\rVert_1\Big],
\qquad \alpha = 0.85. \tag{1}
$$

**Decision rule.** Residuals of overlapping tiles are averaged and smoothed; the outer 1.2 mm of the
course is masked:

$$
A = G_\sigma * |x-\hat{x}|,\qquad D(u,v) = \mathbf{1}\big[A(u,v) > \tau\big],\qquad
\tau = \max_{x\in V_0,\,(u,v)} A(u,v), \tag{2}
$$

where $V_0$ is a set of defect-free validation scans and $\sigma = 3$ px. Connected components of $D$
with area ≥ 0.2 mm² are reported with their bounding boxes in millimetres.

| Layer | Output | Parameters |
|---|---|---|
| input tile | 64 × 64 × 1 | — |
| 4 × Conv 3×3/2 + LeakyReLU (16, 32, 64, 64 ch.) | 4 × 4 × 64 | 60 224 |
| Dense → latent $z$ | 32 | 32 800 |
| Dense + LeakyReLU, reshape | 4 × 4 × 64 | 33 792 |
| 4 × (Up×2 + Conv 3×3 + LeakyReLU) (64, 32, 16, 16 ch.) | 64 × 64 × 16 | 62 336 |
| Conv 3×3 + sigmoid | 64 × 64 × 1 | 145 |
| **Total** | | **189 297 (0.72 MB FP32), 25 MMAC per tile** |

## Results (synthetic data)

Synthetic data emulate laser-profilometer scans of a 12-tow dry carbon-tape course
(`afp_cae/synth.py`: tow steering, crowned cross-sections, fibre texture, sensor noise and dropouts,
seven defect types with pixel-exact masks). See `results/samples.png`.

* Training: 16 000 random tiles from 400 defect-free frames, Adam, cosine LR from 1e-3, batch 64, 25 epochs.
* Threshold: $\tau$ = 0.1081, maximum of $A$ over 100 defect-free validation frames.
* Test: 570 frames (150 defect-free + 60 per defect type) = 14.6 m of course.

| Metric | Value |
|---|---|
| Tile-level AUROC / AP | 0.987 / 0.884 |
| Tile-level precision / recall / F1 at $\tau$ | 0.828 / 0.871 / 0.849 |
| Pixel-level AUROC | 0.977 |
| Defect instances detected | 87.4 % |
| False alarms (components > 3 mm from any defect) | 0.34 per metre (5 in 14.6 m) |
| False alarms on the 150 defect-free test frames | 0 |

| Defect type | gap | overlap | twist | foreign object | fuzzball | wrinkle | missing tow |
|---|---|---|---|---|---|---|---|
| Detected, % | 100 | 100 | 96.7 | 75.0 | 60.0 | 80.0 | 100 |

**Inference** (FP32, CPU only, median of repeated runs; a frame is 161 tiles):

| | Abstract run: Intel Xeon 2.8 GHz, 2 vCPU | Re-run of this repo (`results/*.json`): Intel Xeon 2.1 GHz, 2 vCPU |
|---|---|---|
| ONNX Runtime, 1 tile | 0.83 ms | 0.66 ms |
| ONNX Runtime, 1 frame | 121 ms (1330 tiles/s) | 87 ms (1850 tiles/s) |
| Full pipeline with ONNX Runtime (median filter, tiling, CAE, fusion, threshold, components) | 154 ms → 6.5 frames/s → 10 m/min | 121 ms → 8.2 frames/s → 12.6 m/min |
| JAX/XLA, 1 tile / 1 frame | 1.86 ms / 163 ms | 1.10 ms / 137 ms |

The maximum layup speed is frames/s × 25.6 mm of course per frame. Timings depend on the hardware;
detection metrics are deterministic and are reproduced exactly by `scripts/evaluate.py`.

## Extended experiments (journal article)

The scripts in `experiments/` reproduce the extended results of the journal article (same synthetic data and
evaluation protocol; numbers below are read from `results/article/*.json`).

```bash
python experiments/compare_methods.py        # baselines, map variants, hybrid, thresholds (~6 min)
python experiments/latent_ablation.py        # latent size d = 8, 16, 32, 64 (~35 min)
python experiments/benchmark_inference.py    # batch / threads / stride / stage timing (~3 min, idle machine)
python experiments/make_article_figures.py   # black-and-white figures with Russian labels
# every script accepts --quick for a smoke test
```

**Methods and variants** (test: 570 frames = 14.6 m of course; every method uses its own τ
calibrated on the same 100 defect-free frames; PCA explains 89.1 % of the tile variance):

| Method | Tile AUROC | Pixel AUROC | Detected, % | Mean IoU | False alarms / m |
|---|---|---|---|---|---|
| CAE, σ = 3 px (main) | 0.987 | 0.977 | 87.4 | 0.52 | 0.34 |
| CAE, σ = 1 px | 0.980 | 0.961 | 52.9 | 0.50 | 0.07 |
| CAE, σ = 2 px | 0.987 | 0.971 | 78.8 | 0.51 | 0.41 |
| CAE, σ = 5 px | 0.984 | 0.984 | 85.5 | 0.50 | 0.55 |
| CAE, map (1 − SSIM)/2 | 0.953 | 0.902 | 36.4 | 0.55 | 0.00 |
| CAE, non-overlapping tiles | 0.975 | 0.974 | 63.6 | 0.54 | 0.27 |
| PCA, 32 components | 0.973 | 0.911 | 77.1 | 0.23 | 0.55 |
| Background subtraction along the layup direction (no learning) | 0.940 | 0.774 | 97.9 | 0.37 | 0.00 |
| Hybrid: max(A_CAE/τ_CAE, A_bg/τ_bg) | 0.994 | 0.992 | 97.9 | 0.57 | 0.34 |

A training-free background subtraction along the layup direction finds more defect instances on these synthetic
scans (the simulated tows are homogeneous along the track) but ranks and localises them much worse; the hybrid
keeps the recall of the filter and the ranking quality of the CAE at the same false-alarm rate.

**Threshold operating points** of the main CAE:

| Threshold (main CAE) | τ | Detected, % | False alarms / m | Defect-free frames with an alarm |
|---|---|---|---|---|
| 1.2 τ | 0.1297 | 73.1 | 0.21 | 0 of 150 |
| τ = max over V0 (default) | 0.1081 | 87.4 | 0.34 | 0 of 150 |
| quantile 0.99 | 0.1047 | 89.0 | 0.55 | 0 of 150 |
| quantile 0.95 | 0.1019 | 89.8 | 0.69 | 0 of 150 |
| quantile 0.90 | 0.0989 | 90.2 | 0.96 | 1 of 150 |
| 0.8 τ | 0.0865 | 93.6 | 7.33 | 21 of 150 |

**Latent dimension** (reduced budget: 8000 tiles, 10 epochs; test 50 defect-free + 20 frames per type). A larger
code lowers the training loss but starts to reconstruct defects, so d = 32 gives the best detection:

| d | Parameters | Final loss | Tile AUROC | Detected, % | False alarms / m |
|---|---|---|---|---|---|
| 8 | 140,121 | 0.165 | 0.902 | 28.6 | 0.41 |
| 16 | 156,513 | 0.142 | 0.966 | 42.9 | 0.00 |
| 32 | 189,297 | 0.119 | 0.967 | 57.1 | 0.00 |
| 64 | 254,865 | 0.086 | 0.965 | 37.1 | 0.00 |

**Inference** (Intel(R) Xeon(R) Processor @ 2.10GHz, 2 vCPU, FP32, medians):

| Batch (tiles) | ONNX Runtime, 1 thread, ms | ONNX Runtime, 2 threads, ms | JAX, ms | Tiles/s (ORT, 2 threads) |
|---|---|---|---|---|
| 1 | 0.87 | 0.66 | 1.27 | 1504 |
| 8 | 7.05 | 4.10 | 5.99 | 1952 |
| 32 | 29.00 | 16.15 | 23.18 | 1981 |
| 64 | 58.96 | 31.70 | 48.32 | 2019 |
| 161 | 158.49 | 95.26 | 123.51 | 1690 |

| Stage, ms | Stride 32 (161 tiles) | Stride 64 (48 tiles) |
|---|---|---|
| Median filter 3×3 | 25.8 | 26.3 |
| Tiling | 0.3 | 0.1 |
| CAE (ONNX Runtime, 2 threads) | 92.4 | 23.9 |
| Fusion of overlaps | 3.1 | 1.0 |
| Gaussian smoothing | 4.1 | 4.1 |
| Threshold + connected components | 0.6 | 0.6 |
| **Full frame pipeline** | 135.4 | 61.0 |
| Frames/s → max layup speed | 7.4 → 11.3 m/min | 16.4 → 25.2 m/min |

## DRAEM-CAE (branch `draem`, pilot study)

DRAEM (Zavrtanik et al., ICCV 2021) built on the CAE above (`afp_cae/draem.py`). The CAE keeps its architecture
and loss (1) but learns to *restore* a defect-free tile $x$ from a corrupted copy $x_a$; a small U-Net
$h_\psi$ segments the corruption from the pair $(x_a, \hat{x})$:

$$
\mathcal{L} = \alpha\big(1-\mathrm{SSIM}(x,\hat{x})\big) + (1-\alpha)\,\lVert x-\hat{x}\rVert_1
+ \mathcal{L}_{\mathrm{focal}}\big(\sigma(h_\psi(x_a, \hat{x})), M\big),\qquad \hat{x} = g_\varphi(f_\theta(x_a)).
$$

The corruption $(x_a, M)$ is drawn on the device during training: a thresholded Perlin-noise region inside which the
height becomes a blend of the tile and a height step, a smooth relief or another tile. The defect generator of
`afp_cae/synth.py` is **never** used for training. At inference the anomaly map is $\sigma(h_\psi)$ instead of
$|x-\hat{x}|$; fusion of tiles, smoothing, edge band and the decision rule (2) are unchanged
(`anomaly_map(..., kind="map")`).

| | CAE | DRAEM-CAE |
|---|---|---|
| Parameters | 189 297 | 247 362 (CAE + 58 065 U-Net) |
| MMAC per tile | 25 | 92 |
| Training, 16 000 tiles (RTX 4080 SUPER, JAX in WSL2) | 25 epochs, ~20 s | 100 epochs, ~2.5 min |

**Synthetic data** (protocol of `scripts/evaluate.py`; mean ± s.d. over 5 training seeds, `scripts/seeds.py`):

| Metric | CAE | DRAEM-CAE |
|---|---|---|
| Tile-level AUROC / AP | 0.989 / 0.894 | 1.000 / 0.995 |
| Defect instances detected | 87.2 ± 2.9 % (83–90) | 99.9 ± 0.1 % |
| Mean IoU | 0.53 ± 0.01 | 0.81 ± 0.03 |
| False alarms per metre | 0.30 ± 0.24 | 0.55 ± 0.57 (0–1.37) |
| Defect-free test frames with an alarm, per seed | 0, 0, 0, 1, 4 | 0, 8, 1, 0, 0 |

| Defect type, detected % | gap | overlap | twist | foreign object | fuzzball | wrinkle | missing tow |
|---|---|---|---|---|---|---|---|
| CAE | 100 | 100 | 96 | 78 | 58 | 79 | 100 |
| DRAEM-CAE | 100 | 100 | 100 | 99 | 100 | 100 | 100 |

The synthetic benchmark is saturated by DRAEM-CAE; differences between its variants are evaluated on real scans.

**Real scans** of Cardoso et al. (2026), Zenodo [10.5281/zenodo.17048470](https://doi.org/10.5281/zenodo.17048470),
CC BY 4.0: 2 layers × 3 passes of a KUKA AFP cell, resampled to 0.1 mm/px (`afp_cae/cardoso.py`), synthetic defects
of 5 types injected in millimetres into the clean scans (`cardoso.inject`). Protocol of `scripts/eval_cardoso.py`:
cross-validation over passes (train on 4, test on 2), threshold = 99.5th percentile of scores of models that did not
see the scored pass, 10 injections × 2 test passes × 3 folds; 2 775 defects per method for 5 seeds
(`results/cardoso_5seeds/`); 95 % bootstrap intervals. False zones: components of $\{A>\tau\}$ that do not touch a
defect (no minimal area), per metre of scan. These numbers are not comparable with the synthetic false alarms above
(different threshold rule and data).

| Metric (5 seeds) | CAE | DRAEM-CAE |
|---|---|---|
| Tile-level AUROC | 0.78 | 0.84 |
| Defects detected | 57.7 % (55.9–59.5) | 64.9 % (63.1–66.7) |
| Defects detected, per seed | 56.8, 60.2, 56.9, 57.5, 57.3 | 64.9, 68.6, 62.2, 66.3, 62.5 |
| False zones per metre | 32.4 | 28.4 |
| False zones per metre, per seed | 33.1, 36.3, 32.5, 32.6, 27.3 | 29.0, 42.8, 20.8, 28.3, 21.0 |
| False zones on tow seams per metre (seams cover 17.5 % of the area) | 24.4 (76 %) | 9.7 (34 %) |

| Defect type, detected % | gap | overlap | foreign object | twist | wrinkle |
|---|---|---|---|---|---|
| CAE | 67 | 81 | 55 | 43 | 36 |
| DRAEM-CAE | 76 | 81 | 71 | 49 | 40 |

DRAEM-CAE detects 7 percentage points more defects (foreign objects and gaps most of all) with fewer false zones on
average. Three quarters of the false zones of the CAE lie on tow seams, where their density is ~15 times that on the
rest of the tape; DRAEM-CAE has 2.5 times fewer of them there, its remaining false zones lie mostly on scanner artefacts
of layer 2. Its false-zone count depends on the training run (one seed of five gives 42.8 per metre): GPU training is
not deterministic, and the same seed retrained gives a different count.

**Ensemble** (`eval_cardoso.py --ensemble 3`, `results/cardoso_ens3/`): the anomaly map is the mean of 3 DRAEM-CAE
models trained with different seeds; every model of the protocol (auxiliary and final) is such an ensemble.
Two ensembles (seeds 0, 1000, 2000 and 1, 1001, 2001):

| Metric | DRAEM-CAE, single model (5 seeds) | DRAEM-CAE, ensemble of 3 (2 ensembles) |
|---|---|---|
| Tile-level AUROC | 0.84 | 0.85 |
| Defects detected | 64.9 % (63.1–66.7) | 67.3 % (64.6–70.0) |
| False zones per metre, per run | 29.0, 42.8, 20.8, 28.3, 21.0 | 27.9, 32.4 |
| False zones on tow seams per metre | 9.7 | 9.2 |

The ensemble removes the outlier runs at the same detection rate, at 3 times the inference cost
(~2.7 m/min on the Ryzen of the inference table below).

**Exclusion of tape boundaries** (`scripts/eval_boundaries.py`, `results/cardoso_boundaries/`, masks in
`masks.png`): tow seams and tape edges are found on the clean scan (0.1 mm/px grid, edge slope 0.3 mm/mm, ±0.5 mm
band) and the anomaly map is set to 0 there, as the 1.2 mm course-edge band. Detectors: Sobel along x on the map
smoothed along the layup; Canny edges running along the layup; straight probabilistic-Hough segments (±15° of the
layup, ≥ 10 mm) on the Canny edges. The threshold is that of the variant without exclusion, so exclusion only removes
what lies in the zone (a threshold recalibrated on the kept pixels just moves the false zones onto the tape, see
`summary.json`). Defect set with half of the gaps and overlaps centred on tow seams (465 of 1 665 defects on seams),
3 seeds:

| Method | Boundary detector | Area excluded | Detected | on seams | off seams | False zones / m | on seams |
|---|---|---|---|---|---|---|---|
| CAE | none | 0 % | 57.8 % | 79.6 % | 49.2 % | 36.4 | 27.4 |
| CAE | Sobel x | 17.1 % | 49.3 % | 52.5 % | 48.1 % | 35.4 | 3.1 |
| CAE | Canny | 13.2 % | 53.1 % | 64.3 % | 48.7 % | 35.9 | 21.0 |
| CAE | Canny + Hough | 12.1 % | 54.3 % | 68.4 % | 48.8 % | 34.1 | 21.4 |
| DRAEM-CAE | none | 0 % | 67.4 % | 81.7 % | 61.8 % | 27.2 | 9.0 |
| DRAEM-CAE | Sobel x | 17.1 % | 60.7 % | 59.6 % | 61.1 % | 22.8 | 0.5 |
| DRAEM-CAE | Canny | 13.2 % | 64.4 % | 72.5 % | 61.3 % | 24.3 | 5.1 |
| DRAEM-CAE | Canny + Hough | 12.1 % | 64.9 % | 74.0 % | 61.4 % | 25.1 | 6.4 |

Excluding the boundaries removes the false zones on seams but also the real gaps and overlaps that occur there
(on-seam detection drops by 8–27 percentage points), while the total number of false zones falls little: for the
CAE it hardly changes, probably because zones crossing a seam split into pieces on both sides of the excluded band.
Narrower zones (Canny, Canny + Hough) lose fewer seam defects and remove fewer false zones. The tile-level AUROC rises
after exclusion (CAE 0.80 → 0.83) only because the seams, where the clean-tape scores are highest, are no longer
scored. With randomly placed defects the picture is the same (`summary.json`, set `random`). Suppressing seam false
zones by the model (DRAEM-CAE, its ensemble) keeps the seam defects and is preferable to masking.

**Inference** (ONNX Runtime, FP32, 2 threads, AMD Ryzen 9 7900X; both models measured on the same machine):

| | CAE | DRAEM-CAE |
|---|---|---|
| Tiles/s | 3 068 | 908 |
| Full frame pipeline, frames/s | 14.6 | 5.2 |
| Max layup speed | 22.4 m/min | 8.0 m/min |

On the 2-vCPU Xeon of the table above this scales to roughly 4.5 m/min for DRAEM-CAE.

```bash
python scripts/train_draem.py                      # DRAEM-CAE on the synthetic training set -> models/draem_cae.npz
python scripts/evaluate.py --method draem          # synthetic protocol -> results/results_draem.json
python scripts/export_onnx.py --method draem       # models/draem_cae.onnx, results/ort_bench_draem.json
python scripts/detect.py --demo --method draem     # also calibrate.py --method draem
python scripts/seeds.py --method draem             # 5 seeds -> results/seeds_draem.json
python scripts/eval_cardoso.py --data path/to/cardoso/data            # CAE and DRAEM-CAE on real scans
python scripts/eval_cardoso.py --data path/to/cardoso/data --methods draem --ensemble 3
python scripts/eval_boundaries.py --data path/to/cardoso/data         # exclusion of tape boundaries
```

JAX has no CUDA build for native Windows; on Windows train on the GPU inside WSL2 with `pip install "jax[cuda12]"`
and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Quick start

```bash
git clone https://github.com/rzajtsev/cae_afp.git && cd cae_afp
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/detect.py --demo --png demo.png          # synthetic frame -> detections
python scripts/evaluate.py                              # reproduce the metrics (~6 min on 2 cores)
python scripts/export_onnx.py                           # ONNX export, parity check, ORT benchmark
python scripts/train.py                                 # retrain from scratch (~45 min on 2 cores)
python scripts/make_figure.py                           # results/fig1.png
python scripts/show_samples.py                          # results/samples.png
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, JAX 0.11, Optax 0.2.8, ONNX 1.23,
ONNX Runtime 1.29, Matplotlib 3.11. For GPU training install the CUDA build of JAX.

## Using your own scans

Save each height map as a 2-D `.npy` array (0.1 mm/px, rows along the layup direction). Raw heights can be
scaled to [0, 1] with `--normalize` (robust per-scan min-max).

```bash
python scripts/train.py --data scans/train_defect_free --normalize --crops 200 --out models/real.npz
python scripts/calibrate.py --data scans/val_defect_free --normalize --weights models/real.npz
python scripts/detect.py --input scans/new_course.npy --normalize --weights models/real.npz \
                         --json defects.json --png defects.png
```

If the pixel pitch differs from 0.1 mm, resample the scans or adjust `PipelineConfig` in
`afp_cae/pipeline.py` (tile, stride, smoothing, edge band, minimal area).

## Repository layout

```
afp_cae/
  synth.py         synthetic 12-tow AFP height maps with seven defect types
  model.py         CAE in JAX: initialisation, encoder/decoder, SSIM, loss (1), save/load (.npz)
  pipeline.py      pre-processing, tiling, anomaly map, decision rule (2), JAX and ONNX backends
  onnx_export.py   JAX parameters -> ONNX graph (opset 17); DRAEM-CAE graph (build_draem_onnx)
  draem.py         DRAEM-CAE: Perlin corruption, U-Net discriminator, loss, save/load
  cardoso.py       Cardoso et al. scans: loading, 0.1 mm/px grid, tow seams, tape-boundary detectors, defects in mm
scripts/
  train.py         training (synthetic or --data)
  train_draem.py   DRAEM-CAE training (synthetic or --data; --init from a trained CAE)
  seeds.py         train + evaluate with several seeds -> results/seeds_<method>.json
  eval_cardoso.py  CAE / DRAEM-CAE on the Cardoso scans (--seeds, --ensemble) -> results/cardoso*/
  eval_boundaries.py  exclusion of tape boundaries on the Cardoso scans -> results/cardoso_boundaries/
  calibrate.py     threshold calibration on defect-free scans
  evaluate.py      full evaluation protocol + JAX benchmark -> results/results.json, models/threshold.json
  export_onnx.py   ONNX export + ONNX Runtime benchmark -> models/afp_cae.onnx, results/ort_bench.json
  detect.py        detection on one height map (.npy) or a demo frame
  make_figure.py   Figure 1 of the abstract
  show_samples.py  examples of the synthetic defects
experiments/
  common.py                 shared evaluation protocol of the extended experiments
  compare_methods.py        baselines (PCA, background subtraction), map variants, hybrid, threshold sweep
  latent_ablation.py        latent dimension ablation
  benchmark_inference.py    batch / threads / backend / stride / stage timing
  make_article_figures.py   black-and-white figures of the journal article
models/            trained weights (afp_cae.npz, draem_cae.npz), ONNX models, threshold.json, threshold_draem.json
models/seeds/      weights of the seed runs
results/           metrics, benchmarks, training log, figures
results/article/   results and figures of the extended experiments
```

## Limitations

* All metrics are obtained on synthetic data with one defect per frame; they are not a substitute for
  validation on real AFP scans.
* The threshold is calibrated for zero false alarms on the validation set; small low-contrast defects
  (fuzzballs, foreign objects) are the main source of misses. A lower quantile in `calibrate.py`
  trades false alarms for recall.
* Defects are localised but not classified by type.
* DRAEM-CAE: the results on real scans are a pilot (6 passes of one part, synthetic defects drawn on real
  backgrounds, 5 seeds; the ensemble and the boundary study 2 and 3 seeds). Its false-alarm rate varies between
  training runs (an ensemble of 3 models stabilises it); a single model is 3.4 times slower than the CAE, the
  ensemble 10 times. The parameters of the boundary detectors were fixed once by eye, not tuned.
* On the synthetic data a training-free directional background subtraction is a strong baseline at the
  instance level; the hybrid detector (`experiments/compare_methods.py`) is recommended until the CAE is
  validated on real scans.

## Citation

See `CITATION.cff`. Related work by the authors: R.G. Zaytsev, A.R. Zaytseva, "Development directions and
evolutionary characteristics of generative-adversarial networks", *Drukerovskij Vestnik*, 2023, No. 1,
pp. 227–232, doi:10.17213/2312-6469-2023-1-227-232.

## License

Not yet chosen. Add a `LICENSE` file before making the repository public (for example MIT or Apache-2.0,
subject to the university's IP rules); without it the code is "all rights reserved" by default.
