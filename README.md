# Prediction-Powered Inference via Calibration

This repository contains the code and manuscript assets for the paper on calibration-based semisupervised mean inference. The repo is organized around two experiment pipelines: the synthetic simulation study and the real-data reproduction of the original PPI mean benchmarks, together with a user-facing Python package for semisupervised mean estimation with AIPW, calibration, and practical uncertainty quantification.

## Repo layout

- `experiments/`: core experiment drivers and shared estimator code
- `src/ppi_aipw/`: user-facing Python package for AIPW with calibration
- `docs/ppi_aipw_vignette.md`: practitioner-friendly package vignette
- `docs/index.html`: interactive package website and historical review of PPI
- `paper/`: manuscript source files and tracked manuscript-ready assets
- `scripts/`: public entrypoints for reproduction
- `tests/`: smoke and unit tests
- `outputs/`: local-only regenerated runs, caches, and intermediate artifacts

## `ppi_aipw` package

This repo now includes `ppi_aipw`, a practitioner-friendly Python package for
semisupervised mean estimation with AIPW, calibration, and practical
uncertainty quantification. The package is motivated by the perspective that
recent prediction-powered inference methods fit naturally within the broader
semiparametric, missing-data, and debiased machine learning toolkit.

Most users should start with:

```python
from ppi_aipw import mean_pointestimate, mean_ci

estimate = mean_pointestimate(Y, Yhat, Yhat_unlabeled, method="linear")
lower, upper = mean_ci(Y, Yhat, Yhat_unlabeled, method="linear", alpha=0.1)
```

Plain aliases are available:

- `mean_pointestimate`
- `mean_ci`
- `mean_se`

The original explicit names like `aipw_mean_pointestimate` and `aipw_mean_ci`
are still available too.

Available methods:

- `method="aipw"`: no calibration, just AIPW on the raw predictions
- `method="linear"`: linear recalibration before AIPW
- `method="platt"`: sigmoid-style recalibration with outcome-range rescaling
- `method="isocal"`: isotonic recalibration before AIPW

Available interval types:

- `inference="wald"`: fast analytic interval
- `inference="bootstrap"`: resampling-based interval that refits calibration in each resample

Recommended defaults:

- start with `method="linear"`
- start with `inference="wald"`
- use `method="platt"` when predictions are probability-like or naturally bounded
- use `method="isocal"` when you expect monotone nonlinear miscalibration and have enough labeled data
- use `inference="bootstrap"` as a robustness check when you can afford extra compute

The full user guide is in [docs/ppi_aipw_vignette.md](docs/ppi_aipw_vignette.md).
There is also a tiny runnable notebook example in [examples/ppi_aipw_quickstart.ipynb](examples/ppi_aipw_quickstart.ipynb).
The package website lives at [docs/index.html](docs/index.html).

## Environment setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Smoke test

```bash
.venv/bin/pytest -q
./scripts/reproduce_paper.sh --smoke
```

The smoke run validates the full pipeline with tiny replication counts. It is intended to check that the code paths, downloads, plotting, and asset export all work.

## Full paper reproduction

```bash
./scripts/reproduce_paper.sh
```

This runs the finalized paper settings end-to-end:

- the synthetic simulation pipeline
- synthetic plotting/export
- the real-data PPI mean reproduction pipeline
- export of manuscript-facing assets into `paper/assets/`

## What gets generated

- Heavy local outputs are written under `outputs/`
- Tracked manuscript-ready assets are written to `paper/assets/`

The repository intentionally tracks only lightweight paper assets, not the full raw outputs. Heavy outputs are excluded from git and are meant to be regenerated locally.

## Rough runtime

The full reproduction is a multi-hour run. The finalized paper settings use large replication counts for both the synthetic study and the real-data benchmark rerandomization, so expect the full command to take several hours on a typical laptop or workstation.

## Dataset downloads

The real-data reproduction uses the official `ppi_py` dataset bundle. If the cached files are not already present, they are downloaded automatically into the local cache directory under `outputs/cache/ppi_datasets/`.
# calibrated-ppi
