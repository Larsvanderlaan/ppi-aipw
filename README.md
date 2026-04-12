# Prediction-Powered Inference via Calibration

This repository contains the code and manuscript assets for the paper on calibration-based semisupervised mean inference. The repo is organized around two experiment pipelines: the synthetic simulation study and the real-data reproduction of the original PPI mean benchmarks, together with a user-facing Python package for semisupervised mean estimation with AIPW, calibration, and practical uncertainty quantification.

## Repo layout

- `experiments/`: core experiment drivers and shared estimator code
- `src/ppi_aipw/`: user-facing Python package for AIPW with calibration
- `docs/ppi_aipw_vignette.md`: package vignette
- `docs/index.html`: interactive package website with docs, examples, and paper links
- `paper/`: manuscript source files and tracked manuscript-ready assets
- `scripts/`: public entrypoints for reproduction
- `tests/`: smoke and unit tests
- `outputs/`: local-only regenerated runs, caches, and intermediate artifacts

## `ppi_aipw` package

This repo now includes `ppi_aipw`, a Python package for semisupervised mean
estimation with AIPW, calibration, and uncertainty quantification.

Most users should start with:

```python
from ppi_aipw import mean_inference

result = mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    alpha=0.1,
)

estimate = result.pointestimate
standard_error = result.se
lower, upper = result.ci
```

If you want a linear prognostic adjustment with optional extra covariates:

```python
result = mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    X=X,
    X_unlabeled=X_unlabeled,
    method="prognostic_linear",
)
```

If you only want one output at a time, the narrower convenience functions are
still available:

```python
from ppi_aipw import mean_pointestimate, mean_ci

estimate = mean_pointestimate(Y, Yhat, Yhat_unlabeled)
lower, upper = mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1)
```

If you want raw-score AIPW with empirical efficiency maximization, turn on the
new flag:

```python
result = mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="aipw",
    efficiency_maximization=True,
)

estimate = result.pointestimate
lambda_hat = result.efficiency_lambda
```

If you want the package to choose among a small set of methods by
cross-validated empirical efficiency maximization:

```python
result = mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="auto",
    candidate_methods=("aipw", "linear", "isotonic"),
    num_folds=100,
)

estimate = result.pointestimate
selected_method = result.method
selected_candidate = result.selected_candidate
```

Plain aliases are available:

- `mean_inference`
- `mean_pointestimate`
- `mean_ci`
- `mean_se`

The original explicit names like `aipw_mean_inference`,
`aipw_mean_pointestimate`, and `aipw_mean_ci` are still available too.

Available methods:

- `method="aipw"`: no calibration, just AIPW on the raw predictions
- `method="linear"`: linear recalibration before AIPW
- `method="prognostic_linear"`: semisupervised linear adjustment using the prediction score plus optional extra covariates `X`; the intercept and score coefficient are unpenalized, while the extra covariates are ridge-tuned on the labeled sample
- `method="sigmoid"`: sigmoid-style recalibration with outcome-range rescaling
- `method="monotone_spline"`: smooth monotone spline recalibration before AIPW
- `method="isotonic"`: monotone isotonic recalibration before AIPW; by default this uses a one-round monotone XGBoost calibrator with `isocal_min_child_weight=10`, and you can switch to `isocal_backend="sklearn"` for scikit-learn isotonic regression
- `method="auto"`: choose among candidate methods by cross-validated influence-function variance minimization; if `aipw` is among the candidates, `auto` also compares against an efficiency-maximized AIPW candidate
- `num_folds=100`: default number of folds used by `method="auto"`; it is capped at the labeled sample size
- for `method="auto"`, the foldwise objective uses an unlabeled subset of size
  `min(n_unlabeled, 10 * n_labeled)` by default
- for `method="auto"`, the selected calibration map is refit on the full labeled
  sample, the final point estimate uses the full unlabeled sample, and any
  final lambda scaling is learned from pooled out-of-fold labeled predictions
  plus unlabeled-subset predictions for the selected method and then reused for
  the point estimate and Wald SE
- for `method="auto"` with bootstrap inference, the package selects the method
  once on the original sample and then bootstraps only that chosen method

Optional score rescaling:

- `efficiency_maximization=True`: replace the predictor used by the estimator by
  `lambda m(X)`, where `m(X)` is the raw score for `method="aipw"` and the
  calibrated score for the other methods
- the package estimates the coordinatewise `lambda` by minimizing the empirical
  influence-function variance and then uses the empirically chosen
  efficiency-maximizing rescaled predictor in the final estimator
- with `method="aipw"`, this is the unrestricted efficiency-maximized raw-score
  correction without clipping lambda
- with `method="auto"`, this flag does not affect the method-choice stage; it
  only says whether the finally selected method also gets a cross-fitted lambda
  after selection

Available interval types:

- `inference="wald"`: fast analytic interval
- `inference="bootstrap"`: resampling-based interval that refits calibration in each resample

Recommended defaults:

- start with `method="monotone_spline"`
- start with `inference="wald"`
- use `method="linear"` when you want the simplest affine recalibration and maximum interpretability
- use `method="prognostic_linear"` when you want a linear prognostic adjustment with optional extra covariates `X`
- use `method="sigmoid"` when predictions are probability-like or naturally bounded
- use `method="monotone_spline"` when you want the default smooth monotone nonlinear recalibration that is less stepwise than isotonic
- use `method="isotonic"` when you expect monotone nonlinear miscalibration and have enough labeled data; the default backend is monotone XGBoost, with `isocal_backend="sklearn"` available as a simpler fallback
- use `efficiency_maximization=True` when you want the package to empirically rescale the score for lower variance, especially for raw-score `method="aipw"`
- use `method="auto"` when you want a data-adaptive choice among `aipw`, `linear`, and `isotonic`, while also letting the selector compare against an efficiency-maximized AIPW candidate
- use `num_folds=100` as the default auto-selection setting unless you have a reason to make it smaller
- use `inference="bootstrap"` as a robustness check when you can afford extra compute

Result object:

- `result.pointestimate`: point estimate
- `result.se`: standard error
- `result.ci`: confidence interval for the requested `alpha` and `alternative`
- `result.method`: selected base method
- `result.selected_candidate`: selected auto candidate label
- `result.efficiency_lambda`: estimated lambda when efficiency maximization is active
- `result.diagnostics`: method-selection and inference metadata

Causal API:

- `causal_inference(Y, A, Yhat_potential, ...)` computes arm-specific potential
  outcome means plus all control-vs-treatment ATEs in one Wald-only call
- `Yhat_potential` should have one column per treatment arm and one row per unit
- optional `X` is passed through arm-by-arm, so `method="prognostic_linear"`
  gives the same score-plus-covariate linear adjustment inside each arm
- `control_arm=None` defaults to the minimum resolved treatment level
- this API is a convenience layer over the semisupervised mean engine, not a
  full observational causal pipeline

The full user guide is in [docs/ppi_aipw_vignette.md](docs/ppi_aipw_vignette.md).
There is also a tiny runnable notebook example in [examples/ppi_aipw_quickstart.ipynb](examples/ppi_aipw_quickstart.ipynb).
The package website lives at [docs/index.html](docs/index.html), with a
dedicated causal page at [docs/causal.html](docs/causal.html).

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
# ppi-aipw
