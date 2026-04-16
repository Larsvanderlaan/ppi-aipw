# Prediction-Powered Inference via Calibration

Many modern applications involve a small labeled dataset, a much larger unlabeled dataset, and model predictions for far more units than those with observed outcomes. A common example is a small randomized trial paired with a larger observational study or registry: predictions may improve power and precision, but outcome labels are scarce and the prediction model may be imperfect.

`ppi_aipw` is designed for this setting. It uses AIPW (Robins et al., 1994) as a safe and statistically efficient baseline for semisupervised mean inference when outcomes are missing completely at random. AIPW remains valid even when the prediction model is misspecified because it is doubly robust. The package then goes beyond mean-bias correction by adding calibration methods that improve the reliability of the prediction score itself, which can yield more efficient semisupervised inference when the raw predictions are miscalibrated.

[Package Website](https://larsvanderlaan.github.io/ppi-aipw/)
## What calibration means here

From an ML perspective, calibration means the prediction scale is trustworthy, not just the ranking. If a model outputs values near `0.8`, we want outcomes near `0.8` on average for cases scored around `0.8`.

From an inference perspective, calibration matters because this package does not just rank units; it averages predictions and uses them inside AIPW-style estimators. A miscalibrated score can therefore affect bias correction and efficiency, while recalibration can improve semisupervised mean inference without retraining the original model.

This repository contains the code and manuscript assets for our paper on calibration-based semisupervised mean inference. The repository is organized around two experiment pipelines: a synthetic simulation study and a real-data reproduction of the original PPI mean benchmarks, together with a user-facing Python package for semisupervised mean estimation with AIPW, calibration, and practical uncertainty quantification.

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

Install from PyPI:

```bash
python -m pip install ppi-aipw
```

If you want the latest GitHub version instead:

```bash
python -m pip install "git+https://github.com/Larsvanderlaan/ppi-aipw.git"
```

For local development from this checkout:

```bash
python -m pip install -e .
```

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
print(result.summary())
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
    candidate_methods=("aipw", "linear", "monotone_spline", "isotonic"),
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
- for `method="auto"` with jackknife inference, the package selects the method
  once on the original sample and then refits only that chosen method in each
  leave-fold-out refit
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

Optional weights:

- `w` and `w_unlabeled` are available throughout the mean API
- weights are used in the weighted means, calibration/regression fits, method selection, and standard errors
- these can be inverse probability of missingness weights to adjust for informative missingness, or balancing weights if you want to reweight toward a covariate-adjusted target population
- uniform weights reproduce the unweighted behavior
- `compute_two_sample_balancing_weights(X_labeled, X_unlabeled, ...)` computes nonnegative labeled-sample balancing weights for the two-sample mean problem
- this utility is intended for the semisupervised mean setup and does not construct causal-arm-specific weights

Available interval types:

- `inference="wald"`: fast analytic interval
- `inference="jackknife"`: recommended resampling-style interval using a V-fold delete-a-group jackknife with `jackknife_folds=20` by default
- `inference="bootstrap"`: percentile bootstrap interval that refits calibration in each resample

Recommended defaults:

- start with `method="monotone_spline"`
- start with `inference="wald"`
- use `method="linear"` when you want the simplest affine recalibration and maximum interpretability
- use `method="prognostic_linear"` when you want a linear prognostic adjustment with optional extra covariates `X`
- use `method="sigmoid"` when predictions are probability-like or naturally bounded
- use `method="monotone_spline"` when you want the default smooth monotone nonlinear recalibration that is less stepwise than isotonic
- use `method="isotonic"` when you expect monotone nonlinear miscalibration and have enough labeled data; the default backend is monotone XGBoost, with `isocal_backend="sklearn"` available as a simpler fallback
- use `efficiency_maximization=True` when you want the package to empirically rescale the score for lower variance, especially for raw-score `method="aipw"`
- use `method="auto"` when you want a data-adaptive choice among `aipw`, `linear`, `monotone_spline`, and `isotonic`, while also letting the selector compare against an efficiency-maximized AIPW candidate
- use `num_folds=100` as the default auto-selection setting unless you have a reason to make it smaller
- use `inference="jackknife", jackknife_folds=20` as the default resampling-style uncertainty check
- use `inference="bootstrap"` when you specifically want percentile bootstrap intervals and can afford extra compute

Result object:

- `result.pointestimate`: point estimate
- `result.se`: standard error
- `result.ci`: confidence interval for the requested `alpha` and `alternative`
- `result.method`: selected base method
- `result.selected_candidate`: selected auto candidate label
- `result.efficiency_lambda`: estimated lambda when efficiency maximization is active
- `result.diagnostics`: method-selection and inference metadata
- `result.summary()`: human-readable summary including the Wald t-statistic and p-value for the default null `0`

Calibration diagnostics helpers:

- `calibration_diagnostics(result_or_model, Y, Yhat, diagnostic_mode="out_of_fold", num_folds=10, ...)`: optional honest held-out calibration diagnostics
- `plot_calibration(diagnostics, ...)`: optional plotting helper built on those diagnostics; install `matplotlib` or the optional `plot` extra if you want the plotting convenience
- `diagnostics.summary()`: compact calibrated-BLP slope summary with coefficient, CI, and p-value against the default calibration null `slope = 1`

How to read the diagnostic plot:

- the fitted curve is the score-to-outcome map implied by the fitted calibrator
- the raw-score points place each bin's mean outcome at that bin's mean raw score
- the calibrated-score points place that same bin mean outcome at the bin's mean calibrated score
- horizontal movement toward the identity line means the recalibrated score is on a better scale

Causal API:

- `causal_inference(Y, A, Yhat_potential, ...)` computes arm-specific potential
  outcome means plus all control-vs-treatment ATEs in one Wald-only call
- `Yhat_potential` should have one column per treatment arm and one row per unit
- optional `X` is passed through arm-by-arm, so `method="prognostic_linear"`
  gives the same score-plus-covariate linear adjustment inside each arm
- optional full-sample weights `w` are also supported and can be inverse
  propensity weights or balancing weights; in randomized studies they can
  adjust for chance covariate imbalance for efficiency, and in observational
  studies they can adjust for confounding for bias reduction
- the returned causal result object exposes `arm_means`, `ate`, interval fields,
  and `result.summary()` for a compact Wald summary of both arm means and
  control-vs-treatment effects
- `control_arm=None` defaults to the minimum resolved treatment level
- this API is a convenience layer over the semisupervised mean engine, not a
  full observational causal pipeline

The full user guide is in [docs/ppi_aipw_vignette.md](docs/ppi_aipw_vignette.md).
There is also a compact runnable notebook covering both the mean and causal APIs in [examples/ppi_aipw_quickstart.ipynb](examples/ppi_aipw_quickstart.ipynb), and it can be opened directly in Colab at [colab.research.google.com/github/Larsvanderlaan/ppi-aipw/blob/main/examples/ppi_aipw_quickstart.ipynb](https://colab.research.google.com/github/Larsvanderlaan/ppi-aipw/blob/main/examples/ppi_aipw_quickstart.ipynb). The notebook installs `ppi-aipw` automatically in Colab.
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

## References

- Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). "Estimation of regression coefficients when some regressors are not always observed." *Journal of the American Statistical Association* 89(427): 846-866.
- Rubin, D. B. and van der Laan, M. J. (2008). "Empirical efficiency maximization: improved locally efficient covariate adjustment in randomized experiments and survival analysis." *The International Journal of Biostatistics* 4(1).
- van der Laan, L. and van der Laan, M. (2024). "Prediction-Powered Inference via Calibration." [arXiv:2411.02771](https://arxiv.org/pdf/2411.02771).
