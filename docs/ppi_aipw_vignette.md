# `ppi_aipw` Vignette

`ppi_aipw` is a practitioner-friendly package for estimating a population mean
when you have:

- a labeled sample with observed outcomes `Y`
- model predictions on those same labeled rows, `Yhat`
- a larger unlabeled sample where you only have predictions, `Yhat_unlabeled`

The package uses AIPW-style augmentation: it starts from a prediction-based mean
and then debiases it using the labeled sample. It sits in the broader
semiparametric, missing-data, and debiased machine learning toolkit.

This vignette is written for applied ML and practitioner workflows. It focuses on
what to call, what the arguments mean, and how to choose among the available
calibration methods.

## Start Here

The main recommended function is:

- `mean_inference`

It returns the point estimate, standard error, confidence interval, fitted
calibrator, and auto-selection diagnostics in one pass:

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

If you prefer narrower wrappers, the package also exports:

- `mean_pointestimate`
- `mean_ci`
- `mean_se`
- `aipw_mean_pointestimate`
- `aipw_mean_ci`
- `aipw_mean_se`

Typical wrapper usage:

```python
from ppi_aipw import mean_pointestimate, mean_ci

estimate = mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
)

lower, upper = mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    alpha=0.1,
)
```

If you only remember one pattern, remember `result = mean_inference(...)`.

A tiny notebook version of this workflow is in
`examples/ppi_aipw_quickstart.ipynb`.

## What The Arguments Mean

The core wrapper API follows the same basic shape everywhere:

```python
aipw_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    w=None,
    w_unlabeled=None,
)
```

The one-call result-object API follows the same core arguments:

```python
mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    alpha=0.1,
    inference="wald",
)
```

Argument guide:

- `Y`: observed outcomes for the labeled sample
- `Yhat`: predictions for those exact same labeled rows
- `Yhat_unlabeled`: predictions for the unlabeled rows
- `method`: which calibration strategy to use before the AIPW augmentation step
- `w`: optional weights for the labeled sample
- `w_unlabeled`: optional weights for the unlabeled sample
- `X`, `X_unlabeled`: optional extra covariates used by
  `method="prognostic_linear"`; the intercept and score stay unpenalized,
  while the extra covariates are ridge-tuned on the labeled sample
- `efficiency_maximization`: optional rescaling to `lambda m(X)`, where `m(X)`
  is the predictor after the chosen calibration step and `lambda` is estimated
  by empirical influence-function variance minimization
- `candidate_methods`: candidate methods used when `method="auto"` selects by
  cross-validated influence-function variance minimization; if `"aipw"` is in
  the list, the selector also compares against an efficiency-maximized AIPW
  candidate
- `num_folds`: number of folds used when `method="auto"`; the default is `100`
  and it is capped at the labeled sample size
- `auto_unlabeled_subsample_size`: unlabeled subset size used by
  `method="auto"` during foldwise selection and cross-fitted lambda
  estimation; the default is `min(n_unlabeled, 10 * n_labeled)`
- `isocal_backend`: backend used by `method="isotonic"`; choose `"xgboost"`
  for the default one-round monotone XGBoost calibrator or `"sklearn"` for
  scikit-learn isotonic regression
- `isocal_min_child_weight`: `min_child_weight` for the default XGBoost
  isotonic backend; the package default is `10`

Shape rules:

- for scalar outcomes, use 1D arrays
- for multi-output outcomes, use 2D arrays with shape `(n, d)`
- `Y` and `Yhat` must have the same shape
- `Yhat_unlabeled` must have the same number of columns as `Y`

## Which Method Should I Use?

Available `method=` values:

- `"aipw"`
- `"linear"`
- `"prognostic_linear"`
- `"sigmoid"`
- `"monotone_spline"`
- `"isotonic"`
- `"auto"`

Here is the practical summary.

### `method="prognostic_linear"`

What it does:

- Fits a semisupervised linear adjustment using the prediction score plus
  optional extra covariates `X`.
- Leaves the intercept and prognostic-score coefficient unpenalized.
- Uses ridge tuning only on the additional covariates.

When it is a good choice:

- You want the score to stay in the model as the main prognostic feature.
- You have additional covariates that may explain residual outcome variation.
- You want a transparent regression-style adjustment rather than a nonlinear
  calibration map.

Tradeoffs:

- More expressive than score-only linear recalibration when `X` matters.
- Still easy to inspect and explain.
- Requires `X` and `X_unlabeled` to take advantage of the extra covariate layer.

## Optional Efficiency Maximization

Set `efficiency_maximization=True` if you want the package to replace the
predictor used by the estimator by `lambda m(X)`, where `m(X)` is the raw score
for `method="aipw"` and the calibrated score for the other methods. The package
estimates `lambda` from the data by empirical influence-function variance
minimization.

```python
estimate = mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="aipw",
    efficiency_maximization=True,
)
```

Practical interpretation:

- with `method="aipw"`, this gives the unrestricted efficiency-maximized
  raw-score correction
- with other `method=` choices, it applies the same `lambda m(X)` rescaling to
  the calibrated predictor
- for multi-output outcomes, the package computes the lambda separately for each
  output coordinate

When to use it:

- you want a direct efficiency-maximized alternative to plain raw-score AIPW
- you want the package to adapt the score scale automatically instead of fixing it
- you want a simple variance-focused check without changing the calibration family

## Automatic Method Selection

Set `method="auto"` if you want the package to choose among a candidate set of
methods by cross-validated influence-function variance minimization.

```python
estimate = mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="auto",
    candidate_methods=("aipw", "linear", "isotonic"),
    num_folds=100,
)
```

What it does:

- fits each candidate method on training folds of the labeled sample
- if `"aipw"` is among the candidates, also compares against an
  efficiency-maximized AIPW candidate
- evaluates them using a cross-validated estimate of the influence-function variance
- uses an unlabeled subset of size `min(n_unlabeled, 10 * n_labeled)` by
  default in the foldwise objective
- refits the selected calibration map on the full labeled sample before
  computing the final estimate, then uses the full unlabeled sample for the
  final point estimate
- if the final procedure uses efficiency maximization, learns `lambda` from the
  selected method's cross-fitted labeled predictions and unlabeled-subset
  predictions, then reuses that cross-fitted lambda for the point
  estimate and Wald variance
- for bootstrap inference, selects the method once on the original sample and
  then bootstraps only that chosen method

This means `mean_inference(..., method="auto")` returns:

- the full-sample point estimate for the selected method
- a Wald SE that reuses the same cross-fitted layer used to learn `lambda`
- bootstrap SEs and intervals that select once, then refit only the chosen method
- `result.diagnostics` with the selected candidate, unlabeled subset size, and related metadata

Use it when:

- you want a small data-adaptive comparison among a few plausible methods
- you want selection aligned with efficiency rather than plain calibration loss
- you want to let plain AIPW compete against an efficiency-maximized AIPW
  variant without manually adding a separate method name

Implementation note:

- `method="isotonic"` uses a one-round monotone XGBoost calibrator by default,
  with `isocal_min_child_weight=10`
- if you want the classic scikit-learn isotonic fit instead, pass
  `isocal_backend="sklearn"`

### `method="aipw"`

What it does:

- Uses the raw predictions directly.
- Applies AIPW augmentation with no calibration layer.

When it is a good choice:

- You already trust the scale of the predictions.
- You want the cleanest baseline.
- You want to compare calibration methods against an uncalibrated AIPW estimator.

Tradeoffs:

- Simple and stable.
- Best when the model is already well calibrated.
- Can leave efficiency on the table if predictions are systematically mis-scaled.

### `method="linear"`

What it does:

- Fits a straight-line mapping from predictions to observed outcomes on the labeled sample.
- Runs AIPW using those recalibrated predictions.

When it is a good choice:

- You want a strong default.
- You think the model is mostly right but maybe shifted or stretched.
- You do not have a huge labeled calibration sample.

Tradeoffs:

- Usually the easiest method to trust and debug.
- More stable than highly flexible calibration with small labeled samples.
- Cannot capture strongly nonlinear calibration errors.

### `method="sigmoid"`

What it does:

- Fits a sigmoid-shaped calibration map.
- For nonbinary outcomes, it first rescales outcomes into the observed labeled range, fits the sigmoid map there, and rescales back.

When it is a good choice:

- Predictions behave like probabilities or bounded scores.
- You want a smooth monotone calibration map.
- You expect a saturating S-shaped calibration pattern.

Tradeoffs:

- More structured than isotonic calibration, so it can be more stable with limited data.
- Less flexible than isotonic calibration.
- If the true calibration curve is not close to sigmoid-shaped, it may underfit.

### `method="monotone_spline"`

What it does:

- Fits a smooth monotone spline calibration map from predictions to outcomes.
- Preserves monotonicity while avoiding the step-function shape of isotonic calibration.

When it is a good choice:

- You expect monotone nonlinear miscalibration.
- You want something smoother than isotonic calibration.
- You want a middle ground between linear and isotonic recalibration.

Tradeoffs:

- More flexible than linear calibration.
- Smoother and often easier to interpret than isotonic calibration.
- Still more complex than linear calibration, so it can be less stable with very tiny labeled samples.

### `method="isotonic"`

What it does:

- Fits an isotonic, monotone calibration map from predictions to outcomes.
- By default this is implemented with a one-round monotone XGBoost fit; you can
  switch to `isocal_backend="sklearn"` for scikit-learn isotonic regression.
- Preserves ordering but allows nonlinear adjustments.

When it is a good choice:

- You believe predictions are directionally right but not on the right scale.
- You expect monotone but nonlinear miscalibration.
- You have enough labeled data to support a more flexible map.

Tradeoffs:

- Most flexible option in the package.
- Often attractive when monotonicity is believable and calibration is visibly nonlinear.
- Can be less stable than linear or sigmoid calibration when the labeled sample is small.

## Recommended Starting Point

If you are not sure what to use:

1. Start with `method="monotone_spline"`.
2. Compare against `method="aipw"` as a baseline.
3. Try `method="linear"` if you want the simplest affine recalibration.
4. Try `method="isotonic"` if you have enough labeled data and suspect nonlinear miscalibration.
5. Try `method="sigmoid"` when predictions are probability-like or naturally bounded.

In practice, `monotone_spline` is now the package's default first choice for applied ML workflows when a smooth monotone recalibration is plausible.

## Recommended Defaults

If you want a practical default recipe for real work:

- use `method="monotone_spline"`
- use `inference="wald"`

Why this is the default recommendation:

- `monotone_spline` keeps the monotonicity structure many score problems suggest, while staying smoother and less stepwise than isotonic calibration
- it performed well in the package’s quick simulation and benchmark checks
- `wald` is much faster than bootstrap and is a good first-pass interval in routine workflows

When to switch away from the defaults:

- switch to `method="sigmoid"` when your predictions are probability-like, bounded, or naturally interpreted through a smooth sigmoid recalibration
- switch to `method="monotone_spline"` when you want a smooth monotone nonlinear recalibration instead of a stepwise isotonic curve
- switch to `method="isotonic"` when you believe calibration is monotone but clearly nonlinear, and you have enough labeled data to fit it stably
- switch to `inference="bootstrap"` when you want a more empirical uncertainty check and can afford the extra computation

Short version:

- default: `method="monotone_spline", inference="wald"`
- simplest affine recalibration: try `method="linear"`
- bounded or probability-like scores: try `method="sigmoid"`
- smooth monotone nonlinear recalibration: try `method="monotone_spline"`
- more flexible monotone recalibration: try `method="isotonic"`
- extra uncertainty robustness check: try `inference="bootstrap"`

## Confidence Intervals

Use `aipw_mean_ci` for uncertainty intervals:

```python
from ppi_aipw import aipw_mean_ci

lower, upper = aipw_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    alpha=0.05,
    inference="wald",
)
```

Interpretation:

- `alpha=0.05` gives a 95% interval
- `alpha=0.1` gives a 90% interval

Available interval types:

- `inference="wald"`: faster and usually a good default when you want something simple
- `inference="bootstrap"`: more computationally expensive, but often easier to explain to applied users because it directly resamples the data and refits calibration each time

Bootstrap behavior in this package:

- the prediction model is treated as fixed
- the labeled rows are resampled with replacement
- the unlabeled rows are resampled with replacement
- the calibration step is refit inside each bootstrap sample

Example bootstrap interval:

```python
lower, upper = aipw_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="isotonic",
    inference="bootstrap",
    n_resamples=1000,
    random_state=0,
)
```

## Shortcut Functions

If you prefer explicit method names instead of `method=...`, these wrappers are available:

- `linear_calibration_mean_pointestimate`
- `linear_calibration_mean_ci`
- `sigmoid_mean_pointestimate`
- `sigmoid_mean_ci`
- `isotonic_mean_pointestimate`
- `isotonic_mean_ci`

Example:

```python
from ppi_aipw import isotonic_mean_pointestimate

estimate = isotonic_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
```

These are just convenience wrappers around the generic API.

## Causal Wrapper

If you have treatment-specific predictions with one column per arm, the package
also exposes a thin causal wrapper:

```python
from ppi_aipw import causal_inference

result = causal_inference(
    Y,
    A,
    Yhat_potential,
    treatment_levels=("control", "treated"),
    method="monotone_spline",
    alpha=0.1,
)

mu_control = result.arm_means["control"]
mu_treated = result.arm_means["treated"]
ate = result.ate["treated"]
```

What it does:

- treats each arm-specific mean as a semisupervised mean problem
- uses units in the target arm as labeled and units outside the target arm as unlabeled
- returns arm-specific Wald intervals and all control-vs-treatment ATEs

Important notes:

- this first version supports one-dimensional outcomes
- `Yhat_potential` must have one column per treatment arm
- `control_arm=None` defaults to the minimum resolved treatment level
- `inference="wald"` is the only supported option in the causal wrapper

## Reusing A Fitted Calibration Model

If you want to inspect or reuse the calibration step directly:

```python
from ppi_aipw import fit_calibrator

calibrator = fit_calibrator(Y, Yhat)
calibrated_labeled = calibrator.predict(Yhat)
calibrated_unlabeled = calibrator.predict(Yhat_unlabeled)
```

This is useful when you want to:

- inspect the calibrated predictions themselves
- compare the raw and calibrated predictions
- reuse the same calibration model across downstream analyses

## Weighted Data

If your labeled or unlabeled samples need observation weights:

```python
estimate = aipw_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    w=labeled_weights,
    w_unlabeled=unlabeled_weights,
)
```

The weights should be 1D arrays matching the number of rows in the corresponding sample.

## Advanced Controls

The package intentionally keeps the public AIPW interface small.

The main practitioner-facing choices are:

- `method` for calibration
- `inference` for uncertainty estimation
- optional sample weights

This is deliberate: the goal is to keep the default workflow easy to explain and easy to use.

## End-To-End Example

```python
import numpy as np
from ppi_aipw import aipw_mean_pointestimate, aipw_mean_ci

rng = np.random.default_rng(0)

n_labeled = 200
n_unlabeled = 1000

Y = rng.normal(loc=10.0, scale=2.0, size=n_labeled)
Yhat = Y + rng.normal(scale=1.0, size=n_labeled)
Yhat_unlabeled = rng.normal(loc=10.1, scale=1.8, size=n_unlabeled)

estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
lower, upper = aipw_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1)

print("estimate:", estimate)
print("90% CI:", (lower, upper))
```

## Summary

If you want the shortest possible recommendation:

- call `aipw_mean_pointestimate` and `aipw_mean_ci`
- start with `method="monotone_spline"`
- use `method="linear"` when you want the simplest affine recalibration
- turn on `efficiency_maximization=True` when you want the score scale chosen by empirical influence-function variance minimization
- under `method="auto"`, remember that the full unlabeled sample is reused in
  every fold and any final lambda is learned from cross-fitted predictions
- compare to `method="aipw"` as a baseline
- try `method="monotone_spline"` for smooth monotone nonlinear calibration
- try `method="isotonic"` for monotone nonlinear calibration
- try `method="sigmoid"` for bounded or probability-like scores
