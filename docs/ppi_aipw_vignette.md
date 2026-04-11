# `ppi_aipw` Vignette

`ppi_aipw` is a small package for estimating a population mean when you have:

- a labeled sample with observed outcomes `Y`
- model predictions on those same labeled rows, `Yhat`
- a larger unlabeled sample where you only have predictions, `Yhat_unlabeled`

The package uses AIPW-style augmentation: it starts from a prediction-based mean
and then debiases it using the labeled sample.

This vignette is written for applied ML and practitioner workflows. It focuses on
what to call, what the arguments mean, and how to choose among the available
calibration methods.

## Start Here

The two main functions are:

- `aipw_mean_pointestimate`
- `aipw_mean_ci`

If you prefer plainer names, the package also exports:

- `mean_pointestimate`
- `mean_ci`
- `mean_se`

Typical usage:

```python
from ppi_aipw import mean_pointestimate, mean_ci

estimate = mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
)

lower, upper = mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
    alpha=0.1,
)
```

If you only remember one pattern, remember that one.

A tiny notebook version of this workflow is in
`examples/ppi_aipw_quickstart.ipynb`.

## What The Arguments Mean

The core API follows the same basic shape everywhere:

```python
aipw_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
    w=None,
    w_unlabeled=None,
)
```

Argument guide:

- `Y`: observed outcomes for the labeled sample
- `Yhat`: predictions for those exact same labeled rows
- `Yhat_unlabeled`: predictions for the unlabeled rows
- `method`: which calibration strategy to use before the AIPW augmentation step
- `w`: optional weights for the labeled sample
- `w_unlabeled`: optional weights for the unlabeled sample

Shape rules:

- for scalar outcomes, use 1D arrays
- for multi-output outcomes, use 2D arrays with shape `(n, d)`
- `Y` and `Yhat` must have the same shape
- `Yhat_unlabeled` must have the same number of columns as `Y`

## Which Method Should I Use?

Available `method=` values:

- `"aipw"`
- `"linear"`
- `"platt"`
- `"isocal"`

Here is the practical summary.

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

### `method="platt"`

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

### `method="isocal"`

What it does:

- Fits an isotonic, monotone calibration map from predictions to outcomes.
- Preserves ordering but allows nonlinear adjustments.

When it is a good choice:

- You believe predictions are directionally right but not on the right scale.
- You expect monotone but nonlinear miscalibration.
- You have enough labeled data to support a more flexible map.

Tradeoffs:

- Most flexible option in the package.
- Often attractive when monotonicity is believable and calibration is visibly nonlinear.
- Can be less stable than linear or Platt scaling when the labeled sample is small.

## Recommended Starting Point

If you are not sure what to use:

1. Start with `method="linear"`.
2. Compare against `method="aipw"` as a baseline.
3. Try `method="isocal"` if you have enough labeled data and suspect nonlinear miscalibration.
4. Try `method="platt"` when predictions are probability-like or naturally bounded.

In practice, `linear` is often the safest first choice for applied ML workflows.

## Recommended Defaults

If you want a practical default recipe for real work:

- use `method="linear"`
- use `inference="wald"`

Why this is the default recommendation:

- `linear` is usually the easiest calibration method to explain, debug, and trust
- it performed well in the package’s quick simulation checks
- `wald` is much faster than bootstrap and is a good first-pass interval in routine workflows

When to switch away from the defaults:

- switch to `method="platt"` when your predictions are probability-like, bounded, or naturally interpreted through a smooth sigmoid recalibration
- switch to `method="isocal"` when you believe calibration is monotone but clearly nonlinear, and you have enough labeled data to fit it stably
- switch to `inference="bootstrap"` when you want a more empirical uncertainty check and can afford the extra computation

Short version:

- default: `method="linear", inference="wald"`
- bounded or probability-like scores: try `method="platt"`
- more flexible monotone recalibration: try `method="isocal"`
- extra uncertainty robustness check: try `inference="bootstrap"`

## Confidence Intervals

Use `aipw_mean_ci` for uncertainty intervals:

```python
from ppi_aipw import aipw_mean_ci

lower, upper = aipw_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
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
    method="isocal",
    inference="bootstrap",
    n_resamples=1000,
    random_state=0,
)
```

## Shortcut Functions

If you prefer explicit method names instead of `method=...`, these wrappers are available:

- `linear_calibration_mean_pointestimate`
- `linear_calibration_mean_ci`
- `platt_scaling_mean_pointestimate`
- `platt_scaling_mean_ci`
- `isocal_mean_pointestimate`
- `isocal_mean_ci`

Example:

```python
from ppi_aipw import isocal_mean_pointestimate

estimate = isocal_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
```

These are just convenience wrappers around the generic API.

## Reusing A Fitted Calibration Model

If you want to inspect or reuse the calibration step directly:

```python
from ppi_aipw import fit_calibrator

calibrator = fit_calibrator(Y, Yhat, method="linear")
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
    method="linear",
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

estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled, method="linear")
lower, upper = aipw_mean_ci(Y, Yhat, Yhat_unlabeled, method="linear", alpha=0.1)

print("estimate:", estimate)
print("90% CI:", (lower, upper))
```

## Summary

If you want the shortest possible recommendation:

- call `aipw_mean_pointestimate` and `aipw_mean_ci`
- start with `method="linear"`
- compare to `method="aipw"` as a baseline
- try `method="isocal"` for monotone nonlinear calibration
- try `method="platt"` for bounded or probability-like scores
