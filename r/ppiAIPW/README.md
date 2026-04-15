# ppiAIPW

`ppiAIPW` is the native R package for semisupervised mean inference with AIPW
and calibration.

It takes labeled outcomes, labeled predictions, and unlabeled predictions, and
returns point estimates, standard errors, intervals, diagnostics, and a fitted
calibrator in one workflow.

## Install

From GitHub:

```r
remotes::install_github("Larsvanderlaan/ppi-aipw", subdir = "r/ppiAIPW")
```

From a local checkout:

```r
remotes::install_local("r/ppiAIPW")
```

All numeric inputs to the R mean, calibration, diagnostics, weighting, and
causal APIs must be finite. `NaN`, `NA`, and `Inf` values are rejected with
clear validation errors.

## Quickstart

```r
library(ppiAIPW)

set.seed(1)
Y <- rnorm(80)
Yhat <- Y + rnorm(80, sd = 0.35)
Yhat_unlabeled <- rnorm(200)

result <- mean_inference(
  Y,
  Yhat,
  Yhat_unlabeled,
  method = "monotone_spline",
  alpha = 0.1
)

print(result)
summary(result)
```

Use `method = "auto"` if you want the package to choose among a short list of
candidate methods.

```r
auto_result <- mean_inference(
  Y,
  Yhat,
  Yhat_unlabeled,
  method = "auto",
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = 5,
  selection_random_state = 0
)
```

## Main functions

- `mean_inference()`
- `mean_pointestimate()`
- `mean_se()`
- `mean_ci()`
- `fit_calibrator()`
- `calibrate_predictions()`
- `select_mean_method_cv()`
- `causal_inference()`

See the package vignettes in [`vignettes/`](./vignettes).
