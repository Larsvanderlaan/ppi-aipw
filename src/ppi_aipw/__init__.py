"""ppi_aipw: semisupervised mean inference with AIPW and calibration.

Most users should start with `mean_inference(...)`:

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

Arguments:
    Y:
        Observed outcomes on the labeled sample.
    Yhat:
        Model predictions for the same labeled rows.
    Yhat_unlabeled:
        Model predictions for the unlabeled rows.
    method:
        One of ``"aipw"``, ``"linear"``, ``"prognostic_linear"``, ``"sigmoid"``,
        ``"isotonic"``, ``"monotone_spline"``, or ``"auto"``.
    X, X_unlabeled:
        Optional extra covariates used by ``method="prognostic_linear"``. This
        method fits a semisupervised linear adjustment with an unpenalized
        intercept and prognostic score, plus ridge-tuned coefficients on the
        additional covariates.
    w, w_unlabeled:
        Optional observation weights for the labeled and unlabeled samples.
        These can be inverse probability of missingness weights to adjust for
        informative missingness, or balancing weights if you want the mean
        target to be reweighted toward a covariate-adjusted population.
        Uniform weights reproduce the unweighted behavior.
        The package also exposes
        :func:`compute_two_sample_balancing_weights` for constructing
        nonnegative labeled-sample balancing weights in the two-sample mean
        setting.
    efficiency_maximization:
        Optional rescaling to ``lambda m(X)``, where ``m(X)`` is the predictor
        after the chosen calibration step. The package estimates ``lambda`` by
        empirical influence-function variance minimization. With
        ``method="aipw"``, this is the unrestricted efficiency-maximized
        raw-score correction.
    candidate_methods:
        Candidate methods considered when ``method="auto"``. The package
        exposes :func:`select_mean_method_cv` if you want to inspect the CV
        scores directly.
    num_folds:
        Number of folds used by ``method="auto"``. The default is ``100`` and
        is capped at the number of labeled observations.
    inference:
        Use ``"wald"`` for fast analytic intervals, ``"jackknife"`` for the
        recommended V-fold resampling-based normal interval, or
        ``"bootstrap"`` for percentile bootstrap intervals via
        :func:`aipw_mean_ci`.

Rule of thumb:
    Use ``method="monotone_spline"`` as the package default, ``"aipw"`` for
    uncalibrated AIPW on the original score, ``"linear"`` when you want the
    simplest affine recalibration, ``"prognostic_linear"`` when you want linear
    adjustment on the score plus optional extra covariates, ``"sigmoid"`` when
    predictions are probability-like or bounded scores, and ``"isotonic"`` when
    you want the most flexible monotone calibration curve and have enough
    labeled data to fit it stably. ``method="isotonic"`` uses a one-round
    monotone XGBoost calibrator by default, with
    ``isocal_min_child_weight=10`` and an optional
    ``isocal_backend="sklearn"`` fallback. Turn on
    ``efficiency_maximization=True`` when you want the package to replace the
    chosen predictor by ``lambda m(X)`` using empirical influence-function
    variance minimization, or use ``method="auto"`` to choose among candidate
    methods by cross-validated IF-variance minimization with ``num_folds=100``
    by default. The default automatic shortlist is
    ``("aipw", "linear", "monotone_spline", "isotonic")``. If ``"aipw"`` is
    among the candidates, ``method="auto"`` also compares against an
    efficiency-maximized AIPW candidate. Under
    ``method="auto"``, the foldwise objective uses an unlabeled subset of size
    ``min(n_unlabeled, 10 * n_labeled)`` by default, the selected calibration
    map is refit on the full labeled sample, the final point estimate uses the
    full unlabeled sample, and any final lambda scaling is learned from
    cross-fitted predictions plus that unlabeled subset and reused for Wald
    inference. For resampling-style inference, ``inference="jackknife"`` uses
    a V-fold delete-a-group jackknife with ``jackknife_folds=20`` by default,
    while ``inference="bootstrap"`` remains available as a slower alternative.

Causal API:
    Use ``causal_inference(...)`` when you have treatment-specific predictions
    with one column per arm and you want arm-specific potential outcome means
    plus control-vs-treatment ATEs. Optional covariates ``X`` are passed
    through arm-by-arm, so ``method="prognostic_linear"`` works there too.
    Optional full-sample weights ``w`` are also supported there, including
    inverse propensity weights or balancing weights. In randomized studies,
    they can be used to adjust for chance covariate imbalance and improve
    efficiency; in observational studies, they can be used to adjust for
    confounding and reduce bias. This first version is Wald-only and treats
    each arm-specific mean as a semisupervised mean problem under the hood.
"""

from ._api import (
    MeanInferenceResult,
    aipw_mean_ci,
    aipw_mean_inference,
    aipw_mean_pointestimate,
    aipw_mean_se,
    isotonic_mean_ci,
    isotonic_mean_pointestimate,
    linear_calibration_mean_ci,
    linear_calibration_mean_pointestimate,
    mean_ci,
    mean_inference,
    mean_pointestimate,
    mean_se,
    ppi_aipw_mean_ci,
    ppi_aipw_mean_inference,
    ppi_aipw_mean_pointestimate,
    pi_aipw_mean_ci,
    pi_aipw_mean_inference,
    pi_aipw_mean_pointestimate,
    select_mean_method_cv,
    sigmoid_mean_ci,
    sigmoid_mean_pointestimate,
)
from ._calibration import CalibrationModel, calibrate_predictions, fit_calibrator
from ._causal import CausalInferenceResult, causal_inference
from ._diagnostics import CalibrationCurveDiagnostics, CalibrationDiagnostics, calibration_diagnostics, plot_calibration
from ._weights import compute_two_sample_balancing_weights

__all__ = [
    "CalibrationModel",
    "CalibrationCurveDiagnostics",
    "CalibrationDiagnostics",
    "CausalInferenceResult",
    "MeanInferenceResult",
    "aipw_mean_ci",
    "aipw_mean_inference",
    "aipw_mean_pointestimate",
    "aipw_mean_se",
    "calibration_diagnostics",
    "calibrate_predictions",
    "causal_inference",
    "compute_two_sample_balancing_weights",
    "fit_calibrator",
    "isotonic_mean_ci",
    "isotonic_mean_pointestimate",
    "linear_calibration_mean_ci",
    "linear_calibration_mean_pointestimate",
    "mean_ci",
    "mean_inference",
    "mean_pointestimate",
    "mean_se",
    "ppi_aipw_mean_ci",
    "ppi_aipw_mean_inference",
    "ppi_aipw_mean_pointestimate",
    "pi_aipw_mean_ci",
    "pi_aipw_mean_inference",
    "pi_aipw_mean_pointestimate",
    "plot_calibration",
    "select_mean_method_cv",
    "sigmoid_mean_ci",
    "sigmoid_mean_pointestimate",
]
