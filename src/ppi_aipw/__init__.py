"""ppi_aipw: semisupervised AIPW / DML-style mean inference with easy calibration.

Most users should start with these two functions:

```python
from ppi_aipw import aipw_mean_pointestimate, aipw_mean_ci

estimate = aipw_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
)

lower, upper = aipw_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="linear",
    alpha=0.1,
)
```

Arguments:
    Y:
        Observed outcomes on the labeled sample.
    Yhat:
        Model predictions for the same labeled rows.
    Yhat_unlabeled:
        Model predictions for the unlabeled rows.
    method:
        One of ``"aipw"``, ``"linear"``, ``"platt"``, or ``"isocal"``.
    inference:
        Use ``"wald"`` for fast analytic intervals or ``"bootstrap"`` for
        resampling-based intervals via :func:`aipw_mean_ci`.

Rule of thumb:
    Use ``method="linear"`` as a strong default, ``"aipw"`` as the no-calibration
    baseline, ``"platt"`` when predictions are probability-like or bounded scores,
    and ``"isocal"`` when you expect a monotone but nonlinear calibration curve and
    have enough labeled data to fit it stably.
"""

from ._api import (
    aipw_mean_ci,
    aipw_mean_pointestimate,
    aipw_mean_se,
    isocal_mean_ci,
    isocal_mean_pointestimate,
    linear_calibration_mean_ci,
    linear_calibration_mean_pointestimate,
    mean_ci,
    mean_pointestimate,
    mean_se,
    ppi_aipw_mean_ci,
    ppi_aipw_mean_pointestimate,
    pi_aipw_mean_ci,
    pi_aipw_mean_pointestimate,
    platt_scaling_mean_ci,
    platt_scaling_mean_pointestimate,
)
from ._calibration import CalibrationModel, calibrate_predictions, fit_calibrator

__all__ = [
    "CalibrationModel",
    "aipw_mean_ci",
    "aipw_mean_pointestimate",
    "aipw_mean_se",
    "calibrate_predictions",
    "fit_calibrator",
    "isocal_mean_ci",
    "isocal_mean_pointestimate",
    "linear_calibration_mean_ci",
    "linear_calibration_mean_pointestimate",
    "mean_ci",
    "mean_pointestimate",
    "mean_se",
    "ppi_aipw_mean_ci",
    "ppi_aipw_mean_pointestimate",
    "pi_aipw_mean_ci",
    "pi_aipw_mean_pointestimate",
    "platt_scaling_mean_ci",
    "platt_scaling_mean_pointestimate",
]
