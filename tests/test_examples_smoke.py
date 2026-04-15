from __future__ import annotations

import numpy as np

from ppi_aipw import calibration_diagnostics, causal_inference, mean_inference


def test_mean_quickstart_example_smoke() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=30)
    yhat = y + rng.normal(scale=0.25, size=30)
    yhat_unlabeled = rng.normal(size=80)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="monotone_spline",
        alpha=0.1,
    )

    assert np.isfinite(float(result.pointestimate))
    assert np.isfinite(float(result.se))
    assert result.ci[0] <= result.ci[1]


def test_auto_and_calibration_diagnostics_example_smoke() -> None:
    rng = np.random.default_rng(456)
    yhat = np.linspace(-1.0, 1.0, 40)
    y = 0.4 + 1.7 * yhat + rng.normal(scale=0.08, size=40)
    yhat_unlabeled = np.linspace(-1.2, 1.2, 120)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "monotone_spline", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )
    diagnostics = calibration_diagnostics(result, y, yhat, num_bins=5)

    assert result.method in {"aipw", "linear", "monotone_spline", "isotonic"}
    assert diagnostics.method == result.calibrator.method
    assert diagnostics.n_labeled == y.shape[0]


def test_causal_quickstart_example_smoke() -> None:
    rng = np.random.default_rng(789)
    n = 60
    x = rng.normal(size=(n, 2))
    a = rng.integers(0, 2, size=n)
    mu0 = 0.3 + 0.6 * x[:, 0] - 0.2 * x[:, 1]
    mu1 = mu0 + 0.8
    y = np.where(a == 1, mu1, mu0) + rng.normal(scale=0.2, size=n)
    yhat_potential = np.column_stack([mu0 + 0.05, mu1 - 0.05])

    result = causal_inference(
        y,
        a,
        yhat_potential,
        method="linear",
        alpha=0.1,
    )

    assert 1 in result.ate
    assert np.isfinite(float(result.arm_means[0]))
    assert np.isfinite(float(result.ate[1]))
