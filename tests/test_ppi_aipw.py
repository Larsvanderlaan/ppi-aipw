from __future__ import annotations

import numpy as np

from ppi_aipw import (
    aipw_mean_ci,
    aipw_mean_pointestimate,
    aipw_mean_se,
    calibrate_predictions,
    fit_calibrator,
    isocal_mean_pointestimate,
    linear_calibration_mean_pointestimate,
    mean_ci,
    mean_pointestimate,
    mean_se,
    ppi_aipw_mean_pointestimate,
    platt_scaling_mean_pointestimate,
)


def test_aipw_matches_manual_augmented_estimator() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat = np.array([0.1, 0.4, 0.6, 0.8], dtype=float)
    yhat_unlabeled = np.array([0.2, 0.3, 0.7], dtype=float)

    estimate = aipw_mean_pointestimate(y, yhat, yhat_unlabeled)
    rho = len(y) / (len(y) + len(yhat_unlabeled))
    expected = rho * yhat.mean() + (1.0 - rho) * yhat_unlabeled.mean() + np.mean(y - yhat)

    np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-12)


def test_linear_calibration_is_identity_on_perfect_labeled_predictions() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat_unlabeled = np.array([0.2, 0.8, 0.4], dtype=float)
    pred_labeled, pred_unlabeled = calibrate_predictions(y, y, yhat_unlabeled, method="linear")

    np.testing.assert_allclose(pred_labeled, y)
    np.testing.assert_allclose(pred_unlabeled, yhat_unlabeled)


def test_platt_scaling_rescales_nonbinary_outcomes_into_observed_range() -> None:
    y = np.array([10.0, 11.5, 13.0, 16.0, 18.0, 20.0], dtype=float)
    yhat = np.array([8.0, 10.5, 12.0, 14.5, 17.5, 22.0], dtype=float)
    yhat_unlabeled = np.array([7.0, 9.0, 13.0, 15.0, 19.0, 21.0], dtype=float)

    pred_labeled, pred_unlabeled = calibrate_predictions(y, yhat, yhat_unlabeled, method="platt")

    assert np.all(pred_labeled >= y.min())
    assert np.all(pred_labeled <= y.max())
    assert np.all(pred_unlabeled >= y.min())
    assert np.all(pred_unlabeled <= y.max())
    assert np.all(np.diff(pred_unlabeled[np.argsort(yhat_unlabeled)]) >= -1e-12)


def test_isocal_predictions_are_bounded_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_unlabeled = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    pred_labeled, pred_unlabeled = calibrate_predictions(y, yhat, yhat_unlabeled, method="isocal")

    assert np.all(pred_labeled >= 0.0)
    assert np.all(pred_labeled <= 1.0)
    assert np.all(pred_unlabeled >= 0.0)
    assert np.all(pred_unlabeled <= 1.0)
    assert np.all(np.diff(pred_labeled[np.argsort(yhat)]) >= -1e-12)


def test_vector_outputs_and_aliases_match() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(30, 2))
    yhat = y + rng.normal(scale=0.3, size=(30, 2))
    yhat_unlabeled = rng.normal(size=(50, 2))

    estimate = ppi_aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="linear")
    lower, upper = aipw_mean_ci(y, yhat, yhat_unlabeled, method="linear", alpha=0.1)

    assert estimate.shape == (2,)
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(lower <= upper)


def test_high_level_wrappers_agree_with_generic_api() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.4, size=40)
    yhat_unlabeled = rng.normal(size=100)

    linear_generic = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="linear")
    linear_wrapper = linear_calibration_mean_pointestimate(y, yhat, yhat_unlabeled)
    isocal_generic = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="isocal")
    isocal_wrapper = isocal_mean_pointestimate(y, yhat, yhat_unlabeled)
    platt_generic = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="platt")
    platt_wrapper = platt_scaling_mean_pointestimate(y, yhat, yhat_unlabeled)

    np.testing.assert_allclose(linear_generic, linear_wrapper)
    np.testing.assert_allclose(isocal_generic, isocal_wrapper)
    np.testing.assert_allclose(platt_generic, platt_wrapper)


def test_plain_aliases_agree_with_generic_api() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.35, size=40)
    yhat_unlabeled = rng.normal(size=100)

    estimate = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="linear")
    lower, upper = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=40,
        random_state=0,
    )
    se = aipw_mean_se(y, yhat, yhat_unlabeled, method="linear")

    np.testing.assert_allclose(mean_pointestimate(y, yhat, yhat_unlabeled, method="linear"), estimate)
    alias_lower, alias_upper = mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=40,
        random_state=0,
    )
    np.testing.assert_allclose(alias_lower, lower)
    np.testing.assert_allclose(alias_upper, upper)
    np.testing.assert_allclose(mean_se(y, yhat, yhat_unlabeled, method="linear"), se)


def test_fit_calibrator_returns_reusable_model() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.1, 0.3, 0.4, 0.8, 0.9], dtype=float)
    model = fit_calibrator(y, yhat, method="linear")
    pred = model.predict(np.array([0.2, 0.7], dtype=float))

    assert pred.shape == (2,)
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)


def test_bootstrap_ci_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=30)
    yhat = y + rng.normal(scale=0.4, size=30)
    yhat_unlabeled = rng.normal(size=80)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=60,
        random_state=0,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=60,
        random_state=0,
    )

    np.testing.assert_allclose(ci_one[0], ci_two[0])
    np.testing.assert_allclose(ci_one[1], ci_two[1])
    assert ci_one[0] <= ci_one[1]


def test_bootstrap_refits_isocal_and_returns_positive_se() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_unlabeled = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.7], dtype=float)

    lower, upper = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="isocal",
        inference="bootstrap",
        n_resamples=50,
        random_state=1,
    )
    se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="isocal",
        inference="bootstrap",
        n_resamples=50,
        random_state=1,
    )

    assert lower <= upper
    assert se > 0.0
