from __future__ import annotations

import numpy as np
import pytest

from ppi_aipw._api import (
    _predict_prognostic_linear_from_coef,
    _prepare_auto_variance_inputs,
    _prepare_mean_estimation_inputs,
    _select_mean_method_cv_internal,
    _solve_prognostic_linear_system,
)
from ppi_aipw import (
    MeanInferenceResult,
    aipw_mean_ci,
    aipw_mean_inference,
    aipw_mean_pointestimate,
    aipw_mean_se,
    calibrate_predictions,
    compute_two_sample_balancing_weights,
    fit_calibrator,
    isotonic_mean_pointestimate,
    linear_calibration_mean_pointestimate,
    mean_ci,
    mean_inference,
    mean_pointestimate,
    mean_se,
    ppi_aipw_mean_pointestimate,
    sigmoid_mean_pointestimate,
    select_mean_method_cv,
)


def _manual_efficiency_lambda(y: np.ndarray, yhat: np.ndarray, yhat_unlabeled: np.ndarray) -> float:
    n = len(y)
    N = len(yhat_unlabeled)
    c = N / float(n + N)
    labeled_outcome = y
    labeled_score = c * yhat
    unlabeled_score = c * yhat_unlabeled
    numerator = np.mean(
        (labeled_outcome - labeled_outcome.mean()) * (labeled_score - labeled_score.mean())
    )
    denominator = np.var(labeled_score) + (n / float(N)) * np.var(unlabeled_score)
    return float(numerator / denominator)


def test_aipw_matches_manual_augmented_estimator() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat = np.array([0.1, 0.4, 0.6, 0.8], dtype=float)
    yhat_unlabeled = np.array([0.2, 0.3, 0.7], dtype=float)

    estimate = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="aipw")
    rho = len(y) / (len(y) + len(yhat_unlabeled))
    expected = rho * yhat.mean() + (1.0 - rho) * yhat_unlabeled.mean() + np.mean(y - yhat)

    np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-12)


def test_weighted_aipw_matches_manual_augmented_estimator() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat = np.array([0.1, 0.4, 0.6, 0.8], dtype=float)
    yhat_unlabeled = np.array([0.2, 0.3, 0.7], dtype=float)
    w = np.array([1.0, 2.0, 1.5, 0.5], dtype=float)
    w_unlabeled = np.array([2.0, 1.0, 3.0], dtype=float)

    estimate = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        w=w,
        w_unlabeled=w_unlabeled,
    )

    w_norm = w / w.sum() * len(w)
    w_unlabeled_norm = w_unlabeled / w_unlabeled.sum() * len(w_unlabeled)
    rho = len(y) / (len(y) + len(yhat_unlabeled))
    c = 1.0 - rho
    expected = np.mean(w_norm * (y - c * yhat)) + np.mean(w_unlabeled_norm * (c * yhat_unlabeled))

    np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-12)


def test_compute_two_sample_balancing_weights_matches_pooled_target_moments() -> None:
    x_labeled = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    x_unlabeled = np.array([[1.0], [2.0]], dtype=float)

    weights, diagnostics = compute_two_sample_balancing_weights(
        x_labeled,
        x_unlabeled,
        target="pooled",
        return_diagnostics=True,
    )

    rho = x_labeled.shape[0] / float(x_labeled.shape[0] + x_unlabeled.shape[0])
    pooled_mean = rho * np.mean(x_labeled[:, 0]) + (1.0 - rho) * np.mean(x_unlabeled[:, 0])
    np.testing.assert_allclose(np.mean(weights), 1.0, atol=1e-8)
    np.testing.assert_allclose(np.mean(weights * x_labeled[:, 0]), pooled_mean, atol=1e-8)
    assert diagnostics["target"] == "pooled"
    assert diagnostics["max_abs_balance_error"] <= 1e-8
    assert np.all(weights >= 0.0)


def test_compute_two_sample_balancing_weights_matches_unlabeled_target_moments() -> None:
    x_labeled = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    x_unlabeled = np.array([[1.5], [2.5]], dtype=float)

    weights = compute_two_sample_balancing_weights(
        x_labeled,
        x_unlabeled,
        target="unlabeled",
    )

    np.testing.assert_allclose(np.mean(weights), 1.0, atol=1e-8)
    np.testing.assert_allclose(
        np.mean(weights * x_labeled[:, 0]),
        np.mean(x_unlabeled[:, 0]),
        atol=1e-8,
    )
    assert np.all(weights >= 0.0)


def test_compute_two_sample_balancing_weights_returns_ones_when_already_balanced() -> None:
    x_labeled = np.array([[0.0], [1.0], [2.0]], dtype=float)
    x_unlabeled = np.array([[0.0], [1.0], [2.0]], dtype=float)

    weights = compute_two_sample_balancing_weights(
        x_labeled,
        x_unlabeled,
        target="unlabeled",
    )

    np.testing.assert_allclose(weights, np.ones(x_labeled.shape[0]), atol=1e-8)


def test_compute_two_sample_balancing_weights_raises_when_nonnegative_balance_is_infeasible() -> None:
    x_labeled = np.array([[0.0], [1.0]], dtype=float)
    x_unlabeled = np.array([[10.0]], dtype=float)

    with pytest.raises(ValueError, match="Could not compute nonnegative balancing weights|Could not achieve"):
        compute_two_sample_balancing_weights(
            x_labeled,
            x_unlabeled,
            target="unlabeled",
        )


def test_two_sample_balancing_weights_can_be_supplied_to_mean_fit() -> None:
    y = np.array([0.2, 0.7, 1.0, 1.3], dtype=float)
    yhat = np.array([0.1, 0.8, 0.9, 1.1], dtype=float)
    yhat_unlabeled = np.array([0.3, 0.6, 1.0], dtype=float)
    x_labeled = np.array([[0.0], [0.5], [1.0], [1.5]], dtype=float)
    x_unlabeled = np.array([[0.4], [0.8], [1.2]], dtype=float)

    weights = compute_two_sample_balancing_weights(
        x_labeled,
        x_unlabeled,
        target="pooled",
    )
    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        w=weights,
    )

    assert np.isfinite(float(result.pointestimate))
    assert np.isfinite(float(result.se))
    assert np.all(weights >= 0.0)


def test_default_method_is_monotone_spline() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=30)
    yhat = y + rng.normal(scale=0.3, size=30)
    yhat_unlabeled = rng.normal(size=60)

    default_result = mean_inference(y, yhat, yhat_unlabeled)
    explicit_result = mean_inference(y, yhat, yhat_unlabeled, method="monotone_spline")
    default_model = fit_calibrator(y, yhat)

    np.testing.assert_allclose(default_result.pointestimate, explicit_result.pointestimate)
    np.testing.assert_allclose(default_result.se, explicit_result.se)
    np.testing.assert_allclose(default_result.ci[0], explicit_result.ci[0])
    np.testing.assert_allclose(default_result.ci[1], explicit_result.ci[1])
    assert default_result.method == "monotone_spline"
    assert default_model.method == "monotone_spline"


def test_efficiency_maximization_uses_unrestricted_if_variance_minimizer() -> None:
    yhat = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y = 2.0 * yhat
    yhat_unlabeled = np.array([1.5, 2.5, 3.5], dtype=float)

    estimate, calibrator = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        return_calibrator=True,
    )

    manual_lambda = _manual_efficiency_lambda(y, yhat, yhat_unlabeled)
    c = len(yhat_unlabeled) / float(len(y) + len(yhat_unlabeled))
    expected = np.mean(y - c * manual_lambda * yhat) + np.mean(c * manual_lambda * yhat_unlabeled)

    assert manual_lambda > 1.0
    np.testing.assert_allclose(calibrator.metadata["efficiency_lambda"], manual_lambda, rtol=0, atol=1e-12)
    np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-12)


def test_efficiency_maximization_typo_alias_is_removed_from_public_api() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=30)
    yhat = y + rng.normal(scale=0.6, size=30)
    yhat_unlabeled = rng.normal(size=50)

    with pytest.raises(TypeError, match="efficency_maximization"):
        aipw_mean_pointestimate(
            y,
            yhat,
            yhat_unlabeled,
            method="aipw",
            efficency_maximization=True,
        )


def test_selection_folds_alias_is_removed_from_public_api() -> None:
    rng = np.random.default_rng(12)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.25, size=40)
    yhat_unlabeled = rng.normal(size=80)

    with pytest.raises(TypeError, match="selection_folds"):
        select_mean_method_cv(
            y,
            yhat,
            yhat_unlabeled,
            candidate_methods=("aipw", "linear"),
            num_folds=7,
            selection_folds=5,
            selection_random_state=0,
        )


def test_cv_method_selection_prefers_linear_when_score_is_affinely_miscalibrated() -> None:
    rng = np.random.default_rng(21)
    yhat = np.linspace(-1.0, 1.0, 60)
    y = 0.75 + 2.5 * yhat + rng.normal(scale=0.05, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.2, 1.2, 120)

    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear"),
        num_folds=5,
        selection_random_state=0,
    )

    assert selected_method == "linear"
    assert diagnostics["scores"]["linear"] < diagnostics["scores"]["aipw"]

    auto_estimate, calibrator = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear"),
        num_folds=5,
        selection_random_state=0,
        return_calibrator=True,
    )
    linear_estimate = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="linear")

    np.testing.assert_allclose(auto_estimate, linear_estimate)
    assert calibrator.metadata["selected_method"] == "linear"


def test_prognostic_linear_matches_manual_semisupervised_linear_adjustment() -> None:
    yhat = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    y = 1.5 + 2.0 * yhat
    yhat_unlabeled = np.array([-0.8, -0.2, 0.3, 0.9], dtype=float)

    estimate, calibrator = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="prognostic_linear",
        return_calibrator=True,
    )

    design_labeled = np.column_stack([np.ones_like(yhat), yhat])
    coef, _, _, _ = np.linalg.lstsq(design_labeled, y, rcond=None)
    pred_labeled = design_labeled @ coef
    pred_unlabeled = np.column_stack([np.ones_like(yhat_unlabeled), yhat_unlabeled]) @ coef
    c = len(yhat_unlabeled) / float(len(y) + len(yhat_unlabeled))
    manual = np.mean(y - c * pred_labeled) + np.mean(c * pred_unlabeled)

    np.testing.assert_allclose(estimate, manual)
    assert calibrator.method == "prognostic_linear"
    assert calibrator.metadata["uses_covariates"] is False
    assert calibrator.metadata["ridge_alpha"] == pytest.approx(0.0)


def test_prognostic_linear_uses_covariates_and_auto_can_select_it() -> None:
    rng = np.random.default_rng(202)
    x = rng.normal(size=(60, 2))
    yhat = x[:, 0] + rng.normal(scale=0.3, size=60)
    y = 0.5 + 0.7 * yhat + 2.5 * x[:, 1] + rng.normal(scale=0.02, size=60)

    x_unlabeled = rng.normal(size=(120, 2))
    yhat_unlabeled = x_unlabeled[:, 0] + rng.normal(scale=0.3, size=120)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        X=x,
        X_unlabeled=x_unlabeled,
        method="prognostic_linear",
    )
    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        X=x,
        X_unlabeled=x_unlabeled,
        candidate_methods=("aipw", "linear", "prognostic_linear"),
        num_folds=5,
        selection_random_state=0,
    )

    assert np.isfinite(float(result.pointestimate))
    assert np.isfinite(float(result.se))
    assert result.calibrator.metadata["uses_covariates"] is True
    assert result.calibrator.metadata["x_dim"] == 2
    assert result.calibrator.metadata["ridge_penalizes_only_covariates"] is True
    assert "prognostic_linear" in diagnostics["scores"]
    assert selected_method == "prognostic_linear"


def test_uniform_weights_reproduce_unweighted_mean_inference() -> None:
    rng = np.random.default_rng(219)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.3, size=40)
    yhat_unlabeled = rng.normal(size=70)

    unweighted = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="wald",
    )
    weighted = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="wald",
        w=np.full(y.shape[0], 5.0),
        w_unlabeled=np.full(yhat_unlabeled.shape[0], 7.0),
    )

    np.testing.assert_allclose(weighted.pointestimate, unweighted.pointestimate)
    np.testing.assert_allclose(weighted.se, unweighted.se)
    np.testing.assert_allclose(weighted.ci[0], unweighted.ci[0])
    np.testing.assert_allclose(weighted.ci[1], unweighted.ci[1])


def test_weighted_linear_calibration_matches_row_duplication() -> None:
    y = np.array([0.2, 1.0, -0.3, 0.8], dtype=float)
    yhat = np.array([0.1, 0.7, -0.1, 0.6], dtype=float)
    yhat_unlabeled = np.array([0.0, 0.4, 0.9], dtype=float)
    w = np.array([1, 2, 3, 1], dtype=int)

    weighted_model = fit_calibrator(
        y,
        yhat,
        method="linear",
        w=w,
    )
    duplicated_model = fit_calibrator(
        np.repeat(y, w),
        np.repeat(yhat, w),
        method="linear",
    )

    np.testing.assert_allclose(weighted_model.predict(yhat), duplicated_model.predict(yhat))
    np.testing.assert_allclose(
        weighted_model.predict(yhat_unlabeled),
        duplicated_model.predict(yhat_unlabeled),
    )


def test_weighted_prognostic_linear_regression_matches_row_duplication() -> None:
    y = np.array([0.5, 1.2, -0.2, 0.9], dtype=float)
    yhat = np.array([0.4, 0.8, -0.1, 0.5], dtype=float)
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.5],
            [0.3, -0.2],
            [1.2, 0.7],
        ],
        dtype=float,
    )
    w = np.array([1, 3, 2, 1], dtype=int)
    normalized_weight = w / w.sum() * len(w)
    alpha = 1.5
    duplicated_alpha = alpha * (w.sum() / len(w))

    weighted_coef = _solve_prognostic_linear_system(
        yhat,
        x,
        y,
        normalized_weight,
        alpha=alpha,
    )
    duplicated_coef = _solve_prognostic_linear_system(
        np.repeat(yhat, w),
        np.repeat(x, w, axis=0),
        np.repeat(y, w),
        np.repeat(np.ones_like(y), w),
        alpha=duplicated_alpha,
    )

    np.testing.assert_allclose(
        _predict_prognostic_linear_from_coef(weighted_coef, yhat, x),
        _predict_prognostic_linear_from_coef(duplicated_coef, yhat, x),
    )


def test_auto_method_bootstrap_path_is_reproducible() -> None:
    rng = np.random.default_rng(22)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 1.0 + 1.8 * yhat + rng.normal(scale=0.07, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="bootstrap",
        n_resamples=30,
        random_state=5,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="bootstrap",
        n_resamples=30,
        random_state=5,
    )

    np.testing.assert_allclose(ci_one[0], ci_two[0])
    np.testing.assert_allclose(ci_one[1], ci_two[1])
    assert ci_one[0] <= ci_one[1]


def test_auto_method_jackknife_path_is_reproducible() -> None:
    rng = np.random.default_rng(122)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 1.0 + 1.8 * yhat + rng.normal(scale=0.07, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="jackknife",
        jackknife_folds=5,
        random_state=5,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="jackknife",
        jackknife_folds=5,
        random_state=5,
    )

    np.testing.assert_allclose(ci_one[0], ci_two[0])
    np.testing.assert_allclose(ci_one[1], ci_two[1])
    assert ci_one[0] <= ci_one[1]


def test_auto_bootstrap_reuses_selected_method_instead_of_rerunning_cv() -> None:
    rng = np.random.default_rng(24)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 0.9 + 2.0 * yhat + rng.normal(scale=0.06, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    auto_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )
    explicit_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )
    auto_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )
    explicit_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )

    np.testing.assert_allclose(auto_ci[0], explicit_ci[0])
    np.testing.assert_allclose(auto_ci[1], explicit_ci[1])
    np.testing.assert_allclose(auto_se, explicit_se)


def test_auto_jackknife_reuses_selected_method_instead_of_rerunning_cv() -> None:
    rng = np.random.default_rng(124)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 0.9 + 2.0 * yhat + rng.normal(scale=0.06, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    auto_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )
    explicit_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )
    auto_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )
    explicit_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )

    np.testing.assert_allclose(auto_ci[0], explicit_ci[0])
    np.testing.assert_allclose(auto_ci[1], explicit_ci[1])
    np.testing.assert_allclose(auto_se, explicit_se)


def test_auto_wald_uses_full_sample_pointestimate_and_cross_fitted_lambda_for_wald_se() -> None:
    rng = np.random.default_rng(23)
    yhat = np.linspace(-1.0, 1.0, 60)
    y = 0.5 + 2.2 * yhat + rng.normal(scale=0.08, size=yhat.shape[0])
    yhat_unlabeled = np.linspace(-1.2, 1.2, 120)

    auto_estimate = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )
    auto_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    selected_method, diagnostics, pred_labeled_cf, pred_unlabeled_cf = _select_mean_method_cv_internal(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )
    y_2d, pred_labeled_full, pred_unlabeled_full, weights, weights_unlabeled, _ = _prepare_mean_estimation_inputs(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        w=None,
        w_unlabeled=None,
        efficiency_maximization=False,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    if diagnostics["selected_efficiency_maximization"]:
        lambda_cf = _manual_efficiency_lambda(
            y,
            pred_labeled_cf.reshape(-1),
            pred_unlabeled_cf.reshape(-1),
        )
        pred_labeled_full = pred_labeled_full * lambda_cf
        pred_unlabeled_full = pred_unlabeled_full * lambda_cf
        pred_labeled_cf = pred_labeled_cf * lambda_cf
        pred_unlabeled_cf = pred_unlabeled_cf * lambda_cf

    c = len(yhat_unlabeled) / float(len(y) + len(yhat_unlabeled))
    manual_estimate = np.mean(y - c * pred_labeled_full.reshape(-1)) + np.mean(c * pred_unlabeled_full.reshape(-1))
    labeled_component = weights * (y_2d - c * pred_labeled_cf)
    unlabeled_component = weights_unlabeled * (c * pred_unlabeled_cf)
    manual_se = np.sqrt(
        np.var(labeled_component, axis=0) / y_2d.shape[0]
        + np.var(unlabeled_component, axis=0) / pred_unlabeled_cf.shape[0]
    ).reshape(-1)[0]

    np.testing.assert_allclose(auto_estimate, manual_estimate)
    np.testing.assert_allclose(auto_se, manual_se)


def test_auto_selection_adds_efficiency_maximized_aipw_candidate_and_uses_cross_fitted_lambda() -> None:
    yhat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    y = 2.0 * yhat
    yhat_unlabeled = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)

    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw",),
        efficiency_maximization=False,
        num_folds=3,
        selection_random_state=0,
    )
    selected_method_requested, diagnostics_requested = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw",),
        efficiency_maximization=True,
        num_folds=3,
        selection_random_state=0,
    )

    assert selected_method == "aipw"
    assert selected_method_requested == "aipw"
    assert diagnostics["selected_candidate"] == "aipw_efficiency_maximization"
    assert diagnostics["selected_efficiency_maximization"] is True
    assert diagnostics["scores"]["aipw_efficiency_maximization"] < diagnostics["scores"]["aipw"]
    np.testing.assert_allclose(
        diagnostics["scores"]["aipw"],
        diagnostics_requested["scores"]["aipw"],
    )
    np.testing.assert_allclose(
        diagnostics["scores"]["aipw_efficiency_maximization"],
        diagnostics_requested["scores"]["aipw_efficiency_maximization"],
    )

    auto_estimate, calibrator = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw",),
        num_folds=3,
        selection_random_state=0,
        return_calibrator=True,
    )
    _, _, pred_labeled_cf, pred_unlabeled_cf = _select_mean_method_cv_internal(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw",),
        num_folds=3,
        selection_random_state=0,
    )
    y_2d, pred_labeled_full, pred_unlabeled_full, weights, weights_unlabeled, _ = _prepare_mean_estimation_inputs(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        w=None,
        w_unlabeled=None,
        efficiency_maximization=False,
        candidate_methods=("aipw",),
        num_folds=3,
        selection_random_state=0,
    )

    lambda_cf = _manual_efficiency_lambda(
        y,
        pred_labeled_cf.reshape(-1),
        pred_unlabeled_cf.reshape(-1),
    )
    c = len(yhat_unlabeled) / float(len(y) + len(yhat_unlabeled))
    manual_estimate = np.mean(y_2d.reshape(-1) - c * lambda_cf * pred_labeled_full.reshape(-1)) + np.mean(
        c * lambda_cf * pred_unlabeled_full.reshape(-1)
    )

    np.testing.assert_allclose(auto_estimate, manual_estimate)
    assert calibrator.metadata["selected_efficiency_maximization"] is True
    assert calibrator.metadata["efficiency_lambda_source"] == "cross_fitted"


def test_auto_uses_unlabeled_subsample_for_selection_but_full_unlabeled_for_pointestimate() -> None:
    rng = np.random.default_rng(31)
    yhat = rng.normal(size=5)
    y = 0.7 + 1.4 * yhat + rng.normal(scale=0.05, size=5)
    yhat_unlabeled = rng.normal(size=80)

    _, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear"),
        num_folds=3,
        selection_random_state=0,
    )

    (
        _,
        _,
        pred_unlabeled_point,
        _,
        pred_unlabeled_variance,
        _,
        weights_unlabeled_point,
        weights_unlabeled_variance,
        calibrator,
    ) = _prepare_auto_variance_inputs(
        y,
        yhat,
        yhat_unlabeled,
        w=None,
        w_unlabeled=None,
        efficiency_maximization=False,
        candidate_methods=("aipw", "linear"),
        num_folds=3,
        auto_unlabeled_subsample_size=None,
        selection_random_state=0,
        isocal_backend="xgboost",
        isocal_max_depth=20,
        isocal_min_child_weight=10.0,
    )

    assert diagnostics["auto_unlabeled_subsample_size"] == 50
    assert diagnostics["unlabeled_strategy"] == "subsampled_unlabeled_rows_in_each_fold"
    assert pred_unlabeled_point.shape[0] == 80
    assert pred_unlabeled_variance.shape[0] == 50
    assert weights_unlabeled_point.shape[0] == 80
    assert weights_unlabeled_variance.shape[0] == 50
    assert calibrator.metadata["auto_unlabeled_subsample_size"] == 50


def test_mean_inference_linear_wald_matches_existing_wrappers() -> None:
    rng = np.random.default_rng(32)
    y = rng.normal(size=45)
    yhat = y + rng.normal(scale=0.35, size=45)
    yhat_unlabeled = rng.normal(size=90)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        alpha=0.1,
        inference="wald",
    )

    assert isinstance(result, MeanInferenceResult)
    np.testing.assert_allclose(result.pointestimate, mean_pointestimate(y, yhat, yhat_unlabeled, method="linear"))
    np.testing.assert_allclose(result.se, mean_se(y, yhat, yhat_unlabeled, method="linear"))
    lower, upper = mean_ci(y, yhat, yhat_unlabeled, method="linear", alpha=0.1)
    np.testing.assert_allclose(result.ci[0], lower)
    np.testing.assert_allclose(result.ci[1], upper)
    assert result.method == "linear"
    assert result.selected_candidate == "linear"
    assert result.selected_efficiency_maximization is False
    assert result.efficiency_lambda is None
    assert result.calibrator.method == "linear"


def test_mean_inference_auto_wald_matches_existing_separate_calls() -> None:
    rng = np.random.default_rng(33)
    yhat = np.linspace(-1.0, 1.0, 60)
    y = 0.75 + 2.1 * yhat + rng.normal(scale=0.08, size=60)
    yhat_unlabeled = np.linspace(-1.2, 1.2, 120)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="wald",
    )
    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    np.testing.assert_allclose(
        result.pointestimate,
        mean_pointestimate(
            y,
            yhat,
            yhat_unlabeled,
            method="auto",
            candidate_methods=("aipw", "linear", "isotonic"),
            num_folds=5,
            selection_random_state=0,
        ),
    )
    np.testing.assert_allclose(
        result.se,
        mean_se(
            y,
            yhat,
            yhat_unlabeled,
            method="auto",
            candidate_methods=("aipw", "linear", "isotonic"),
            num_folds=5,
            selection_random_state=0,
        ),
    )
    lower, upper = mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        alpha=0.1,
    )
    np.testing.assert_allclose(result.ci[0], lower)
    np.testing.assert_allclose(result.ci[1], upper)
    assert result.method == selected_method
    assert result.selected_candidate == diagnostics["selected_candidate"]
    assert result.selected_efficiency_maximization == diagnostics["selected_efficiency_maximization"]
    assert result.diagnostics["auto_unlabeled_subsample_size"] == diagnostics["auto_unlabeled_subsample_size"]
    if result.diagnostics["final_efficiency_maximization"]:
        assert result.efficiency_lambda is not None
        assert result.diagnostics["lambda_from_cross_fitted_estimates"] is True
    else:
        assert result.efficiency_lambda is None


def test_mean_inference_auto_bootstrap_matches_selected_method_semantics() -> None:
    rng = np.random.default_rng(34)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 0.9 + 2.0 * yhat + rng.normal(scale=0.06, size=50)
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    result = aipw_mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )
    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    explicit_pointestimate = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
    )
    explicit_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )
    explicit_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="bootstrap",
        n_resamples=30,
        random_state=7,
    )

    np.testing.assert_allclose(result.pointestimate, explicit_pointestimate)
    np.testing.assert_allclose(result.se, explicit_se)
    np.testing.assert_allclose(result.ci[0], explicit_ci[0])
    np.testing.assert_allclose(result.ci[1], explicit_ci[1])
    assert result.method == selected_method
    assert result.selected_candidate == diagnostics["selected_candidate"]
    assert result.selected_efficiency_maximization == diagnostics["selected_efficiency_maximization"]
    assert result.diagnostics["bootstrap_selected_once"] is True


def test_mean_inference_auto_jackknife_matches_selected_method_semantics() -> None:
    rng = np.random.default_rng(134)
    yhat = np.linspace(-1.0, 1.0, 50)
    y = 0.9 + 2.0 * yhat + rng.normal(scale=0.06, size=50)
    yhat_unlabeled = np.linspace(-1.1, 1.1, 100)

    result = aipw_mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="auto",
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )
    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=5,
        selection_random_state=0,
    )

    explicit_pointestimate = aipw_mean_pointestimate(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
    )
    explicit_se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )
    explicit_ci = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method=selected_method,
        efficiency_maximization=diagnostics["selected_efficiency_maximization"],
        inference="jackknife",
        jackknife_folds=5,
        random_state=7,
    )

    np.testing.assert_allclose(result.pointestimate, explicit_pointestimate)
    np.testing.assert_allclose(result.se, explicit_se)
    np.testing.assert_allclose(result.ci[0], explicit_ci[0])
    np.testing.assert_allclose(result.ci[1], explicit_ci[1])
    assert result.method == selected_method
    assert result.selected_candidate == diagnostics["selected_candidate"]
    assert result.selected_efficiency_maximization == diagnostics["selected_efficiency_maximization"]
    assert result.diagnostics["jackknife_selected_once"] is True
    assert result.diagnostics["jackknife_method"] == selected_method
    assert result.diagnostics["jackknife_efficiency_maximization"] == diagnostics["selected_efficiency_maximization"]
    assert result.diagnostics["jackknife_folds"] == 5


def test_mean_inference_reports_efficiency_lambda_when_applicable() -> None:
    rng = np.random.default_rng(35)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.5, size=40)
    yhat_unlabeled = rng.normal(size=80)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        efficiency_maximization=True,
    )

    assert result.selected_efficiency_maximization is True
    assert result.efficiency_lambda is not None
    assert result.diagnostics["efficiency_lambda_source"] == "full_sample"
    assert result.calibrator.metadata["efficiency_lambda_source"] == "full_sample"


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

    pred_labeled, pred_unlabeled = calibrate_predictions(y, yhat, yhat_unlabeled, method="sigmoid")

    assert np.all(pred_labeled >= y.min())
    assert np.all(pred_labeled <= y.max())
    assert np.all(pred_unlabeled >= y.min())
    assert np.all(pred_unlabeled <= y.max())
    assert np.all(np.diff(pred_unlabeled[np.argsort(yhat_unlabeled)]) >= -1e-12)


def test_isocal_predictions_are_bounded_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_unlabeled = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    pred_labeled, pred_unlabeled, model = calibrate_predictions(
        y,
        yhat,
        yhat_unlabeled,
        method="isotonic",
        return_model=True,
    )

    assert np.all(pred_labeled >= 0.0)
    assert np.all(pred_labeled <= 1.0)
    assert np.all(pred_unlabeled >= 0.0)
    assert np.all(pred_unlabeled <= 1.0)
    assert np.all(np.diff(pred_labeled[np.argsort(yhat)]) >= -1e-12)
    assert model.metadata["isocal_backend"] == "xgboost"
    assert model.calibrators[0].metadata["backend"] == "xgboost"
    assert model.calibrators[0].metadata["min_child_weight"] == 10.0


def test_isocal_sklearn_backend_is_available_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_unlabeled = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    pred_labeled, pred_unlabeled, model = calibrate_predictions(
        y,
        yhat,
        yhat_unlabeled,
        method="isotonic",
        isocal_backend="sklearn",
        return_model=True,
    )

    assert np.all(pred_labeled >= 0.0)
    assert np.all(pred_labeled <= 1.0)
    assert np.all(pred_unlabeled >= 0.0)
    assert np.all(pred_unlabeled <= 1.0)
    assert np.all(np.diff(pred_labeled[np.argsort(yhat)]) >= -1e-12)
    assert model.metadata["isocal_backend"] == "sklearn"
    assert model.calibrators[0].metadata["backend"] == "sklearn"


def test_monotone_spline_predictions_are_bounded_and_monotone() -> None:
    x = np.linspace(0.0, 1.0, 30)
    y = 0.2 + 0.6 * x**2
    yhat = x + 0.05 * np.sin(4.0 * np.pi * x)
    yhat_unlabeled = np.linspace(0.0, 1.0, 40)

    pred_labeled, pred_unlabeled, model = calibrate_predictions(
        y,
        yhat,
        yhat_unlabeled,
        method="monotone_spline",
        return_model=True,
    )

    order_l = np.argsort(yhat)
    assert np.all(pred_labeled >= y.min() - 1e-12)
    assert np.all(pred_labeled <= y.max() + 1e-12)
    assert np.all(pred_unlabeled >= y.min() - 1e-12)
    assert np.all(pred_unlabeled <= y.max() + 1e-12)
    assert np.all(np.diff(pred_labeled[order_l]) >= -1e-10)
    assert np.all(np.diff(pred_unlabeled) >= -1e-10)
    assert model.method == "monotone_spline"


def test_monotone_spline_mean_inference_runs_and_auto_accepts_it() -> None:
    rng = np.random.default_rng(37)
    x = np.linspace(0.0, 1.0, 80)
    y = 0.3 + 0.5 * x**2 + rng.normal(scale=0.01, size=80)
    yhat = x + rng.normal(scale=0.03, size=80)
    yhat_unlabeled = np.linspace(0.0, 1.0, 160)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="monotone_spline",
    )
    selected_method, diagnostics = select_mean_method_cv(
        y,
        yhat,
        yhat_unlabeled,
        candidate_methods=("aipw", "linear", "monotone_spline"),
        num_folds=5,
        selection_random_state=0,
    )

    assert np.isfinite(float(result.pointestimate))
    assert np.isfinite(float(result.se))
    assert result.calibrator.method == "monotone_spline"
    assert selected_method in {"aipw", "linear", "monotone_spline"}
    assert "monotone_spline" in diagnostics["scores"]


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
    isocal_generic = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="isotonic")
    isocal_wrapper = isotonic_mean_pointestimate(y, yhat, yhat_unlabeled)
    platt_generic = aipw_mean_pointestimate(y, yhat, yhat_unlabeled, method="sigmoid")
    platt_wrapper = sigmoid_mean_pointestimate(y, yhat, yhat_unlabeled)

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


def test_separate_calls_still_match_mean_inference_regression() -> None:
    rng = np.random.default_rng(36)
    y = rng.normal(size=40)
    yhat = y + rng.normal(scale=0.35, size=40)
    yhat_unlabeled = rng.normal(size=100)

    result = mean_inference(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=40,
        random_state=0,
    )

    np.testing.assert_allclose(mean_pointestimate(y, yhat, yhat_unlabeled, method="linear"), result.pointestimate)
    np.testing.assert_allclose(
        mean_se(
            y,
            yhat,
            yhat_unlabeled,
            method="linear",
            inference="bootstrap",
            n_resamples=40,
            random_state=0,
        ),
        result.se,
    )
    lower, upper = mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="bootstrap",
        n_resamples=40,
        random_state=0,
    )
    np.testing.assert_allclose(lower, result.ci[0])
    np.testing.assert_allclose(upper, result.ci[1])


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


def test_jackknife_ci_is_reproducible() -> None:
    rng = np.random.default_rng(103)
    y = rng.normal(size=30)
    yhat = y + rng.normal(scale=0.4, size=30)
    yhat_unlabeled = rng.normal(size=80)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="jackknife",
        jackknife_folds=10,
        random_state=0,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="linear",
        inference="jackknife",
        jackknife_folds=10,
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
        method="isotonic",
        inference="bootstrap",
        n_resamples=50,
        random_state=1,
    )
    se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="isotonic",
        inference="bootstrap",
        n_resamples=50,
        random_state=1,
    )

    assert lower <= upper
    assert se > 0.0


def test_jackknife_refits_isocal_and_returns_positive_se() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_unlabeled = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.7], dtype=float)

    lower, upper = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="isotonic",
        inference="jackknife",
        jackknife_folds=3,
        random_state=1,
    )
    se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="isotonic",
        inference="jackknife",
        jackknife_folds=3,
        random_state=1,
    )

    assert lower <= upper
    assert se > 0.0


def test_efficiency_maximization_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(size=35)
    yhat = y + rng.normal(scale=0.5, size=35)
    yhat_unlabeled = rng.normal(size=90)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        inference="bootstrap",
        n_resamples=50,
        random_state=3,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        inference="bootstrap",
        n_resamples=50,
        random_state=3,
    )
    se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
    )

    np.testing.assert_allclose(ci_one[0], ci_two[0])
    np.testing.assert_allclose(ci_one[1], ci_two[1])
    assert ci_one[0] <= ci_one[1]
    assert se > 0.0


def test_efficiency_maximization_jackknife_is_reproducible() -> None:
    rng = np.random.default_rng(107)
    y = rng.normal(size=35)
    yhat = y + rng.normal(scale=0.5, size=35)
    yhat_unlabeled = rng.normal(size=90)

    ci_one = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        inference="jackknife",
        jackknife_folds=10,
        random_state=3,
    )
    ci_two = aipw_mean_ci(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        inference="jackknife",
        jackknife_folds=10,
        random_state=3,
    )
    se = aipw_mean_se(
        y,
        yhat,
        yhat_unlabeled,
        method="aipw",
        efficiency_maximization=True,
        inference="jackknife",
        jackknife_folds=10,
        random_state=3,
    )

    np.testing.assert_allclose(ci_one[0], ci_two[0])
    np.testing.assert_allclose(ci_one[1], ci_two[1])
    assert ci_one[0] <= ci_one[1]
    assert se > 0.0


def test_invalid_inference_and_jackknife_validation_raise_clear_errors() -> None:
    rng = np.random.default_rng(111)
    y = rng.normal(size=6)
    yhat = y + rng.normal(scale=0.2, size=6)
    yhat_unlabeled = rng.normal(size=8)

    with pytest.raises(ValueError, match="'wald', 'jackknife', or 'bootstrap'"):
        aipw_mean_ci(y, yhat, yhat_unlabeled, inference="sandwich")

    with pytest.raises(ValueError, match="jackknife_folds must be at least 2"):
        aipw_mean_ci(
            y,
            yhat,
            yhat_unlabeled,
            inference="jackknife",
            jackknife_folds=1,
        )

    with pytest.raises(ValueError, match="Try fewer jackknife_folds or inference='wald'"):
        aipw_mean_ci(
            y[:2],
            yhat[:2],
            yhat_unlabeled[:2],
            inference="jackknife",
            jackknife_folds=2,
            random_state=0,
        )
