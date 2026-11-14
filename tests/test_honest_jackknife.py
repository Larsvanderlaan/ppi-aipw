"""Inference identities and training/evaluation independence."""
import itertools

import numpy as np
import pytest
from scipy.stats import t
from sklearn.model_selection import KFold

from ppi_aipw import mean_ci, mean_inference, mean_pointestimate, mean_se
from ppi_aipw import _api


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_vfold_jackknife_matches_paper_pseudovalues_and_t_reference(alternative):
    y = np.linspace(-2, 3, 40) ** 2
    scores = np.zeros(40)
    scores_u = np.zeros(80)
    result = mean_inference(y, scores, scores_u, method="aipw", inference="jackknife",
                            random_state=7, alternative=alternative)
    split = KFold(n_splits=10, shuffle=True, random_state=7)
    deleted = np.array([y[train].mean() for train, _ in split.split(y)])
    pseudo = 10 * y.mean() - 9 * deleted
    se = pseudo.std(ddof=1) / np.sqrt(10)
    np.testing.assert_allclose(result.se, se)
    np.testing.assert_allclose(result.pointestimate, y.mean())
    tail = 0.05 if alternative == "two-sided" else 0.1
    radius = t.ppf(1 - tail, df=9) * se
    expected = (y.mean() - radius, y.mean() + radius)
    if alternative == "larger": expected = (expected[0], np.inf)
    if alternative == "smaller": expected = (-np.inf, expected[1])
    np.testing.assert_allclose(result.ci, expected)
    result.summary(null=1.0, alternative=alternative)
    stat = (y.mean() - 1) / se
    p_value = 2 * t.sf(abs(stat), 9) if alternative == "two-sided" else (
        t.sf(stat, 9) if alternative == "larger" else t.cdf(stat, 9))
    np.testing.assert_allclose(result.diagnostics["wald_p_value"], p_value)
    assert result.diagnostics["jackknife_degrees_of_freedom"] == 9
    assert "Student t (df=9)" in result.summary()


def test_jackknife_caps_folds_and_keeps_coordinatewise_shapes():
    y = np.column_stack([np.arange(5.0), np.arange(5.0) ** 2])
    result = mean_inference(y, np.zeros_like(y), np.zeros((7, 2)),
                            method="aipw", inference="jackknife", random_state=0)
    assert result.diagnostics["jackknife_folds"] == 5
    assert result.diagnostics["jackknife_degrees_of_freedom"] == 4
    np.testing.assert_allclose(result.ci[1] - result.pointestimate,
                               t.ppf(0.95, 4) * result.se)


def test_auto_jackknife_repeats_selection_in_each_deleted_sample(monkeypatch):
    original = _api._select_mean_method_cv_internal
    sample_sizes = []
    def record(y, *args, **kwargs):
        sample_sizes.append(len(y))
        return original(y, *args, **kwargs)
    monkeypatch.setattr(_api, "_select_mean_method_cv_internal", record)
    y = np.arange(20.0)
    result = mean_inference(y, y + 1, np.arange(40.0), method="auto",
                            candidate_methods=("aipw", "linear"), num_folds=2,
                            selection_random_state=1, inference="jackknife", random_state=2)
    assert sample_sizes == [20] + [18] * 10
    assert result.diagnostics["jackknife_reselects_method"] is True
    assert result.diagnostics["jackknife_selected_once"] is False


@pytest.mark.parametrize("weighted", [False, True])
def test_honest_fold_score_differences_cancel_for_constant_predictions(weighted):
    # Unequal fold sizes in both samples must still use matching fold weights.
    y = np.arange(23.0) ** 2
    w = np.linspace(1, 3, len(y)) if weighted else None
    w_u = np.linspace(1, 4, 47) if weighted else None
    result = mean_inference(y, np.zeros(23), np.zeros(47), method="linear",
                            cross_fit=True, w=w, w_unlabeled=w_u)
    np.testing.assert_allclose(result.pointestimate, np.average(y, weights=w))
    assert result.diagnostics["calibration_folds"] == 10
    assert result.diagnostics["pointestimate_uses_full_sample_refit"] is False


def test_honest_auto_calibration_excludes_held_out_labels_and_unlabeled_covariates():
    y = np.arange(12.0) ** 2 + 0.3
    scores = np.linspace(0.1, 2, 12)
    scores_u = np.linspace(0.2, 3, 20)
    options = dict(method="auto", w=None, w_unlabeled=None, efficiency_maximization=True,
                   candidate_methods=("aipw", "linear"), num_folds=2,
                   cross_fit=True, calibration_folds=3, selection_random_state=4)
    fitted = _api._prepare_inference_inputs(y, scores, scores_u, **options)
    _, val = next(KFold(3, shuffle=True, random_state=4).split(y))
    _, val_u = next(KFold(3, shuffle=True, random_state=4).split(scores_u))
    changed_y = y.copy(); changed_y[val] += 1000
    changed_u = scores_u.copy(); changed_u[val_u] += 500
    changed = _api._prepare_inference_inputs(changed_y, scores, changed_u, **options)
    np.testing.assert_allclose(fitted.pred_labeled_point[val], changed.pred_labeled_point[val])
    assert fitted.diagnostics["calibration_fold_methods"][0] == changed.diagnostics["calibration_fold_methods"][0]
    np.testing.assert_allclose(fitted.diagnostics["calibration_fold_efficiency_lambda"][0],
                               changed.diagnostics["calibration_fold_efficiency_lambda"][0])
    np.testing.assert_array_equal(fitted.pred_labeled_point, fitted.pred_labeled_variance)
    np.testing.assert_array_equal(fitted.pred_unlabeled_point, fitted.pred_unlabeled_variance)


def test_honest_point_estimate_is_exactly_unbiased_in_finite_discrete_example():
    # Enumerate the joint distribution rather than relying on a Monte Carlo tolerance.
    estimates = []
    for values in itertools.product((0.0, 1.0, 2.0), repeat=6):
        x = np.asarray(values)
        estimates.append(mean_pointestimate(x[:4] ** 2, x[:4], x[4:],
                                             method="linear", cross_fit=True, calibration_folds=2))
    np.testing.assert_allclose(np.mean(estimates), 5 / 3, atol=1e-12)


@pytest.mark.parametrize("inference", ["wald", "jackknife", "bootstrap"])
def test_honest_estimator_is_shared_by_point_se_ci_and_result(inference):
    x = np.linspace(-1, 1, 23); y = x + x ** 2; u = np.linspace(-2, 2, 47)
    common = dict(method="linear", cross_fit=True, calibration_folds=3)
    options = dict(**common, inference=inference, random_state=7, n_resamples=4)
    result = mean_inference(y, x, u, **options)
    np.testing.assert_allclose(result.pointestimate, mean_pointestimate(y, x, u, **common))
    np.testing.assert_allclose(result.se, mean_se(y, x, u, **options))
    np.testing.assert_allclose(result.ci, mean_ci(y, x, u, **options))


def test_leave_one_out_calibration_and_default_false():
    x = np.arange(6.0); y = x ** 2; u = np.arange(12.0)
    options = dict(method="linear", w=None, w_unlabeled=None, efficiency_maximization=False)
    fitted = _api._prepare_inference_inputs(y, x, u, cross_fit=True, calibration_folds=len(y), **options)
    split = KFold(len(y), shuffle=True, random_state=0)
    for train, val in split.split(y):
        model = _api._fit_and_calibrate(y[train], x[train], u, method="linear", w=None,
                                       X=None, X_unlabeled=None, isocal_backend="xgboost",
                                       isocal_max_depth=20, isocal_min_child_weight=10)[-1]
        np.testing.assert_allclose(fitted.pred_labeled_point[val].reshape(-1), model.predict(x[val]))
    default = mean_inference(y, x, u, method="linear")
    explicit = mean_inference(y, x, u, method="linear", cross_fit=False)
    np.testing.assert_allclose(default.pointestimate, explicit.pointestimate)
    np.testing.assert_allclose(default.se, explicit.se)
    assert default.diagnostics["cross_fit"] is False


@pytest.mark.parametrize("name,value", [("calibration_folds", 1), ("calibration_folds", 2.5),
                                         ("calibration_folds", True), ("cross_fit", "yes"),
                                         ("jackknife_folds", 2.5)])
def test_invalid_folding_options(name, value):
    options = dict(cross_fit=True, inference="jackknife")
    options[name] = value
    with pytest.raises(ValueError, match=name):
        mean_inference(np.arange(6.0), np.arange(6.0), np.arange(12.0), **options)


@pytest.mark.parametrize("method", ["linear", "prognostic_linear", "sigmoid", "monotone_spline", "isotonic"])
def test_honest_calibration_handles_covariates_weights_and_multiple_outputs(method):
    rng = np.random.default_rng(135)
    x = rng.normal(size=(24, 2)); x_u = rng.normal(size=(48, 2))
    scores = 1 / (1 + np.exp(-x)); scores_u = 1 / (1 + np.exp(-x_u))
    y = scores ** 2 + rng.normal(scale=.1, size=scores.shape)
    result = mean_inference(y, scores, scores_u, method=method, cross_fit=True,
                            calibration_folds=3, X=x, X_unlabeled=x_u,
                            w=np.linspace(1, 2, 24), w_unlabeled=np.linspace(1, 3, 48))
    assert np.asarray(result.pointestimate).shape == (2,)
    assert np.all(np.isfinite(result.pointestimate))
    assert np.all(np.asarray(result.se) > 0)
    assert result.diagnostics["calibration_folds"] == 3


@pytest.mark.parametrize("weighted", [False, True])
def test_causal_wald_passes_honest_calibration_to_each_arm(weighted):
    from ppi_aipw import causal_inference
    rng = np.random.default_rng(17)
    a = np.repeat([0, 1], 20)
    x = rng.normal(size=40)
    predictions = np.column_stack([x, x + 1])
    y = x + a + rng.normal(scale=.2, size=40)
    result = causal_inference(y, a, predictions, method="linear", cross_fit=True,
                              calibration_folds=3, w=np.linspace(1, 2, 40) if weighted else None)
    for arm in [0, 1]:
        assert result.arm_results[arm].diagnostics["cross_fit"] is True
    np.testing.assert_allclose(result.ate[1], result.arm_means[1] - result.arm_means[0])
    assert np.isfinite(result.ate_ses[1])
