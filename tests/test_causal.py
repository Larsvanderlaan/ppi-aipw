from __future__ import annotations

import numpy as np
import pytest

from ppi_aipw import CausalInferenceResult, causal_inference, mean_inference


def _make_causal_data(
    *,
    seed: int,
    n: int,
    treatment_levels: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    arm_probs = np.full(len(treatment_levels), 1.0 / len(treatment_levels))
    arm_indices = rng.choice(len(treatment_levels), size=n, p=arm_probs)
    A = np.asarray([treatment_levels[idx] for idx in arm_indices], dtype=object)

    potential_means = []
    potential_outcomes = []
    for idx in range(len(treatment_levels)):
        mean = 0.5 * idx + 0.8 * x + 0.15 * idx * x
        outcome = mean + rng.normal(scale=0.25, size=n)
        potential_means.append(mean + rng.normal(scale=0.08, size=n))
        potential_outcomes.append(outcome)

    Y = np.zeros(n, dtype=float)
    for idx, arm in enumerate(treatment_levels):
        mask = A == arm
        Y[mask] = potential_outcomes[idx][mask]

    Yhat_potential = np.column_stack(potential_means)
    return Y, A, Yhat_potential


def _normalized_full_sample_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum() * weights.shape[0]


def _manual_weighted_arm_mean(
    Y: np.ndarray,
    A: np.ndarray,
    arm_predictions: np.ndarray,
    *,
    arm: object,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights_norm = _normalized_full_sample_weights(weights)
    mask = np.asarray(A == arm, dtype=float)
    arm_probability = float(np.mean(mask))
    pseudo_outcome = weights_norm * (
        arm_predictions + (mask / arm_probability) * (Y - arm_predictions)
    )
    influence = (pseudo_outcome - np.mean(pseudo_outcome)) / Y.shape[0]
    return float(np.mean(pseudo_outcome)), influence


def _oracle_ate_simulation_summary(
    *,
    weighted: bool,
    n_rep: int = 200,
    n: int = 3000,
    alpha: float = 0.1,
    seed: int = 20260413,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1000 * int(weighted))
    true_ate = 1.0 if not weighted else (1.0 + 0.3 * 0.45 - 0.15 * (-0.25))

    estimates = []
    covered = []
    for _ in range(n_rep):
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        A = rng.binomial(1, 0.5, size=n)
        mu0 = 0.5 + 0.4 * x1 - 0.2 * x2
        tau = 1.0 + 0.3 * x1 - 0.15 * x2
        mu1 = mu0 + tau
        y0 = mu0 + rng.normal(scale=1.0, size=n)
        y1 = mu1 + rng.normal(scale=1.0, size=n)
        Y = np.where(A == 1, y1, y0)
        Yhat_potential = np.column_stack([mu0, mu1])

        w = None
        if weighted:
            raw_w = np.exp(0.45 * x1 - 0.25 * x2)
            w = raw_w / raw_w.mean()

        result = causal_inference(
            Y,
            A,
            Yhat_potential,
            method="aipw",
            w=w,
            alpha=alpha,
        )
        estimates.append(float(result.ate[1]))
        lower, upper = result.ate_cis[1]
        covered.append(lower <= true_ate <= upper)

    estimate_arr = np.asarray(estimates)
    error = estimate_arr - true_ate
    return {
        "true_ate": float(true_ate),
        "mean_estimate": float(np.mean(estimate_arr)),
        "bias": float(np.mean(error)),
        "mse": float(np.mean(error**2)),
        "coverage": float(np.mean(covered)),
    }


def test_causal_binary_matches_direct_armwise_mean_inference() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=1, n=120, treatment_levels=(0, 1))

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="linear",
        alpha=0.1,
    )

    assert isinstance(result, CausalInferenceResult)
    assert result.control_arm == 0
    for arm_idx, arm in enumerate((0, 1)):
        mask = A == arm
        direct = mean_inference(
            Y[mask],
            Yhat_potential[mask, arm_idx],
            Yhat_potential[~mask, arm_idx],
            method="linear",
            alpha=0.1,
            inference="wald",
        )
        np.testing.assert_allclose(result.arm_means[arm], direct.pointestimate)
        np.testing.assert_allclose(result.arm_ses[arm], direct.se)
        np.testing.assert_allclose(result.arm_cis[arm][0], direct.ci[0])
        np.testing.assert_allclose(result.arm_cis[arm][1], direct.ci[1])

    np.testing.assert_allclose(result.ate[1], result.arm_means[1] - result.arm_means[0])
    assert np.isfinite(result.ate_ses[1])
    assert result.ate_cis[1][0] <= result.ate_cis[1][1]


def test_causal_multiarms_returns_all_noncontrol_vs_control_ates() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=2, n=150, treatment_levels=(0, 1, 2))

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="monotone_spline",
        alpha=0.1,
    )

    assert result.control_arm == 0
    assert result.treatment_levels == (0, 1, 2)
    assert set(result.ate) == {1, 2}
    for arm in (1, 2):
        np.testing.assert_allclose(result.ate[arm], result.arm_means[arm] - result.arm_means[0])
        assert np.isfinite(result.ate_ses[arm])
        assert result.ate_cis[arm][0] <= result.ate_cis[arm][1]


def test_causal_summary_reports_arm_means_ates_and_wald_statistics() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=44, n=160, treatment_levels=(0, 1, 2))

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="auto",
        candidate_methods=("aipw", "linear"),
        num_folds=4,
        selection_random_state=0,
    )

    text = result.summary()

    assert "CausalInferenceResult summary" in text
    assert "control_arm: 0" in text
    assert "Arm means:" in text
    assert "ATEs vs control:" in text
    assert "arm=0: estimate=" in text
    assert "1 - 0: estimate=" in text
    assert "wald_t=" in text
    assert "p_value=" in text
    assert "arm_wald_t_statistic" in result.diagnostics
    assert "ate_wald_p_value" in result.diagnostics


def test_causal_summary_null_override_changes_reported_wald_statistics() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=45, n=140, treatment_levels=(0, 1))

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="linear",
    )

    baseline = result.summary()
    shifted = result.summary(null=1.0, alternative="larger")

    assert "wald_null: 0" in baseline
    assert "wald_alternative: two-sided" in baseline
    assert "wald_null: 1" in shifted
    assert "wald_alternative: larger" in shifted
    assert baseline != shifted
    assert result.diagnostics["ate_wald_alternative"] == "larger"
    assert result.diagnostics["arm_wald_null"] == 1.0


def test_uniform_weights_reproduce_unweighted_causal_inference() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=22, n=120, treatment_levels=(0, 1, 2))

    unweighted = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="linear",
        alpha=0.1,
    )
    weighted = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="linear",
        alpha=0.1,
        w=np.full(Y.shape[0], 4.0),
    )

    for arm in unweighted.treatment_levels:
        np.testing.assert_allclose(weighted.arm_means[arm], unweighted.arm_means[arm])
        np.testing.assert_allclose(weighted.arm_ses[arm], unweighted.arm_ses[arm])
        np.testing.assert_allclose(weighted.arm_cis[arm], unweighted.arm_cis[arm])
    for arm in unweighted.ate:
        np.testing.assert_allclose(weighted.ate[arm], unweighted.ate[arm])
        np.testing.assert_allclose(weighted.ate_ses[arm], unweighted.ate_ses[arm])
        np.testing.assert_allclose(weighted.ate_cis[arm], unweighted.ate_cis[arm])


def test_causal_supports_categorical_treatments_with_explicit_levels_and_control() -> None:
    Y, A, Yhat_potential = _make_causal_data(
        seed=3,
        n=135,
        treatment_levels=("control", "low", "high"),
    )

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        treatment_levels=("control", "low", "high"),
        control_arm="control",
        method="linear",
    )

    assert result.control_arm == "control"
    assert result.treatment_levels == ("control", "low", "high")
    assert set(result.arm_means) == {"control", "low", "high"}
    assert set(result.ate) == {"low", "high"}


def test_causal_auto_runs_independently_by_arm() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=4, n=90, treatment_levels=(0, 1, 2))

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        method="auto",
        candidate_methods=("aipw", "linear", "monotone_spline", "isotonic"),
        num_folds=4,
        selection_random_state=0,
    )

    for arm in (0, 1, 2):
        assert result.arm_results[arm].method in {"aipw", "linear", "monotone_spline", "isotonic"}
        assert "selected_candidate" in result.arm_results[arm].diagnostics


def test_causal_prognostic_linear_passes_covariates_armwise() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=9, n=120, treatment_levels=(0, 1))
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 2))
    Y = Y + 0.8 * X[:, 0] - 0.4 * X[:, 1]
    Yhat_potential = Yhat_potential + np.column_stack([0.8 * X[:, 0], 0.8 * X[:, 0]])

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        X=X,
        method="prognostic_linear",
    )

    for arm_idx, arm in enumerate((0, 1)):
        mask = A == arm
        direct = mean_inference(
            Y[mask],
            Yhat_potential[mask, arm_idx],
            Yhat_potential[~mask, arm_idx],
            X=X[mask],
            X_unlabeled=X[~mask],
            method="prognostic_linear",
            inference="wald",
        )
        np.testing.assert_allclose(result.arm_means[arm], direct.pointestimate)
        np.testing.assert_allclose(result.arm_ses[arm], direct.se)
        assert result.arm_results[arm].calibrator.metadata["uses_covariates"] is True


def test_causal_weighted_aipw_matches_global_weighted_estimating_equation() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=29, n=120, treatment_levels=(0, 1))
    rng = np.random.default_rng(29)
    weights = rng.uniform(0.5, 2.0, size=Y.shape[0])

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        w=weights,
        method="aipw",
    )

    manual_arm_means = {}
    manual_arm_influences = {}
    for arm_idx, arm in enumerate((0, 1)):
        arm_mean, arm_influence = _manual_weighted_arm_mean(
            Y,
            A,
            Yhat_potential[:, arm_idx],
            arm=arm,
            weights=weights,
        )
        manual_arm_means[arm] = arm_mean
        manual_arm_influences[arm] = arm_influence
        np.testing.assert_allclose(result.arm_means[arm], arm_mean)
        np.testing.assert_allclose(result.arm_ses[arm], np.sqrt(np.sum(arm_influence**2)))

    ate_influence = manual_arm_influences[1] - manual_arm_influences[0]
    np.testing.assert_allclose(result.ate[1], manual_arm_means[1] - manual_arm_means[0])
    np.testing.assert_allclose(result.ate_ses[1], np.sqrt(np.sum(ate_influence**2)))


def test_causal_inference_is_invariant_to_weight_scaling() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=30, n=140, treatment_levels=(0, 1, 2))
    rng = np.random.default_rng(30)
    weights = rng.uniform(0.5, 2.0, size=Y.shape[0])

    baseline = causal_inference(
        Y,
        A,
        Yhat_potential,
        w=weights,
        method="linear",
    )
    rescaled = causal_inference(
        Y,
        A,
        Yhat_potential,
        w=11.0 * weights,
        method="linear",
    )

    for arm in baseline.treatment_levels:
        np.testing.assert_allclose(rescaled.arm_means[arm], baseline.arm_means[arm])
        np.testing.assert_allclose(rescaled.arm_ses[arm], baseline.arm_ses[arm])
        np.testing.assert_allclose(rescaled.arm_cis[arm], baseline.arm_cis[arm])
    for arm in baseline.ate:
        np.testing.assert_allclose(rescaled.ate[arm], baseline.ate[arm])
        np.testing.assert_allclose(rescaled.ate_ses[arm], baseline.ate_ses[arm])
        np.testing.assert_allclose(rescaled.ate_cis[arm], baseline.ate_cis[arm])


def test_oracle_unweighted_ate_simulation_has_small_bias_mse_and_near_nominal_coverage() -> None:
    summary = _oracle_ate_simulation_summary(weighted=False)

    assert summary["true_ate"] == 1.0
    assert abs(summary["bias"]) < 0.03
    assert summary["mse"] < 0.005
    assert 0.84 <= summary["coverage"] <= 0.96


def test_oracle_weighted_ate_simulation_has_small_bias_mse_and_near_nominal_coverage() -> None:
    summary = _oracle_ate_simulation_summary(weighted=True)

    assert summary["true_ate"] > 1.1
    assert abs(summary["bias"]) < 0.03
    assert summary["mse"] < 0.005
    assert 0.84 <= summary["coverage"] <= 0.98


def test_causal_invalid_inputs_raise_clear_errors() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=5, n=80, treatment_levels=(0, 1))

    with pytest.raises(ValueError, match="wald"):
        causal_inference(Y, A, Yhat_potential, inference="bootstrap")

    with pytest.raises(ValueError, match="control_arm"):
        causal_inference(Y, A, Yhat_potential, control_arm=99)

    with pytest.raises(ValueError, match="treatment_levels"):
        causal_inference(Y, A, Yhat_potential[:, :1], treatment_levels=(0, 1))

    with pytest.raises(ValueError, match="at least two observed treatment arms"):
        causal_inference(Y, np.zeros_like(A, dtype=int), Yhat_potential)
