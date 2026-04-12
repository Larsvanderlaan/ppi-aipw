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
        candidate_methods=("aipw", "linear", "isotonic"),
        num_folds=4,
        selection_random_state=0,
    )

    for arm in (0, 1, 2):
        assert result.arm_results[arm].method in {"aipw", "linear", "isotonic"}
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


def test_causal_weights_match_direct_armwise_weighted_mean_inference() -> None:
    Y, A, Yhat_potential = _make_causal_data(seed=29, n=120, treatment_levels=(0, 1))
    rng = np.random.default_rng(29)
    weights = rng.uniform(0.5, 2.0, size=Y.shape[0])

    result = causal_inference(
        Y,
        A,
        Yhat_potential,
        w=weights,
        method="linear",
    )

    for arm_idx, arm in enumerate((0, 1)):
        mask = A == arm
        direct = mean_inference(
            Y[mask],
            Yhat_potential[mask, arm_idx],
            Yhat_potential[~mask, arm_idx],
            method="linear",
            inference="wald",
            w=weights[mask],
            w_unlabeled=weights[~mask],
        )
        np.testing.assert_allclose(result.arm_means[arm], direct.pointestimate)
        np.testing.assert_allclose(result.arm_ses[arm], direct.se)
        np.testing.assert_allclose(result.arm_cis[arm], direct.ci)


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
