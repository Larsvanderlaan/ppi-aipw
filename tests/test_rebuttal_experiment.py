from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments import rebuttal_sample_size as rebuttal_experiment
from experiments.estimators import (
    fit_linear_calibration,
    fit_sklearn_isotonic_calibration,
    predict_linear,
    predict_sklearn_isotonic,
)
from experiments.rebuttal_sample_size import (
    REBUTTAL_METHODS,
    RebuttalExperimentConfig,
    deterministic_three_way_folds,
    expected_raw_rows,
    loo_aipw_from_predictions,
    loo_isotonic_aipw,
    loo_linear_aipw,
    reppi_mlp_cv_score_only,
    reppi_rf_score_only,
    run_rebuttal_experiment,
    two_sample_crossfit_se,
    validate_rebuttal_raw,
)


def _manual_loo_predictions(
    y: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    heldout = np.empty(len(y), dtype=float)
    averaged_unlabeled = np.zeros(len(score_u), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        if method == "linear":
            model = fit_linear_calibration(score_l[keep], y[keep])
            predict = predict_linear
        elif method == "isotonic":
            model = fit_sklearn_isotonic_calibration(score_l[keep], y[keep])
            predict = predict_sklearn_isotonic
        else:
            raise AssertionError(method)
        heldout[index] = predict(model, score_l[index : index + 1])[0]
        averaged_unlabeled += predict(model, score_u) / len(y)
    return heldout, averaged_unlabeled


def test_linear_and_isotonic_loo_match_manual_refits() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    score_l = np.array([0.08, 0.22, 0.31, 0.45, 0.57, 0.74, 0.91])
    score_u = np.array([0.03, 0.19, 0.38, 0.68, 0.83, 0.97])

    for method, implementation in (
        ("linear", loo_linear_aipw),
        ("isotonic", loo_isotonic_aipw),
    ):
        expected_l, expected_u = _manual_loo_predictions(
            y,
            score_l,
            score_u,
            method=method,
        )
        result = implementation(y, score_l, score_u)
        assert result.heldout_predictions == pytest.approx(expected_l)
        assert result.averaged_unlabeled_predictions == pytest.approx(expected_u)
        expected = loo_aipw_from_predictions(y, expected_l, expected_u)
        assert result.estimate == pytest.approx(expected.estimate)
        assert result.se == pytest.approx(expected.se)


def test_loo_heldout_prediction_does_not_use_heldout_outcome() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    changed = y.copy()
    changed[2] = 7.0
    score_l = np.linspace(0.1, 0.9, len(y))
    score_u = np.linspace(0.05, 0.95, 15)

    for implementation in (loo_linear_aipw, loo_isotonic_aipw):
        original = implementation(y, score_l, score_u)
        perturbed = implementation(changed, score_l, score_u)
        assert perturbed.heldout_predictions[2] == pytest.approx(
            original.heldout_predictions[2]
        )


def test_loo_degenerate_calibration_folds_have_finite_labeled_mean_point() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    constant_l = np.full(len(y), 0.4)
    constant_u = np.full(20, 0.4)

    linear = loo_linear_aipw(y, constant_l, constant_u)
    assert linear.singular_fallbacks == len(y)
    assert linear.estimate == pytest.approx(np.mean(y))
    assert np.isfinite(linear.se)

    isotonic = loo_isotonic_aipw(
        np.ones_like(y),
        np.linspace(0.1, 0.9, len(y)),
        np.linspace(0.05, 0.95, len(constant_u)),
    )
    assert isotonic.singular_fallbacks == len(y)
    assert isotonic.estimate == pytest.approx(1.0)
    assert np.isfinite(isotonic.se)


def test_loo_point_estimate_matches_stated_exact_algebra() -> None:
    y = np.array([0.2, 0.7, 0.4, 0.9])
    heldout = np.array([0.1, 0.8, 0.5, 0.6])
    averaged_u = np.array([0.3, 0.4, 0.8, 0.2, 0.7])
    result = loo_aipw_from_predictions(y, heldout, averaged_u)
    c = len(averaged_u) / (len(y) + len(averaged_u))
    manual = np.mean(
        [
            y[index] + c * (np.mean(averaged_u) - heldout[index])
            for index in range(len(y))
        ]
    )
    assert result.estimate == pytest.approx(manual)


def test_two_sample_crossfit_se_matches_manual_sample_variances() -> None:
    y = np.array([0.0, 1.0, 0.25, 0.8, 0.4])
    heldout = np.array([0.2, 0.7, 0.3, 0.75, 0.45])
    averaged_u = np.array([0.1, 0.3, 0.6, 0.9, 0.5, 0.2, 0.8])
    c = len(averaged_u) / (len(y) + len(averaged_u))
    expected = np.sqrt(
        np.var(y - c * heldout, ddof=1) / len(y)
        + np.var(c * averaged_u, ddof=1) / len(averaged_u)
    )
    assert two_sample_crossfit_se(y, heldout, averaged_u) == pytest.approx(expected)


def test_reppi_rf_folds_seeds_and_outputs_are_deterministic() -> None:
    rng = np.random.default_rng(9)
    score_l = np.linspace(0.02, 0.98, 30)
    y = rng.binomial(1, score_l).astype(float)
    score_u = np.linspace(0.01, 0.99, 80)
    first = reppi_rf_score_only(y, score_l, score_u, seed=412, n_estimators=10)
    second = reppi_rf_score_only(y, score_l, score_u, seed=412, n_estimators=10)
    folds = deterministic_three_way_folds(len(y), 412)

    assert first.estimate == pytest.approx(second.estimate)
    assert first.se == pytest.approx(second.se)
    assert first.coefficients == pytest.approx(second.coefficients)
    assert first.evaluation_folds == tuple(
        tuple(int(index) for index in fold)
        for fold in (folds[2], folds[0], folds[1])
    )
    assert first.model_seeds == (105141, 209870, 314599)
    assert sorted(index for fold in first.evaluation_folds for index in fold) == list(
        range(len(y))
    )
    assert first.estimate == pytest.approx(
        np.mean(first.labeled_components) + np.mean(first.unlabeled_components)
    )


def test_reppi_rf_singular_prediction_variance_falls_back_to_labeled_mean() -> None:
    y = np.array([0.0, 1.0] * 9)
    score_l = np.full(len(y), 0.5)
    score_u = np.full(50, 0.5)
    result = reppi_rf_score_only(y, score_l, score_u, seed=18, n_estimators=10)
    assert result.singular_fallbacks == 3
    assert result.coefficients == pytest.approx((0.0, 0.0, 0.0))
    assert result.estimate == pytest.approx(np.mean(y))
    assert result.unlabeled_components == pytest.approx(np.zeros(len(score_u)))
    assert np.isfinite(result.se)


def test_reppi_mlp_cv_is_deterministic_and_uses_paper_v2_fold_aggregation() -> None:
    rng = np.random.default_rng(19)
    score_l = np.linspace(0.02, 0.98, 24)
    y = rng.binomial(1, score_l).astype(float)
    score_u = np.linspace(0.01, 0.99, 64)
    candidates = (((8,), 0.01), ((8,), 0.0001))
    first = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=517,
        candidates=candidates,
    )
    second = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=517,
        candidates=candidates,
    )
    folds = deterministic_three_way_folds(len(y), 517)

    assert first.estimate == pytest.approx(second.estimate)
    assert first.se == pytest.approx(second.se)
    assert first.coefficients == pytest.approx(second.coefficients)
    assert first.selected_candidates == second.selected_candidates
    assert np.asarray(first.candidate_cv_mses) == pytest.approx(
        np.asarray(second.candidate_cv_mses)
    )
    assert first.evaluation_folds == tuple(
        tuple(int(index) for index in fold)
        for fold in (folds[2], folds[0], folds[1])
    )
    assert sorted(index for fold in first.evaluation_folds for index in fold) == list(
        range(len(y))
    )
    assert first.estimate == pytest.approx(
        np.mean(first.labeled_components) + np.mean(first.unlabeled_components)
    )


def test_reppi_mlp_cv_selection_never_uses_outer_evaluation_outcomes() -> None:
    rng = np.random.default_rng(29)
    score_l = np.linspace(0.03, 0.97, 30)
    y = rng.binomial(1, score_l).astype(float)
    score_u = np.linspace(0.01, 0.99, 80)
    seed = 713
    candidates = (((8,), 0.01), ((8,), 0.0001))
    folds = deterministic_three_way_folds(len(y), seed)
    outer_evaluation = folds[2]
    changed = y.copy()
    changed[outer_evaluation] = 1.0 - changed[outer_evaluation]

    original = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=seed,
        candidates=candidates,
    )
    perturbed = reppi_mlp_cv_score_only(
        changed,
        score_l,
        score_u,
        seed=seed,
        candidates=candidates,
    )

    assert perturbed.selected_candidates[0] == original.selected_candidates[0]
    assert perturbed.candidate_cv_mses[0] == pytest.approx(
        original.candidate_cv_mses[0]
    )
    assert perturbed.evaluation_predictions[outer_evaluation] == pytest.approx(
        original.evaluation_predictions[outer_evaluation]
    )


def test_reppi_mlp_cv_ties_select_first_frozen_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConstantModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(len(features), 0.4)

    def fit_constant(*args: object, **kwargs: object) -> tuple[ConstantModel, int, int]:
        return ConstantModel(), 0, 0

    monkeypatch.setattr(rebuttal_experiment, "_fit_reppi_mlp", fit_constant)
    y = np.array([0.0, 1.0] * 9)
    score_l = np.linspace(0.05, 0.95, len(y))
    score_u = np.linspace(0.02, 0.98, 50)
    candidates = (((8,), 0.01), ((32, 32), 0.0001))
    result = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=81,
        candidates=candidates,
    )
    assert result.selected_candidates == ("h8_a1e-02",) * 3


def test_reppi_mlp_cv_frozen_grid_matches_disclosed_config() -> None:
    config_path = (
        Path(rebuttal_experiment.__file__).resolve().parent
        / "reppi_baseline_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = (
        ((8,), 0.01),
        ((8,), 0.0001),
        ((32, 32), 0.01),
        ((32, 32), 0.0001),
        ((32, 32, 32), 0.01),
        ((32, 32, 32), 0.0001),
    )

    assert rebuttal_experiment.REPPI_MLP_CANDIDATES == expected
    assert rebuttal_experiment.REPPI_MLP_INNER_FOLDS == 2
    assert rebuttal_experiment.REPPI_MLP_MAX_ITER == 1000
    assert config["mlp_cv"]["evaluation_fold_used_for_selection"] is False
    assert config["mlp_cv"]["selection_metric"] == "mean held-out outcome MSE"
    assert [list(candidate[0]) for candidate in expected] == [
        layers
        for layers in config["mlp_cv"]["hidden_layer_sizes"]
        for _ in config["mlp_cv"]["alpha"]
    ]
    assert [candidate[1] for candidate in expected] == (
        config["mlp_cv"]["alpha"] * len(config["mlp_cv"]["hidden_layer_sizes"])
    )
    names = [
        rebuttal_experiment._mlp_candidate_name(candidate)
        for candidate in expected
    ]
    assert len(names) == len(set(names))
    assert "h32x32x32_a1e-04" in names


def test_reppi_mlp_cv_propagates_fit_nonfinite_and_warning_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScoreModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.asarray(features[:, 1], dtype=float)

    class NonfiniteModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(len(features), np.nan)

    def controlled_fit(
        score: np.ndarray,
        outcome: np.ndarray,
        *,
        candidate: tuple[tuple[int, ...], float],
        seed: int,
    ) -> tuple[object | None, int, int]:
        del score, outcome, seed
        layers, _ = candidate
        if layers == (16,):
            return None, 0, 1
        if layers == (32,):
            return NonfiniteModel(), 0, 0
        return ScoreModel(), 1, 0

    monkeypatch.setattr(rebuttal_experiment, "_fit_reppi_mlp", controlled_fit)
    score_l = np.linspace(0.02, 0.98, 30)
    y = (score_l > 0.5).astype(float)
    score_u = np.linspace(0.01, 0.99, 80)
    result = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=118,
        candidates=(
            ((8,), 0.01),
            ((16,), 0.01),
            ((32,), 0.01),
        ),
    )

    # Per outer rotation: two warning-bearing inner fits plus one final fit;
    # one failed candidate fit; and one nonfinite validation prediction.
    assert result.convergence_warnings == 9
    assert result.model_fit_failures == 3
    assert result.nonfinite_prediction_failures == 3
    assert result.selected_candidates == ("h8_a1e-02",) * 3
    assert np.all(np.isfinite(result.evaluation_predictions))
    assert np.isfinite(result.estimate)
    assert np.isfinite(result.se)


def test_reppi_mlp_cv_constant_score_falls_back_to_labeled_mean() -> None:
    y = np.array([0.0, 1.0] * 9)
    score_l = np.full(len(y), 0.5)
    score_u = np.full(50, 0.5)
    result = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=18,
        candidates=(((8,), 0.01),),
    )
    assert result.singular_fallbacks == 3
    assert result.coefficients == pytest.approx((0.0, 0.0, 0.0))
    assert result.estimate == pytest.approx(np.mean(y))
    assert result.unlabeled_components == pytest.approx(np.zeros(len(score_u)))
    assert np.isfinite(result.se)


def test_reppi_mlp_cv_retains_only_linear_sized_prediction_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScoreModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.asarray(features[:, 1], dtype=float)

    def select_score_model(
        y: np.ndarray,
        score: np.ndarray,
        train_idx: np.ndarray,
        *,
        seed: int,
        rotation: int,
        candidates: object,
    ) -> tuple[ScoreModel, str, tuple[float], int, int, int, int, int]:
        del y, score, train_idx, candidates
        return (
            ScoreModel(),
            "h8_a1e-02",
            (0.1,),
            seed + rotation,
            seed + rotation + 10,
            0,
            0,
            0,
        )

    monkeypatch.setattr(
        rebuttal_experiment,
        "_select_reppi_mlp",
        select_score_model,
    )
    n = 30
    N = 30_000
    score_l = np.linspace(0.02, 0.98, n)
    y = (score_l > 0.5).astype(float)
    score_u = np.linspace(0.01, 0.99, N)
    result = reppi_mlp_cv_score_only(
        y,
        score_l,
        score_u,
        seed=221,
        candidates=(((8,), 0.01),),
    )

    retained_arrays = {
        name: value
        for name, value in vars(result).items()
        if isinstance(value, np.ndarray)
    }
    assert {name: value.shape for name, value in retained_arrays.items()} == {
        "labeled_components": (n,),
        "unlabeled_components": (N,),
        "evaluation_predictions": (n,),
    }
    assert sum(value.size for value in retained_arrays.values()) == N + 2 * n
    assert not hasattr(result, "fold_by_unlabeled_predictions")


def test_isotonic_loo_retains_only_vector_sized_prediction_outputs() -> None:
    rng = np.random.default_rng(47)
    n = 25
    N = 400
    score_l = np.sort(rng.uniform(size=n))
    y = rng.binomial(1, score_l).astype(float)
    score_u = rng.uniform(size=N)
    result = loo_isotonic_aipw(y, score_l, score_u)
    assert result.heldout_predictions.shape == (n,)
    assert result.averaged_unlabeled_predictions.shape == (N,)
    assert not hasattr(result, "fold_by_unlabeled_predictions")


def test_validation_rejects_duplicate_keys() -> None:
    config = RebuttalExperimentConfig(
        output_dir=Path("/tmp/not-used"),
        recommendations_file=Path("experiments/paper_simulation_recommendations.json"),
        replications=2,
        n_grid=(20,),
        score_qualities=("well_calibrated",),
        methods=("labeled_only",),
        make_figures=False,
    )
    row = {
        "score_quality": "well_calibrated",
        "n_labeled": 20,
        "replicate": 0,
        "estimator": "labeled_only",
        "estimate": 0.5,
        "se": 0.1,
        "bias": 0.0,
        "sq_error": 0.0,
        "covered": 1.0,
        "ci_length": 0.4,
        "ci_lower": 0.3,
        "ci_upper": 0.7,
        "psi0": 0.5,
        "fit_status": "ok",
    }
    raw = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_rebuttal_raw(raw, config)


def test_rebuttal_cli_writes_expected_rows_figures_manifest_and_number_ledger(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "rebuttal"
    recommendations = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "paper_simulation_recommendations.json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "experiments" / "simulate.py"),
        "--rebuttal-sample-size",
        "--replications",
        "3",
        "--n-grid",
        "20",
        "--score-qualities",
        "well_calibrated",
        "--recommendations-file",
        str(recommendations),
        "--output-dir",
        str(output_dir),
        "--jobs",
        "1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    raw = pd.read_csv(output_dir / "raw_results.csv")
    assert len(raw) == 3 * len(REBUTTAL_METHODS)
    assert raw[["score_quality", "n_labeled", "replicate", "estimator"]].duplicated().sum() == 0
    summary = pd.read_csv(output_dir / "summary.csv")
    assert {
        "score_regime",
        "n",
        "method",
        "rmse",
        "mse",
        "bias",
        "coverage",
        "mean_interval_width",
        "relative_efficiency_vs_labeled_mean",
        "relative_efficiency_vs_ppi_plus_plus",
    }.issubset(summary.columns)
    assert set(summary["method"]) == set(REBUTTAL_METHODS)
    number_ledger = pd.read_csv(output_dir / "number_ledger.csv")
    assert {
        "relative_efficiency_vs_labeled_mean",
        "relative_efficiency_vs_ppi_plus_plus",
    }.issubset(set(number_ledger["metric"]))
    assert (output_dir / "fig_small_n_rmse.pdf").stat().st_size > 0
    assert (output_dir / "fig_small_n_reppi.pdf").stat().st_size > 0
    assert (output_dir / "fig_small_n_diagnostics.pdf").stat().st_size > 0

    with (output_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["expected_raw_rows"] == 3 * len(REBUTTAL_METHODS)
    assert manifest["observed_raw_rows"] == manifest["expected_raw_rows"]
    assert manifest["failed_fits"] == 0
    telemetry_columns = (
        "model_fit_failures",
        "nonfinite_prediction_failures",
        "convergence_warnings",
    )
    for column in telemetry_columns:
        expected_telemetry = {
            str(method): int(value)
            for method, value in raw.groupby("estimator")[column].sum().items()
        }
        assert manifest[column] == expected_telemetry
    selected_candidate_counts: dict[str, int] = {}
    for encoded in raw.loc[
        raw["estimator"] == "reppi_mlp_cv",
        "selected_candidates",
    ].astype(str):
        for candidate in encoded.split("|"):
            if candidate:
                selected_candidate_counts[candidate] = (
                    selected_candidate_counts.get(candidate, 0) + 1
                )
    assert manifest["reppi_mlp_selected_candidate_counts"] == (
        selected_candidate_counts
    )
    assert sum(selected_candidate_counts.values()) == 3 * 3
    allowed_candidates = {
        rebuttal_experiment._mlp_candidate_name(candidate)
        for candidate in rebuttal_experiment.REPPI_MLP_CANDIDATES
    } | {"fallback_labeled_mean"}
    assert set(selected_candidate_counts).issubset(allowed_candidates)
    baseline_config = json.loads(
        (
            Path(rebuttal_experiment.__file__).resolve().parent
            / "reppi_baseline_config.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["reppi_baseline_provenance"] == baseline_config
    assert manifest["confirmatory_defaults"]["reppi_mlp_candidates"] == [
        {
            "hidden_layer_sizes": list(layers),
            "alpha": alpha,
            "name": rebuttal_experiment._mlp_candidate_name((layers, alpha)),
        }
        for layers, alpha in rebuttal_experiment.REPPI_MLP_CANDIDATES
    ]
    experiment_dir = Path(rebuttal_experiment.__file__).resolve().parent
    implementation_paths = {
        "rebuttal_sample_size.py": Path(rebuttal_experiment.__file__).resolve(),
        "reppi_baseline_config.json": (
            Path(rebuttal_experiment.__file__).resolve().parent
            / "reppi_baseline_config.json"
        ),
        "simulate.py": experiment_dir / "simulate.py",
        "estimators.py": experiment_dir / "estimators.py",
    }
    assert set(manifest["implementation_hashes"]) == set(implementation_paths)
    for name, path in implementation_paths.items():
        assert manifest["implementation_hashes"][name] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert set(manifest["mc_uncertainty_columns"]) == {
        "mean_bias_mcse",
        "rmse_mcse",
        "coverage_mcse",
        "mean_ci_length_mcse",
        "relative_efficiency_vs_labeled_mean_mcse",
        "relative_efficiency_vs_ppi_plus_plus_mcse",
    }


def test_checkpoint_signature_changes_with_implementation_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RebuttalExperimentConfig(
        output_dir=tmp_path,
        recommendations_file=Path("experiments/paper_simulation_recommendations.json"),
        replications=2,
        n_grid=(20,),
        score_qualities=("well_calibrated",),
        methods=("labeled_only",),
        make_figures=False,
    )
    monkeypatch.setattr(
        rebuttal_experiment,
        "_implementation_hashes",
        lambda: {"rebuttal_sample_size.py": "a", "simulate.py": "b", "estimators.py": "c"},
    )
    first = rebuttal_experiment._checkpoint_path(
        tmp_path,
        config,
        "well_calibrated",
        20,
    )
    monkeypatch.setattr(
        rebuttal_experiment,
        "_implementation_hashes",
        lambda: {"rebuttal_sample_size.py": "d", "simulate.py": "b", "estimators.py": "c"},
    )
    second = rebuttal_experiment._checkpoint_path(
        tmp_path,
        config,
        "well_calibrated",
        20,
    )
    assert first != second


def test_expected_confirmatory_row_count_is_154000(tmp_path: Path) -> None:
    config = RebuttalExperimentConfig(
        output_dir=tmp_path,
        recommendations_file=Path("experiments/paper_simulation_recommendations.json"),
    )
    assert expected_raw_rows(config) == 154_000


def test_matching_checkpoints_are_resumed_without_changing_results(tmp_path: Path) -> None:
    config = RebuttalExperimentConfig(
        output_dir=tmp_path,
        recommendations_file=Path("experiments/paper_simulation_recommendations.json"),
        replications=2,
        n_grid=(20,),
        score_qualities=("well_calibrated",),
        methods=("labeled_only", "linear_loo"),
        make_figures=False,
    )
    first_raw, _ = run_rebuttal_experiment(config)
    second_raw, _ = run_rebuttal_experiment(config)
    pd.testing.assert_frame_equal(first_raw, second_raw)
    with (tmp_path / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["cell_runtimes"][0]["runtime_seconds"] == 0.0
    assert "resumed_from" in manifest["cell_runtimes"][0]
