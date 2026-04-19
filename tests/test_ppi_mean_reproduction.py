from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from ppi_aipw import mean_inference

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.estimators import auto_aipw_pointestimate_and_se
from experiments.ppi_mean_reproduction import (
    isotonic_calibrated_predictions,
    isotonic_min10_calibrated_predictions,
    linear_calibrated_predictions,
    main,
    monotone_spline_calibrated_predictions,
    run_auto_calibration_estimator,
    run_monotone_spline_estimator,
    platt_calibrated_predictions,
    run_aipw_estimator,
    run_aipw_em_estimator,
    run_classical_estimator,
    run_isotonic_calibration_estimator,
    run_linear_calibration_estimator,
    run_ppi_estimator,
    run_ppi_plus_plus_estimator,
    venn_abers_calibrated_predictions,
)


def write_fake_dataset_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    n_binary = 240
    x_bin = rng.normal(size=(n_binary, 2))
    logits = 0.9 * x_bin[:, 0] - 0.6 * x_bin[:, 1]
    p_bin = 1.0 / (1.0 + np.exp(-logits))
    y_forest = (rng.random(n_binary) < p_bin).astype(float)
    yhat_forest = np.clip(0.05 + 0.9 * p_bin, 0.01, 0.99)
    np.savez(cache_dir / "forest.npz", Y=y_forest, Yhat=yhat_forest)

    y_galaxies = (rng.random(n_binary) < np.clip(0.15 + 0.7 * p_bin, 0.02, 0.98)).astype(float)
    yhat_galaxies = np.clip(0.02 + 0.85 * p_bin, 0.01, 0.99)
    np.savez(cache_dir / "galaxies.npz", Y=y_galaxies, Yhat=yhat_galaxies)

    n_cont = 260
    x_cont = rng.normal(size=(n_cont, 2))
    y_income = 42000.0 + 6500.0 * x_cont[:, 0] - 4200.0 * x_cont[:, 1] + rng.normal(scale=1800.0, size=n_cont)
    yhat_income = y_income + rng.normal(scale=1200.0, size=n_cont)
    np.savez(cache_dir / "census_income.npz", Y=y_income, Yhat=yhat_income, X=x_cont)


def test_linear_calibration_matches_perfect_predictions() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    yhat = y.copy()
    pred_l, pred_u = linear_calibrated_predictions(y, yhat, yhat)
    np.testing.assert_allclose(pred_l, y)
    np.testing.assert_allclose(pred_u, yhat)


def test_constant_prediction_degenerates_to_labeled_mean() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat_l = np.full_like(y, 0.2)
    yhat_u = np.full(6, 0.2)
    linear = run_linear_calibration_estimator(y, yhat_l, yhat_u, alpha=0.1)
    isotonic = run_isotonic_calibration_estimator(y, yhat_l, yhat_u, alpha=0.1)
    classical = run_classical_estimator(y, alpha=0.1)
    assert linear["estimate"] == pytest.approx(classical["estimate"])
    assert isotonic["estimate"] == pytest.approx(classical["estimate"])


def test_isotonic_preserves_bounded_binary_range() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    yhat_l = np.array([-3.0, 0.1, 0.9, 2.0], dtype=float)
    yhat_u = np.array([-10.0, -1.0, 0.5, 1.5, 100.0], dtype=float)
    pred_l, pred_u = isotonic_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)


def test_tuned_isotonic_predictions_are_bounded_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat_l = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_u = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    pred_l, pred_u = isotonic_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)
    assert np.all(np.diff(pred_l[np.argsort(yhat_l)]) >= -1e-12)
    assert pred_u.shape == yhat_u.shape


def test_min10_isotonic_predictions_are_bounded_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat_l = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_u = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    pred_l, pred_u = isotonic_min10_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)
    assert np.all(np.diff(pred_l[np.argsort(yhat_l)]) >= -1e-12)
    assert pred_u.shape == yhat_u.shape


def test_venn_abers_predictions_stay_bounded() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=float)
    yhat_l = np.array([0.05, 0.15, 0.75, 0.85, 0.25, 0.65], dtype=float)
    yhat_u = np.array([0.1, 0.2, 0.4, 0.8, 0.9], dtype=float)
    pred_l, pred_u = venn_abers_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)


def test_platt_predictions_stay_bounded() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=float)
    yhat_l = np.array([0.05, 0.15, 0.75, 0.85, 0.25, 0.65], dtype=float)
    yhat_u = np.array([0.1, 0.2, 0.4, 0.8, 0.9], dtype=float)
    pred_l, pred_u = platt_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)


def test_monotone_spline_predictions_stay_bounded_and_monotone() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    yhat_l = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], dtype=float)
    yhat_u = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    pred_l, pred_u = monotone_spline_calibrated_predictions(y, yhat_l, yhat_u)
    assert np.all(pred_l >= 0.0)
    assert np.all(pred_l <= 1.0)
    assert np.all(pred_u >= 0.0)
    assert np.all(pred_u <= 1.0)
    assert np.all(np.diff(pred_l[np.argsort(yhat_l)]) >= -1e-12)
    assert pred_u.shape == yhat_u.shape


def test_aipw_matches_pooled_plugin_plus_augmentation() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    yhat_l = np.array([0.1, 0.4, 0.6, 0.8], dtype=float)
    yhat_u = np.array([0.2, 0.3, 0.7], dtype=float)
    result = run_aipw_estimator(y, yhat_l, yhat_u, alpha=0.1)
    rho = len(yhat_l) / (len(yhat_l) + len(yhat_u))
    pooled_plugin = rho * yhat_l.mean() + (1.0 - rho) * yhat_u.mean()
    expected = pooled_plugin + np.mean(y - yhat_l)
    assert result["estimate"] == pytest.approx(expected)
    assert result["se"] > 0.0


def test_auto_baseline_matches_vendored_helper() -> None:
    rng = np.random.default_rng(8)
    y = rng.normal(size=140)
    yhat_l = y + rng.normal(scale=0.35, size=140)
    yhat_u = rng.normal(loc=float(np.mean(y)), scale=1.0, size=220)

    result = run_auto_calibration_estimator(y, yhat_l, yhat_u, alpha=0.1)
    expected = auto_aipw_pointestimate_and_se(
        y,
        yhat_l,
        yhat_u,
        candidate_methods=("aipw", "linear", "monotone_spline", "isotonic"),
        num_folds=20,
        random_state=0,
    )

    assert result["estimate"] == pytest.approx(expected["estimate"])
    assert result["se"] == pytest.approx(expected["se"])


def test_monotone_spline_baseline_runs() -> None:
    rng = np.random.default_rng(9)
    y = rng.normal(size=160)
    yhat_l = y + rng.normal(scale=0.3, size=160)
    yhat_u = rng.normal(loc=float(np.mean(y)), scale=1.1, size=240)
    result = run_monotone_spline_estimator(y, yhat_l, yhat_u, alpha=0.1)
    assert np.isfinite(result["estimate"])
    assert result["se"] >= 0.0


def test_ppi_parity_with_upstream_on_fixed_split() -> None:
    rng = np.random.default_rng(0)
    y_total = rng.normal(size=300)
    yhat_total = y_total + rng.normal(scale=0.2, size=300)
    idx = rng.permutation(y_total.shape[0])
    n = 100
    y_l = y_total[idx[:n]]
    yhat_l = yhat_total[idx[:n]]
    yhat_u = yhat_total[idx[n:]]

    ours = run_ppi_estimator(y_l, yhat_l, yhat_u, alpha=0.1)
    upstream_estimate = float(np.asarray(ppi_mean_pointestimate(y_l, yhat_l, yhat_u, lam=1)).reshape(-1)[0])
    upstream_ci = ppi_mean_ci(y_l, yhat_l, yhat_u, alpha=0.1, lam=1)

    assert ours["estimate"] == pytest.approx(upstream_estimate)
    assert ours["ci_lower"] == pytest.approx(float(np.asarray(upstream_ci[0]).reshape(-1)[0]))
    assert ours["ci_upper"] == pytest.approx(float(np.asarray(upstream_ci[1]).reshape(-1)[0]))


def test_ppi_plus_plus_parity_with_upstream_on_fixed_split() -> None:
    rng = np.random.default_rng(1)
    y_total = rng.normal(size=320)
    yhat_total = y_total + rng.normal(scale=0.35, size=320)
    idx = rng.permutation(y_total.shape[0])
    n = 110
    y_l = y_total[idx[:n]]
    yhat_l = yhat_total[idx[:n]]
    yhat_u = yhat_total[idx[n:]]

    ours = run_ppi_plus_plus_estimator(y_l, yhat_l, yhat_u, alpha=0.1)
    upstream_estimate = float(np.asarray(ppi_mean_pointestimate(y_l, yhat_l, yhat_u, lam=None)).reshape(-1)[0])
    upstream_ci = ppi_mean_ci(y_l, yhat_l, yhat_u, alpha=0.1, lam=None)

    assert ours["estimate"] == pytest.approx(upstream_estimate)
    assert ours["ci_lower"] == pytest.approx(float(np.asarray(upstream_ci[0]).reshape(-1)[0]))
    assert ours["ci_upper"] == pytest.approx(float(np.asarray(upstream_ci[1]).reshape(-1)[0]))


def test_aipw_em_parity_with_package_api_on_fixed_split() -> None:
    rng = np.random.default_rng(2)
    y_total = rng.normal(size=340)
    yhat_total = y_total + rng.normal(scale=0.4, size=340)
    idx = rng.permutation(y_total.shape[0])
    n = 120
    y_l = y_total[idx[:n]]
    yhat_l = yhat_total[idx[:n]]
    yhat_u = yhat_total[idx[n:]]

    ours = run_aipw_em_estimator(y_l, yhat_l, yhat_u, alpha=0.1)
    expected = mean_inference(
        y_l,
        yhat_l,
        yhat_u,
        method="aipw",
        alpha=0.1,
        efficiency_maximization=True,
    )

    assert ours["estimate"] == pytest.approx(float(expected.pointestimate))
    assert ours["se"] == pytest.approx(float(expected.se))
    assert ours["ci_lower"] == pytest.approx(float(np.asarray(expected.ci[0]).reshape(-1)[0]))
    assert ours["ci_upper"] == pytest.approx(float(np.asarray(expected.ci[1]).reshape(-1)[0]))


def test_aipw_em_differs_from_clipped_ppi_plus_plus_when_lambda_exceeds_one() -> None:
    yhat_l = np.array([-0.3163, 0.411631, 1.042513, -0.128535, 1.366463], dtype=float)
    y_l = np.array([-0.54097, 0.652597, 1.654117, -0.183401, 1.975345], dtype=float)
    yhat_u = np.array([-0.60607, -0.049271, 0.764234, -0.711542, 0.248989, 0.30893, 1.149015], dtype=float)

    aipw_em = run_aipw_em_estimator(y_l, yhat_l, yhat_u, alpha=0.1)
    ppi_plus_plus = run_ppi_plus_plus_estimator(y_l, yhat_l, yhat_u, alpha=0.1)
    package = mean_inference(
        y_l,
        yhat_l,
        yhat_u,
        method="aipw",
        alpha=0.1,
        efficiency_maximization=True,
    )

    assert float(package.efficiency_lambda) > 1.0
    assert aipw_em["estimate"] == pytest.approx(float(package.pointestimate))
    assert aipw_em["estimate"] != pytest.approx(ppi_plus_plus["estimate"])


def test_smoke_run_writes_expected_outputs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_fake_dataset_cache(cache_dir)

    output_dir = tmp_path / "ppi_smoke"
    main(
        [
            "--datasets",
            "galaxies_mean",
            "--replications",
            "2",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--smoke",
        ]
    )

    assert (output_dir / "raw_results.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "table_galaxies_mean.csv").exists()
    assert (output_dir / "fig_galaxies_mean_relative_efficiency.pdf").exists()
    assert (output_dir / "fig_galaxies_mean_coverage.pdf").exists()
    assert (output_dir / "fig_galaxies_mean_mse_vs_ppi.pdf").exists()

    summary = pd.read_csv(output_dir / "summary.csv")
    assert set(summary["estimator"]) == {
        "classical",
        "imputation",
        "aipw",
        "ppi",
        "ppi_plus_plus",
        "aipw_em",
        "auto_calibration",
        "monotone_spline",
        "linear_calibration",
        "platt_calibration",
        "isotonic_calibration_min10",
        "venn_abers_calibration",
    }
    expected_columns = {
        "dataset",
        "source_notebook",
        "estimand",
        "n_labeled",
        "n_unlabeled",
        "estimator",
        "mean_estimate",
        "mean_bias",
        "emp_variance",
        "mse",
        "rmse",
        "coverage",
        "rel_eff_vs_ppi",
        "mse_ratio_vs_ppi",
    }
    assert expected_columns.issubset(summary.columns)
    ppi_rows = summary[summary["estimator"] == "ppi"]
    np.testing.assert_allclose(ppi_rows["mse_ratio_vs_ppi"], 1.0)


def test_reproduce_script_smoke_exports_paper_assets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "cache"
    output_root = tmp_path / "outputs"
    paper_assets = tmp_path / "paper_assets"
    write_fake_dataset_cache(cache_dir)

    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable

    subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/reproduce_paper.sh"),
            "--smoke",
            "--output-root",
            str(output_root),
            "--paper-assets-dir",
            str(paper_assets),
            "--cache-dir",
            str(cache_dir),
            "--seed",
            "0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    expected_assets = {
        "sim_fig_main.pdf",
        "sim_fig_bias_coverage.pdf",
        "fig_paper_main_grid.pdf",
        "fig_paper_metric_grid.pdf",
        "table_paper_summary.tex",
    }
    assert expected_assets.issubset({path.name for path in paper_assets.iterdir()})
