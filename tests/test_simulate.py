from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from ppi_py import ppi_mean_pointestimate
from ppi_aipw import mean_inference

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.estimators import auto_aipw_pointestimate_and_se
from experiments.simulate import (
    DEFAULT_SCORE_SETTINGS,
    PROFILES,
    generate_sample,
    official_ppi_summary,
    run_one,
)


def test_run_one_includes_ppi_plus_plus_and_matches_upstream() -> None:
    rng = np.random.default_rng(7)
    sample = generate_sample(
        n=40,
        N=160,
        dgp="monotone",
        setting=DEFAULT_SCORE_SETTINGS["poorly_calibrated"],
        rng=rng,
    )
    results, _ = run_one(sample, PROFILES["quick"], rng_seed=0)
    result_by_estimator = {row["estimator"]: row for row in results}

    assert "ppi_plus_plus" in result_by_estimator

    expected = float(
        np.asarray(
            ppi_mean_pointestimate(sample["y_l"], sample["score_l"], sample["score_u"], lam=None)
        ).reshape(-1)[0]
    )
    assert result_by_estimator["ppi_plus_plus"]["estimate"] == pytest.approx(expected)

    estimate, se, lower, upper = official_ppi_summary(
        sample["y_l"],
        sample["score_l"],
        sample["score_u"],
        lam=None,
    )
    assert result_by_estimator["ppi_plus_plus"]["estimate"] == pytest.approx(estimate)
    assert result_by_estimator["ppi_plus_plus"]["se"] == pytest.approx(se)
    assert result_by_estimator["ppi_plus_plus"]["ci_lower"] == pytest.approx(lower)
    assert result_by_estimator["ppi_plus_plus"]["ci_upper"] == pytest.approx(upper)


def test_run_one_includes_aipw_em_and_matches_package_api() -> None:
    rng = np.random.default_rng(17)
    sample = generate_sample(
        n=40,
        N=160,
        dgp="monotone",
        setting=DEFAULT_SCORE_SETTINGS["poorly_calibrated"],
        rng=rng,
    )
    results, _ = run_one(sample, PROFILES["quick"], rng_seed=0)
    result_by_estimator = {row["estimator"]: row for row in results}

    assert "aipw_em" in result_by_estimator

    expected = mean_inference(
        sample["y_l"],
        sample["score_l"],
        sample["score_u"],
        method="aipw",
        alpha=0.05,
        efficiency_maximization=True,
    )
    expected_lower, expected_upper = expected.ci

    assert result_by_estimator["aipw_em"]["estimate"] == pytest.approx(float(expected.pointestimate))
    assert result_by_estimator["aipw_em"]["se"] == pytest.approx(float(expected.se))
    assert result_by_estimator["aipw_em"]["ci_lower"] == pytest.approx(float(expected_lower))
    assert result_by_estimator["aipw_em"]["ci_upper"] == pytest.approx(float(expected_upper))


def test_run_one_includes_auto_baseline_and_matches_vendored_helper() -> None:
    rng = np.random.default_rng(11)
    sample = generate_sample(
        n=40,
        N=160,
        dgp="monotone",
        setting=DEFAULT_SCORE_SETTINGS["poorly_calibrated"],
        rng=rng,
    )
    results, _ = run_one(sample, PROFILES["quick"], rng_seed=123)
    result_by_estimator = {row["estimator"]: row for row in results}

    assert "auto_calibration" in result_by_estimator

    expected = auto_aipw_pointestimate_and_se(
        sample["y_l"],
        sample["score_l"],
        sample["score_u"],
        candidate_methods=("aipw", "linear", "monotone_spline", "isotonic"),
        num_folds=20,
        random_state=123,
    )
    assert result_by_estimator["auto_calibration"]["estimate"] == pytest.approx(expected["estimate"])
    assert result_by_estimator["auto_calibration"]["se"] == pytest.approx(expected["se"])
    assert "monotone_spline" in result_by_estimator
