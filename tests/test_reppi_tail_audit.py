from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import rebuttal_sample_size
from experiments.rebuttal_sample_size import (
    _replicate_seed,
    deterministic_three_way_folds,
    reppi_rf_score_only,
)
from experiments.simulate import generate_sample, score_setting_from_record
from scripts import audit_reppi_tail


RECOMMENDATIONS = ROOT / "experiments" / "paper_simulation_recommendations.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _base_row(
    *,
    estimator: str,
    quality: str,
    n: int,
    replicate: int,
    seed: int,
    psi0: float,
    estimate: float,
    se: float,
) -> dict[str, object]:
    recommendations = json.loads(RECOMMENDATIONS.read_text(encoding="utf-8"))
    setting = score_setting_from_record(recommendations[quality])
    return {
        "dgp": "monotone",
        "score_quality": quality,
        "score_setting_id": setting.setting_id,
        "score_family": setting.family,
        "n_labeled": n,
        "n_unlabeled": 16 * n,
        "unlabeled_ratio": 16,
        "replicate": replicate,
        "replicate_seed": seed,
        "psi0": psi0,
        "estimator": estimator,
        "estimate": estimate,
        "se": se,
        "bias": estimate - psi0,
        "fit_status": "ok",
        "singular_fallbacks": 0,
        "model_fit_failures": 0,
        "nonfinite_prediction_failures": 0,
        "convergence_warnings": 0,
        "selected_candidates": "",
    }


def _write_manifest(raw_path: Path, manifest_path: Path) -> None:
    implementation_paths = {
        "rebuttal_sample_size.py": ROOT / "experiments" / "rebuttal_sample_size.py",
        "reppi_baseline_config.json": (
            ROOT / "experiments" / "reppi_baseline_config.json"
        ),
        "simulate.py": ROOT / "experiments" / "simulate.py",
        "estimators.py": ROOT / "experiments" / "estimators.py",
    }
    manifest = {
        "status": "complete",
        "output_hashes": {"raw_results.csv": _sha256(raw_path)},
        "implementation_hashes": {
            name: _sha256(path) for name, path in implementation_paths.items()
        },
        "input_hashes": {str(RECOMMENDATIONS): _sha256(RECOMMENDATIONS)},
        "confirmatory_defaults": {
            "seed": rebuttal_sample_size.REBUTTAL_SEED,
            "unlabeled_ratio": rebuttal_sample_size.REBUTTAL_UNLABELED_RATIO,
            "methods": list(rebuttal_sample_size.REBUTTAL_METHODS),
            "reppi_rf_trees": rebuttal_sample_size.REPPI_RF_TREES,
            "reppi_mlp_candidates": [
                {
                    "hidden_layer_sizes": list(layers),
                    "alpha": alpha,
                    "name": rebuttal_sample_size._mlp_candidate_name((layers, alpha)),
                }
                for layers, alpha in rebuttal_sample_size.REPPI_MLP_CANDIDATES
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_select_tail_rows_is_global_per_estimator_and_deterministic(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for estimator in audit_reppi_tail.TARGET_ESTIMATORS:
        for replicate, error in enumerate((0.1, -0.7, 0.7, 0.2)):
            row = _base_row(
                estimator=estimator,
                quality="well_calibrated",
                n=20,
                replicate=replicate,
                seed=100 + replicate,
                psi0=0.5,
                estimate=0.5 + error,
                se=0.1,
            )
            rows.append(row)
    raw_path = tmp_path / "raw.csv"
    _write_csv(raw_path, rows)

    selected = audit_reppi_tail.select_tail_rows(raw_path, top_k=2)

    assert len(selected) == 4
    for estimator in audit_reppi_tail.TARGET_ESTIMATORS:
        estimator_rows = [row for row in selected if row["estimator"] == estimator]
        # Equal absolute errors use replicate as the deterministic tie break.
        assert [int(row["replicate"]) for row in estimator_rows] == [1, 2]
        assert [int(row["_tail_rank"]) for row in estimator_rows] == [1, 2]


def test_rf_diagnostic_refit_matches_production_result() -> None:
    rng = np.random.default_rng(2026)
    score_l = np.linspace(0.02, 0.98, 30)
    y = rng.binomial(1, score_l).astype(float)
    score_u = np.linspace(0.01, 0.99, 75)

    production = reppi_rf_score_only(
        y,
        score_l,
        score_u,
        seed=811,
        n_estimators=10,
    )
    replay = audit_reppi_tail.reconstruct_rf_details(
        y,
        score_l,
        score_u,
        seed=811,
        n_estimators=10,
    )

    assert replay.estimate == pytest.approx(production.estimate, abs=1e-15)
    assert replay.se == pytest.approx(production.se, abs=1e-15)
    assert replay.coefficients == pytest.approx(production.coefficients, abs=1e-15)
    assert replay.evaluation_folds == production.evaluation_folds
    assert replay.model_seeds == production.model_seeds
    assert replay.evaluation_prediction_variances == pytest.approx(
        audit_reppi_tail._fold_prediction_variances(
            replay.evaluation_predictions,
            replay.evaluation_folds,
        )
    )


def test_tail_audit_writes_hash_checked_reports_and_fails_on_parity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recommendations = json.loads(RECOMMENDATIONS.read_text(encoding="utf-8"))
    quality = "well_calibrated"
    n = 20
    replicate = 0
    seed = _replicate_seed(
        rebuttal_sample_size.REBUTTAL_SEED,
        rebuttal_sample_size.REBUTTAL_SCORE_QUALITIES.index(quality),
        n,
        replicate,
    )
    sample = generate_sample(
        n=n,
        N=16 * n,
        dgp="monotone",
        setting=score_setting_from_record(recommendations[quality]),
        rng=np.random.default_rng(seed),
    )
    y = np.asarray(sample["y_l"], dtype=float)
    psi0 = float(sample["psi0"])
    folds = deterministic_three_way_folds(n, seed)
    evaluation_folds = (
        tuple(int(index) for index in folds[2]),
        tuple(int(index) for index in folds[0]),
        tuple(int(index) for index in folds[1]),
    )
    estimate = float(np.mean(y))
    se = float(np.sqrt(np.var(y, ddof=1) / n))

    def fake_rf(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            estimate=estimate,
            se=se,
            coefficients=(0.0, 0.0, 0.0),
            evaluation_folds=evaluation_folds,
            model_seeds=(1, 2, 3),
            singular_fallbacks=0,
        )

    def fake_rf_reconstruction(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            estimate=estimate,
            se=se,
            coefficients=(0.0, 0.0, 0.0),
            evaluation_prediction_variances=(0.0, 0.0, 0.0),
            evaluation_folds=evaluation_folds,
            model_seeds=(1, 2, 3),
            singular_fallbacks=0,
            model_fit_failures=0,
            nonfinite_prediction_failures=0,
            convergence_warnings=0,
            warning_messages=(),
        )

    def fake_mlp(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            estimate=estimate,
            se=se,
            coefficients=(0.0, 0.0, 0.0),
            evaluation_predictions=np.zeros(n),
            evaluation_folds=evaluation_folds,
            selected_candidates=(),
            candidate_cv_mses=((), (), ()),
            cv_split_seeds=(11, 12, 13),
            final_model_seeds=(21, 22, 23),
            singular_fallbacks=0,
            model_fit_failures=0,
            nonfinite_prediction_failures=0,
            convergence_warnings=0,
        )

    monkeypatch.setattr(audit_reppi_tail, "reppi_rf_score_only", fake_rf)
    monkeypatch.setattr(
        audit_reppi_tail,
        "reconstruct_rf_details",
        fake_rf_reconstruction,
    )
    monkeypatch.setattr(audit_reppi_tail, "reppi_mlp_cv_score_only", fake_mlp)

    rows = [
        _base_row(
            estimator=estimator,
            quality=quality,
            n=n,
            replicate=replicate,
            seed=seed,
            psi0=psi0,
            estimate=estimate,
            se=se,
        )
        for estimator in audit_reppi_tail.TARGET_ESTIMATORS
    ]
    raw_path = tmp_path / "raw_results.csv"
    manifest_path = tmp_path / "manifest.json"
    _write_csv(raw_path, rows)
    _write_manifest(raw_path, manifest_path)
    prefix = tmp_path / "tail"

    csv_path, json_path, hash_path = audit_reppi_tail.run_tail_audit(
        raw_results=raw_path,
        recommendations_file=RECOMMENDATIONS,
        manifest_path=manifest_path,
        output_prefix=prefix,
        top_k=1,
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert len(report["records"]) == 2
    assert all(record["parity_passed"] for record in report["records"])
    assert hashes["output_hashes"][str(csv_path.resolve())] == _sha256(csv_path)
    assert hashes["output_hashes"][str(json_path.resolve())] == _sha256(json_path)
    assert hashes["input_hashes"][str(raw_path.resolve())] == _sha256(raw_path)

    rows[0]["estimate"] = float(rows[0]["estimate"]) + 1e-5
    rows[0]["bias"] = float(rows[0]["estimate"]) - psi0
    _write_csv(raw_path, rows)
    _write_manifest(raw_path, manifest_path)
    with pytest.raises(RuntimeError, match="Replay parity failed"):
        audit_reppi_tail.run_tail_audit(
            raw_results=raw_path,
            recommendations_file=RECOMMENDATIONS,
            manifest_path=manifest_path,
            output_prefix=tmp_path / "bad_tail",
            top_k=1,
        )
