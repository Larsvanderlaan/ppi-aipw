#!/usr/bin/env python3
"""Validate the local RePPI adapter against the pinned public source.

The January-2025 public repository predates the June-2026 v2 paper.  It is
therefore used only for source provenance, its supported MLP code path, and a
single-fold scalar-algebra parity check.  The v2 cross-fit wrapper is not used
as the production oracle because Algorithm 1 changed its scaling and fold
aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor


COMMIT = "60924986fa23ee53af5500104e5821f16dd3a9e5"
REPPI_SHA256 = "085b702bf8cd4dfa47fe50b0809fd9d7c1cf4865a9e4aac36c2b4a11fb3f8b9c"
PPI_SHA256 = "85dad473a8026821d9c4cfc58fc615916c83a878d069f85e7bf73304cff85f35"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_module(source_dir: Path) -> ModuleType:
    module_path = source_dir / "reppi.py"
    spec = importlib.util.spec_from_file_location(
        "reppi_official_60924986",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official RePPI source from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(source_dir))
    return module


def validate_source(source_dir: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes = {
        "reppi.py": sha256(source_dir / "reppi.py"),
        "ppi.py": sha256(source_dir / "ppi.py"),
    }
    if commit != COMMIT:
        raise ValueError(f"Official RePPI commit mismatch: {commit} != {COMMIT}.")
    if hashes["reppi.py"] != REPPI_SHA256:
        raise ValueError("Official reppi.py hash mismatch.")
    if hashes["ppi.py"] != PPI_SHA256:
        raise ValueError("Official ppi.py hash mismatch.")
    return {"commit": commit, **hashes}


def validate_public_mlp_path(module: ModuleType) -> dict[str, float]:
    rng = np.random.default_rng(410)
    score = np.linspace(0.03, 0.97, 18)
    outcome = rng.binomial(1, score).astype(float)
    X = np.ones((len(score), 1), dtype=float)
    Yhat = score[:, None]
    theta = np.array([0.45])
    r = 0.25
    model_seed = 812
    original_constructor = module.MLPRegressor

    def deterministic_constructor(*args: Any, **kwargs: Any) -> MLPRegressor:
        kwargs.setdefault("random_state", model_seed)
        return original_constructor(*args, **kwargs)

    module.MLPRegressor = deterministic_constructor
    try:
        official_gradient = module.grad_fit_ols(
            X,
            outcome,
            Yhat,
            theta,
            r,
            method="mlp",
        )
    finally:
        module.MLPRegressor = original_constructor
    evaluation_score = np.linspace(0.07, 0.93, 11)
    official_value = np.asarray(
        official_gradient(
            np.ones((len(evaluation_score), 1), dtype=float),
            evaluation_score[:, None],
        ),
        dtype=float,
    ).reshape(-1)
    recovered_prediction = theta[0] - (1.0 + r) * official_value

    local_model = MLPRegressor(
        hidden_layer_sizes=(32, 32, 32),
        random_state=model_seed,
    )
    local_model.fit(np.column_stack([np.ones(len(score)), score]), outcome)
    local_prediction = np.asarray(
        local_model.predict(
            np.column_stack([np.ones(len(evaluation_score)), evaluation_score])
        ),
        dtype=float,
    )
    maximum_error = float(np.max(np.abs(recovered_prediction - local_prediction)))
    if maximum_error > 1e-12:
        raise ValueError(f"Official MLP path parity error is {maximum_error}.")
    return {"maximum_absolute_prediction_error": maximum_error}


def validate_single_fold_algebra(module: ModuleType) -> dict[str, float]:
    y = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.7], dtype=float)
    score_l = np.array([0.04, 0.18, 0.37, 0.61, 0.79, 0.95], dtype=float)
    score_u = np.linspace(0.02, 0.98, 23)
    X = np.ones((len(y), 1), dtype=float)
    X_u = np.ones((len(score_u), 1), dtype=float)
    theta0 = np.array([0.46])
    r = len(y) / len(score_u)

    def prediction(score: np.ndarray) -> np.ndarray:
        values = np.asarray(score, dtype=float).reshape(-1)
        return 0.12 + 0.58 * values + 0.21 * values**2

    def gradient(X_new: np.ndarray, score_new: np.ndarray) -> np.ndarray:
        fitted = prediction(score_new)
        return X_new * (theta0[0] - fitted)[:, None] / (1.0 + r)

    official_estimate, official_coefficient = module.ppi_opt_ols_pointestimate(
        X,
        y[:, None],
        score_l[:, None],
        X_u,
        score_u[:, None],
        gradient,
        theta0,
    )
    pred_l = prediction(score_l)
    pred_u = prediction(score_u)
    coefficient = float(
        np.cov(y, pred_l, ddof=1)[0, 1] / np.var(pred_l, ddof=1)
    )
    c = len(score_u) / (len(y) + len(score_u))
    manual_estimate = float(
        np.mean(y) + c * coefficient * (np.mean(pred_u) - np.mean(pred_l))
    )
    estimate_error = abs(float(official_estimate[0, 0]) - manual_estimate)
    coefficient_error = abs(float(official_coefficient[0, 0]) - coefficient)
    if estimate_error > 1e-12 or coefficient_error > 1e-12:
        raise ValueError(
            "Official single-fold scalar algebra does not match the manual "
            f"specialization: estimate={estimate_error}, coefficient={coefficient_error}."
        )
    return {
        "estimate_absolute_error": estimate_error,
        "coefficient_absolute_error": coefficient_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    source = validate_source(source_dir)
    module = load_official_module(source_dir)
    report = {
        "status": "passed",
        "source": source,
        "public_mlp_path": validate_public_mlp_path(module),
        "single_fold_scalar_algebra": validate_single_fold_algebra(module),
        "production_oracle": "June-2026 paper v2 Algorithm 1",
        "public_crossfit_wrapper_used_as_production_oracle": False,
        "known_public_v2_differences": [
            "The public wrapper averages three folds equally; v2 uses evaluation-fold-size weights.",
            "The public core recomputes |D_eval|/N; v2 uses the global n/N scaling.",
        ],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
