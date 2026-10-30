#!/usr/bin/env python3
"""Replay and diagnose the largest RePPI errors in the confirmatory run.

The audit is deliberately read-only with respect to the confirmatory artifacts:
it hashes and validates ``raw_results.csv`` and its manifest, regenerates only
the selected replications from their recorded seeds, and writes separate
diagnostic files.  No result row is silently accepted when its replayed point
estimate or standard error differs beyond the explicitly reported floating
point tolerance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rebuttal_sample_size import (  # noqa: E402
    REPPI_BASELINE_CONFIG_PATH,
    REPPI_MLP_CANDIDATES,
    REPPI_RF_TREES,
    REBUTTAL_SCORE_QUALITIES,
    REBUTTAL_SEED,
    REBUTTAL_UNLABELED_RATIO,
    _mlp_candidate_name,
    _replicate_seed,
    deterministic_three_way_folds,
    reppi_mlp_cv_score_only,
    reppi_rf_score_only,
)
from experiments.simulate import generate_sample, score_setting_from_record  # noqa: E402


TARGET_ESTIMATORS = ("reppi_rf", "reppi_mlp_cv")
DEFAULT_RAW_RESULTS = (
    ROOT / "outputs" / "simulations" / "rebuttal_sample_size" / "raw_results.csv"
)
DEFAULT_RECOMMENDATIONS = ROOT / "experiments" / "paper_simulation_recommendations.json"
DEFAULT_OUTPUT_PREFIX = (
    ROOT
    / "outputs"
    / "simulations"
    / "rebuttal_sample_size"
    / "reppi_tail_audit"
)
DEFAULT_TOP_K = 10
PARITY_EPSILON_MULTIPLIER = 32.0

REQUIRED_RAW_COLUMNS = {
    "dgp",
    "score_quality",
    "score_setting_id",
    "score_family",
    "n_labeled",
    "n_unlabeled",
    "unlabeled_ratio",
    "replicate",
    "replicate_seed",
    "psi0",
    "estimator",
    "estimate",
    "se",
    "bias",
    "fit_status",
    "singular_fallbacks",
    "model_fit_failures",
    "nonfinite_prediction_failures",
    "convergence_warnings",
    "selected_candidates",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        encoded = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    else:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (encoded + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _parity_details(raw_value: float, replayed_value: float) -> dict[str, Any]:
    """Return strict, scale-aware floating point replay diagnostics."""

    if not math.isfinite(raw_value) or not math.isfinite(replayed_value):
        return {
            "passed": False,
            "bitwise_equal": False,
            "absolute_difference": float("inf"),
            "allowed_difference": 0.0,
        }
    scale = max(1.0, abs(raw_value), abs(replayed_value))
    allowed = PARITY_EPSILON_MULTIPLIER * np.finfo(float).eps * scale
    difference = abs(raw_value - replayed_value)
    return {
        "passed": bool(difference <= allowed),
        "bitwise_equal": bool(raw_value == replayed_value),
        "absolute_difference": float(difference),
        "allowed_difference": float(allowed),
    }


def _as_int(row: Mapping[str, str], name: str) -> int:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"Raw result is missing integer value {name!r}.")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Raw result has invalid integer {name}={value!r}.")
    return int(numeric)


def _as_float(row: Mapping[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"Raw result is missing numeric value {name!r}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Raw result has non-finite {name}={value!r}.")
    return numeric


def select_tail_rows(raw_results: Path, *, top_k: int = DEFAULT_TOP_K) -> list[dict[str, str]]:
    """Select the global largest absolute errors within each RePPI estimator."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    candidates: dict[str, list[dict[str, str]]] = {
        estimator: [] for estimator in TARGET_ESTIMATORS
    }
    seen: set[tuple[str, str, int, int]] = set()
    with raw_results.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_RAW_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Raw results lack tail-audit columns: " + ", ".join(sorted(missing))
            )
        for source_row in reader:
            estimator = source_row["estimator"]
            if estimator not in candidates:
                continue
            row = dict(source_row)
            quality = row["score_quality"]
            n = _as_int(row, "n_labeled")
            replicate = _as_int(row, "replicate")
            key = (estimator, quality, n, replicate)
            if key in seen:
                raise ValueError(f"Duplicate RePPI raw key: {key}.")
            seen.add(key)
            estimate = _as_float(row, "estimate")
            psi0 = _as_float(row, "psi0")
            bias = _as_float(row, "bias")
            bias_parity = _parity_details(bias, estimate - psi0)
            if not bias_parity["passed"]:
                raise ValueError(f"Raw bias is inconsistent with estimate - psi0 for {key}.")
            row["_absolute_error"] = repr(abs(estimate - psi0))
            candidates[estimator].append(row)

    selected: list[dict[str, str]] = []
    for estimator in TARGET_ESTIMATORS:
        if not candidates[estimator]:
            raise ValueError(f"Raw results contain no rows for {estimator}.")
        ordered = sorted(
            candidates[estimator],
            key=lambda row: (
                -float(row["_absolute_error"]),
                row["score_quality"],
                _as_int(row, "n_labeled"),
                _as_int(row, "replicate"),
                _as_int(row, "replicate_seed"),
            ),
        )
        for rank, row in enumerate(ordered[:top_k], start=1):
            row["_tail_rank"] = str(rank)
            selected.append(row)
    return selected


def _independent_coefficient(
    outcome: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, bool]:
    """Recompute the production Cov/Var coefficient and prediction variance."""

    if len(outcome) < 2:
        return 0.0, 0.0, True
    centered_prediction = prediction - np.mean(prediction)
    prediction_variance = float(
        np.dot(centered_prediction, centered_prediction) / (len(prediction) - 1)
    )
    scale = max(1.0, float(np.mean(prediction**2)))
    threshold = np.finfo(float).eps * scale * 32.0
    if not np.isfinite(prediction_variance) or prediction_variance <= threshold:
        return 0.0, prediction_variance, True
    centered_outcome = outcome - np.mean(outcome)
    covariance = float(
        np.dot(centered_outcome, centered_prediction) / (len(outcome) - 1)
    )
    coefficient = covariance / prediction_variance
    if not np.isfinite(coefficient):
        return 0.0, prediction_variance, True
    return float(coefficient), prediction_variance, False


def reconstruct_rf_details(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    seed: int,
    n_estimators: int = REPPI_RF_TREES,
) -> SimpleNamespace:
    """Independently refit the RF rotations and retain evaluation diagnostics."""

    y = np.asarray(y_l, dtype=float).reshape(-1)
    s_l = np.asarray(score_l, dtype=float).reshape(-1)
    s_u = np.asarray(score_u, dtype=float).reshape(-1)
    n = len(y)
    N = len(s_u)
    c = N / (n + N)
    folds = deterministic_three_way_folds(n, seed)
    labeled_components = np.empty(n, dtype=float)
    unlabeled_components = np.zeros(N, dtype=float)
    evaluation_predictions = np.empty(n, dtype=float)
    coefficients: list[float] = []
    prediction_variances: list[float] = []
    evaluation_folds: list[tuple[int, ...]] = []
    model_seeds: list[int] = []
    warning_messages: list[str] = []
    singular_fallbacks = 0
    model_fit_failures = 0
    nonfinite_prediction_failures = 0

    rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for rotation, (_, train_fold_id, evaluation_fold_id) in enumerate(rotations):
        train_idx = folds[train_fold_id]
        evaluation_idx = folds[evaluation_fold_id]
        model_seed = int((int(seed) + 104729 * (rotation + 1)) % (2**32 - 1))
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=model_seed,
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.fit(s_l[train_idx, None], y[train_idx])
                prediction_evaluation = np.asarray(
                    model.predict(s_l[evaluation_idx, None]),
                    dtype=float,
                ).reshape(-1)
            except Exception as error:
                model_fit_failures += 1
                raise RuntimeError(
                    f"RF replay fit/prediction failed in rotation {rotation}."
                ) from error
        warning_messages.extend(str(item.message) for item in caught)
        if not np.all(np.isfinite(prediction_evaluation)):
            nonfinite_prediction_failures += 1
            raise RuntimeError(
                f"RF replay produced non-finite evaluation predictions in rotation {rotation}."
            )
        coefficient, prediction_variance, singular = _independent_coefficient(
            y[evaluation_idx],
            prediction_evaluation,
        )
        if singular:
            singular_fallbacks += 1
        weight = len(evaluation_idx) / n
        evaluation_predictions[evaluation_idx] = prediction_evaluation
        labeled_components[evaluation_idx] = (
            y[evaluation_idx] - c * coefficient * prediction_evaluation
        )
        if coefficient != 0.0:
            prediction_unlabeled = np.asarray(
                model.predict(s_u[:, None]),
                dtype=float,
            ).reshape(-1)
            if not np.all(np.isfinite(prediction_unlabeled)):
                nonfinite_prediction_failures += 1
                raise RuntimeError(
                    f"RF replay produced non-finite unlabeled predictions in rotation {rotation}."
                )
            unlabeled_components += c * weight * coefficient * prediction_unlabeled
        coefficients.append(float(coefficient))
        prediction_variances.append(float(prediction_variance))
        evaluation_folds.append(tuple(int(index) for index in evaluation_idx))
        model_seeds.append(model_seed)

    estimate = float(np.mean(labeled_components) + np.mean(unlabeled_components))
    variance = np.var(labeled_components, ddof=1) / n
    variance += np.var(unlabeled_components, ddof=1) / N
    se = float(np.sqrt(max(float(variance), 0.0)))
    return SimpleNamespace(
        estimate=estimate,
        se=se,
        evaluation_predictions=evaluation_predictions,
        coefficients=tuple(coefficients),
        evaluation_prediction_variances=tuple(prediction_variances),
        evaluation_folds=tuple(evaluation_folds),
        model_seeds=tuple(model_seeds),
        singular_fallbacks=singular_fallbacks,
        model_fit_failures=model_fit_failures,
        nonfinite_prediction_failures=nonfinite_prediction_failures,
        convergence_warnings=0,
        warning_messages=tuple(warning_messages),
    )


def _fold_prediction_variances(
    predictions: np.ndarray,
    folds: Sequence[Sequence[int]],
) -> tuple[float, ...]:
    values: list[float] = []
    for fold in folds:
        fold_prediction = predictions[np.asarray(fold, dtype=int)]
        if len(fold_prediction) < 2:
            values.append(0.0)
        else:
            centered = fold_prediction - np.mean(fold_prediction)
            values.append(
                float(np.dot(centered, centered) / (len(fold_prediction) - 1))
            )
    return tuple(values)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _validate_row_and_generate_sample(
    row: Mapping[str, str],
    recommendations: Mapping[str, Mapping[str, object]],
) -> dict[str, np.ndarray]:
    quality = row["score_quality"]
    if quality not in REBUTTAL_SCORE_QUALITIES or quality not in recommendations:
        raise ValueError(f"Unknown score quality {quality!r}.")
    if row["dgp"] != "monotone":
        raise ValueError(f"Tail replay supports only the frozen monotone DGP, not {row['dgp']!r}.")
    n = _as_int(row, "n_labeled")
    N = _as_int(row, "n_unlabeled")
    ratio = _as_int(row, "unlabeled_ratio")
    if ratio != REBUTTAL_UNLABELED_RATIO or N != ratio * n:
        raise ValueError(f"Row has non-frozen sample-size ratio for {quality}, n={n}.")
    replicate = _as_int(row, "replicate")
    seed = _as_int(row, "replicate_seed")
    quality_index = REBUTTAL_SCORE_QUALITIES.index(quality)
    expected_seed = _replicate_seed(REBUTTAL_SEED, quality_index, n, replicate)
    if seed != expected_seed:
        raise ValueError(
            f"Recorded seed {seed} differs from frozen seed {expected_seed} "
            f"for {quality}, n={n}, replicate={replicate}."
        )
    setting = score_setting_from_record(dict(recommendations[quality]))
    if row["score_setting_id"] != setting.setting_id or row["score_family"] != setting.family:
        raise ValueError(f"Raw score setting does not match frozen recommendation for {quality}.")
    sample = generate_sample(
        n=n,
        N=N,
        dgp="monotone",
        setting=setting,
        rng=np.random.default_rng(seed),
    )
    psi_parity = _parity_details(_as_float(row, "psi0"), float(sample["psi0"]))
    if not psi_parity["passed"]:
        raise ValueError(
            f"Regenerated psi0 differs for {quality}, n={n}, replicate={replicate}."
        )
    return sample


def _raw_candidates(row: Mapping[str, str]) -> tuple[str, ...]:
    encoded = row.get("selected_candidates", "")
    return tuple(value for value in encoded.split("|") if value)


def replay_tail_row(
    row: Mapping[str, str],
    recommendations: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Replay one selected row and return machine-checkable diagnostics."""

    sample = _validate_row_and_generate_sample(row, recommendations)
    y = np.asarray(sample["y_l"], dtype=float)
    score_l = np.asarray(sample["score_l"], dtype=float)
    score_u = np.asarray(sample["score_u"], dtype=float)
    seed = _as_int(row, "replicate_seed")
    estimator = row["estimator"]

    if estimator == "reppi_rf":
        replay = reppi_rf_score_only(y, score_l, score_u, seed=seed)
        independent = reconstruct_rf_details(y, score_l, score_u, seed=seed)
        independent_estimate_parity = _parity_details(
            replay.estimate,
            independent.estimate,
        )
        independent_se_parity = _parity_details(replay.se, independent.se)
        if not independent_estimate_parity["passed"] or not independent_se_parity["passed"]:
            raise RuntimeError("Independent RF reconstruction disagrees with production replay.")
        if any(
            not _parity_details(left, right)["passed"]
            for left, right in zip(replay.coefficients, independent.coefficients)
        ):
            raise RuntimeError("Independent RF coefficients disagree with production replay.")
        prediction_variances = independent.evaluation_prediction_variances
        selected_candidates: tuple[str, ...] = ()
        candidate_cv_mses: tuple[tuple[float | None, ...], ...] = ()
        cv_split_seeds: tuple[int, ...] = ()
        final_model_seeds = tuple(int(value) for value in replay.model_seeds)
        model_seeds = tuple(int(value) for value in replay.model_seeds)
        warning_messages = independent.warning_messages
        model_fit_failures = independent.model_fit_failures
        nonfinite_prediction_failures = independent.nonfinite_prediction_failures
        convergence_warnings = independent.convergence_warnings
    elif estimator == "reppi_mlp_cv":
        replay = reppi_mlp_cv_score_only(y, score_l, score_u, seed=seed)
        prediction_variances = _fold_prediction_variances(
            replay.evaluation_predictions,
            replay.evaluation_folds,
        )
        selected_candidates = tuple(replay.selected_candidates)
        candidate_cv_mses = tuple(
            tuple(_finite_or_none(value) for value in fold_values)
            for fold_values in replay.candidate_cv_mses
        )
        cv_split_seeds = tuple(int(value) for value in replay.cv_split_seeds)
        final_model_seeds = tuple(int(value) for value in replay.final_model_seeds)
        model_seeds = ()
        warning_messages = ()
        model_fit_failures = int(replay.model_fit_failures)
        nonfinite_prediction_failures = int(
            replay.nonfinite_prediction_failures
        )
        convergence_warnings = int(replay.convergence_warnings)
    else:
        raise ValueError(f"Unsupported estimator {estimator!r}.")

    raw_estimate = _as_float(row, "estimate")
    raw_se = _as_float(row, "se")
    psi0 = _as_float(row, "psi0")
    estimate_parity = _parity_details(raw_estimate, float(replay.estimate))
    se_parity = _parity_details(raw_se, float(replay.se))
    if not estimate_parity["passed"] or not se_parity["passed"]:
        raise RuntimeError(
            "Replay parity failed for "
            f"{estimator}, {row['score_quality']}, n={row['n_labeled']}, "
            f"replicate={row['replicate']}: estimate={estimate_parity}, se={se_parity}."
        )

    raw_telemetry = {
        "singular_fallbacks": _as_int(row, "singular_fallbacks"),
        "model_fit_failures": _as_int(row, "model_fit_failures"),
        "nonfinite_prediction_failures": _as_int(
            row,
            "nonfinite_prediction_failures",
        ),
        "convergence_warnings": _as_int(row, "convergence_warnings"),
        "selected_candidates": _raw_candidates(row),
    }
    replay_telemetry = {
        "singular_fallbacks": int(replay.singular_fallbacks),
        "model_fit_failures": model_fit_failures,
        "nonfinite_prediction_failures": nonfinite_prediction_failures,
        "convergence_warnings": convergence_warnings,
        "selected_candidates": selected_candidates,
    }
    if raw_telemetry != replay_telemetry:
        raise RuntimeError(
            f"Replay telemetry differs for {estimator}, {row['score_quality']}, "
            f"n={row['n_labeled']}, replicate={row['replicate']}: "
            f"{raw_telemetry} != {replay_telemetry}."
        )

    folds = tuple(tuple(int(index) for index in fold) for fold in replay.evaluation_folds)
    return {
        "estimator": estimator,
        "tail_rank_within_estimator": _as_int(row, "_tail_rank"),
        "score_quality": row["score_quality"],
        "score_setting_id": row["score_setting_id"],
        "n_labeled": _as_int(row, "n_labeled"),
        "n_unlabeled": _as_int(row, "n_unlabeled"),
        "replicate": _as_int(row, "replicate"),
        "replicate_seed": seed,
        "psi0": psi0,
        "raw_estimate": raw_estimate,
        "raw_error": raw_estimate - psi0,
        "raw_absolute_error": abs(raw_estimate - psi0),
        "raw_se": raw_se,
        "replayed_estimate": float(replay.estimate),
        "replayed_error": float(replay.estimate - psi0),
        "replayed_se": float(replay.se),
        "estimate_parity": estimate_parity,
        "se_parity": se_parity,
        "coefficients": [float(value) for value in replay.coefficients],
        "evaluation_prediction_variances": [
            float(value) for value in prediction_variances
        ],
        "evaluation_fold_sizes": [len(fold) for fold in folds],
        "evaluation_folds": [list(fold) for fold in folds],
        "selected_mlp_candidates": list(selected_candidates),
        "candidate_cv_mses": [list(values) for values in candidate_cv_mses],
        "cv_split_seeds": list(cv_split_seeds),
        "final_model_seeds": list(final_model_seeds),
        "rf_model_seeds": list(model_seeds),
        "singular_fallbacks": int(replay.singular_fallbacks),
        "model_fit_failures": model_fit_failures,
        "nonfinite_prediction_failures": nonfinite_prediction_failures,
        "convergence_warnings": convergence_warnings,
        "warning_messages": list(warning_messages),
        "fit_status": row["fit_status"],
        "parity_passed": True,
    }


def _implementation_paths() -> dict[str, Path]:
    return {
        "rebuttal_sample_size.py": ROOT / "experiments" / "rebuttal_sample_size.py",
        "reppi_baseline_config.json": REPPI_BASELINE_CONFIG_PATH,
        "simulate.py": ROOT / "experiments" / "simulate.py",
        "estimators.py": ROOT / "experiments" / "estimators.py",
    }


def validate_canonical_inputs(
    raw_results: Path,
    recommendations_file: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Validate canonical hashes and frozen RePPI defaults before replay."""

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "complete":
        raise ValueError("Canonical manifest is not complete.")
    raw_hash = sha256(raw_results)
    if manifest.get("output_hashes", {}).get("raw_results.csv") != raw_hash:
        raise ValueError("Canonical manifest raw_results.csv hash does not match input.")

    implementation_hashes = manifest.get("implementation_hashes", {})
    for name, path in _implementation_paths().items():
        expected = implementation_hashes.get(name)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Canonical implementation hash mismatch for {name}.")

    recommendations_hash = sha256(recommendations_file)
    matching_inputs = [
        expected
        for name, expected in manifest.get("input_hashes", {}).items()
        if Path(name).name == recommendations_file.name
    ]
    if recommendations_hash not in matching_inputs:
        raise ValueError("Canonical recommendations hash does not match input.")

    defaults = manifest.get("confirmatory_defaults", {})
    if int(defaults.get("seed", -1)) != REBUTTAL_SEED:
        raise ValueError("Canonical manifest seed differs from the frozen seed.")
    if int(defaults.get("unlabeled_ratio", -1)) != REBUTTAL_UNLABELED_RATIO:
        raise ValueError("Canonical manifest unlabeled ratio is not frozen.")
    if int(defaults.get("reppi_rf_trees", -1)) != REPPI_RF_TREES:
        raise ValueError("Canonical manifest RF tree count is not frozen.")
    methods = set(defaults.get("methods", ()))
    if not set(TARGET_ESTIMATORS).issubset(methods):
        raise ValueError("Canonical manifest does not contain both RePPI estimators.")
    expected_candidates = [
        {
            "hidden_layer_sizes": list(layers),
            "alpha": alpha,
            "name": _mlp_candidate_name((layers, alpha)),
        }
        for layers, alpha in REPPI_MLP_CANDIDATES
    ]
    if defaults.get("reppi_mlp_candidates") != expected_candidates:
        raise ValueError("Canonical manifest MLP grid differs from the frozen grid.")

    input_hashes = {
        str(raw_results.resolve()): raw_hash,
        str(recommendations_file.resolve()): recommendations_hash,
        str(manifest_path.resolve()): sha256(manifest_path),
    }
    input_hashes.update(
        {
            str(path.resolve()): sha256(path)
            for path in _implementation_paths().values()
        }
    )
    return input_hashes


CSV_COLUMNS = (
    "estimator",
    "tail_rank_within_estimator",
    "score_quality",
    "score_setting_id",
    "n_labeled",
    "n_unlabeled",
    "replicate",
    "replicate_seed",
    "psi0",
    "raw_estimate",
    "raw_error",
    "raw_absolute_error",
    "raw_se",
    "replayed_estimate",
    "replayed_error",
    "replayed_se",
    "estimate_parity",
    "se_parity",
    "coefficients",
    "evaluation_prediction_variances",
    "evaluation_fold_sizes",
    "selected_mlp_candidates",
    "candidate_cv_mses",
    "cv_split_seeds",
    "final_model_seeds",
    "rf_model_seeds",
    "singular_fallbacks",
    "model_fit_failures",
    "nonfinite_prediction_failures",
    "convergence_warnings",
    "warning_messages",
    "fit_status",
    "parity_passed",
)


def _csv_payload(records: Sequence[Mapping[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(CSV_COLUMNS),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    nested_columns = {
        "estimate_parity",
        "se_parity",
        "coefficients",
        "evaluation_prediction_variances",
        "evaluation_fold_sizes",
        "selected_mlp_candidates",
        "candidate_cv_mses",
        "cv_split_seeds",
        "final_model_seeds",
        "rf_model_seeds",
        "warning_messages",
    }
    for record in records:
        encoded = dict(record)
        for name in nested_columns:
            encoded[name] = json.dumps(
                encoded[name],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        writer.writerow(encoded)
    return stream.getvalue().encode("utf-8")


def run_tail_audit(
    *,
    raw_results: Path,
    recommendations_file: Path,
    manifest_path: Path,
    output_prefix: Path,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Path, Path, Path]:
    """Run the replay audit and write CSV, JSON, and hash-manifest outputs."""

    input_hashes = validate_canonical_inputs(
        raw_results,
        recommendations_file,
        manifest_path,
    )
    with recommendations_file.open("r", encoding="utf-8") as stream:
        recommendations = json.load(stream)
    selected_rows = select_tail_rows(raw_results, top_k=top_k)
    records = [replay_tail_row(row, recommendations) for row in selected_rows]
    if len(records) != top_k * len(TARGET_ESTIMATORS):
        raise RuntimeError("Tail audit did not replay the expected number of rows.")
    if not all(record["parity_passed"] for record in records):
        raise RuntimeError("At least one tail replay failed parity.")

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    hash_path = output_prefix.with_suffix(".sha256.json")
    csv_payload = _csv_payload(records)
    _write_atomic(csv_path, csv_payload)
    csv_hash = sha256(csv_path)

    report_without_payload_hash = {
        "study": "neurips_2026_reppi_tail_replay",
        "status": "passed",
        "selection": {
            "criterion": "global largest absolute estimate minus psi0 within estimator",
            "top_k_per_estimator": top_k,
            "estimators": list(TARGET_ESTIMATORS),
            "selected_rows": len(records),
            "deterministic_tie_break": [
                "score_quality",
                "n_labeled",
                "replicate",
                "replicate_seed",
            ],
        },
        "parity_policy": {
            "epsilon_multiplier": PARITY_EPSILON_MULTIPLIER,
            "formula": (
                "abs(raw-replayed) <= multiplier * float64_epsilon "
                "* max(1, abs(raw), abs(replayed))"
            ),
            "bitwise_equality_also_reported": True,
        },
        "input_hashes": input_hashes,
        "output_hashes": {
            str(csv_path.resolve()): csv_hash,
        },
        "records": records,
    }
    canonical_payload = _json_bytes(report_without_payload_hash, pretty=False)
    report = dict(report_without_payload_hash)
    report["output_hashes"] = {
        **report_without_payload_hash["output_hashes"],
        "canonical_json_payload_before_self_hash": _sha256_bytes(canonical_payload),
    }
    _write_atomic(json_path, _json_bytes(report, pretty=True))

    hash_manifest = {
        "input_hashes": input_hashes,
        "output_hashes": {
            str(csv_path.resolve()): sha256(csv_path),
            str(json_path.resolve()): sha256(json_path),
        },
    }
    _write_atomic(hash_path, _json_bytes(hash_manifest, pretty=True))
    return csv_path, json_path, hash_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", type=Path, default=DEFAULT_RAW_RESULTS)
    parser.add_argument(
        "--recommendations-file",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Canonical manifest; defaults to raw_results.csv's directory.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest or args.raw_results.with_name("manifest.json")
    paths = run_tail_audit(
        raw_results=args.raw_results,
        recommendations_file=args.recommendations_file,
        manifest_path=manifest_path,
        output_prefix=args.output_prefix,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "outputs": [str(path) for path in paths],
                "rows_replayed": args.top_k * len(TARGET_ESTIMATORS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
