"""Locked sample-size experiment for the NeurIPS rebuttal.

This module is intentionally experiment-local.  It does not extend the public
``ppi_aipw`` API.  The confirmatory defaults encode the design fixed before the
new simulation was run: two pre-existing score regimes, seven labeled sample
sizes, ``N = 16 n``, 1,000 replications, and seed 20260411.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from experiments.estimators import (
    aipp_from_prediction,
    fit_linear_calibration,
    fit_monotone_spline_calibration,
    fit_sklearn_isotonic_calibration,
    influence_se_from_prediction,
    influence_se_labeled_only,
    labeled_mean,
    plugin_estimate,
    predict_linear,
    predict_monotone_spline,
    predict_sklearn_isotonic,
)
from experiments.simulate import (
    ScoreSetting,
    ci_bounds,
    generate_sample,
    official_ppi_summary,
    score_setting_from_record,
)


REBUTTAL_N_GRID: Tuple[int, ...] = (20, 50, 100, 200, 400, 800, 1200)
REBUTTAL_METHODS: Tuple[str, ...] = (
    "labeled_only",
    "ppi",
    "ppi_plus_plus",
    "aipw",
    "linear",
    "linear_loo",
    "isotonic",
    "isotonic_loo",
    "monotone_spline",
    "reppi_rf",
    "reppi_mlp_cv",
)
REBUTTAL_SCORE_QUALITIES: Tuple[str, ...] = (
    "well_calibrated",
    "poorly_calibrated",
)
REBUTTAL_SEED = 20260411
REBUTTAL_UNLABELED_RATIO = 16
REBUTTAL_REPLICATIONS = 1000
REPPI_RF_TREES = 100
REPPI_FOLDS = 3
REPPI_BASELINE_CONFIG_PATH = Path(__file__).with_name("reppi_baseline_config.json")
with REPPI_BASELINE_CONFIG_PATH.open("r", encoding="utf-8") as _config_stream:
    REPPI_BASELINE_CONFIG = json.load(_config_stream)
REPPI_MLP_INNER_FOLDS = int(REPPI_BASELINE_CONFIG["mlp_cv"]["inner_folds"])
REPPI_MLP_MAX_ITER = int(REPPI_BASELINE_CONFIG["mlp_cv"]["max_iter"])
REPPI_MLP_CANDIDATES: Tuple[Tuple[Tuple[int, ...], float], ...] = tuple(
    (tuple(int(width) for width in layers), float(alpha))
    for layers in REPPI_BASELINE_CONFIG["mlp_cv"]["hidden_layer_sizes"]
    for alpha in REPPI_BASELINE_CONFIG["mlp_cv"]["alpha"]
)
EXPERIMENT_IMPLEMENTATION_VERSION = "2026-07-23-v2-reppi-mlp-cv"


@dataclass(frozen=True)
class LOOAIPWResult:
    estimate: float
    se: float
    heldout_predictions: np.ndarray
    averaged_unlabeled_predictions: np.ndarray
    singular_fallbacks: int


@dataclass(frozen=True)
class RePPIRFResult:
    estimate: float
    se: float
    labeled_components: np.ndarray
    unlabeled_components: np.ndarray
    coefficients: Tuple[float, ...]
    evaluation_folds: Tuple[Tuple[int, ...], ...]
    model_seeds: Tuple[int, ...]
    singular_fallbacks: int


@dataclass(frozen=True)
class RePPIMLPCVResult:
    estimate: float
    se: float
    labeled_components: np.ndarray
    unlabeled_components: np.ndarray
    evaluation_predictions: np.ndarray
    coefficients: Tuple[float, ...]
    evaluation_folds: Tuple[Tuple[int, ...], ...]
    selected_candidates: Tuple[str, ...]
    candidate_cv_mses: Tuple[Tuple[float, ...], ...]
    cv_split_seeds: Tuple[int, ...]
    final_model_seeds: Tuple[int, ...]
    singular_fallbacks: int
    model_fit_failures: int
    nonfinite_prediction_failures: int
    convergence_warnings: int


@dataclass(frozen=True)
class RebuttalExperimentConfig:
    output_dir: Path
    recommendations_file: Path
    replications: int = REBUTTAL_REPLICATIONS
    seed: int = REBUTTAL_SEED
    n_grid: Tuple[int, ...] = REBUTTAL_N_GRID
    unlabeled_ratio: int = REBUTTAL_UNLABELED_RATIO
    score_qualities: Tuple[str, ...] = REBUTTAL_SCORE_QUALITIES
    methods: Tuple[str, ...] = REBUTTAL_METHODS
    jobs: int = 1
    make_figures: bool = True
    resume: bool = True


def _validate_sample(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_l, dtype=float).reshape(-1)
    s_l = np.asarray(score_l, dtype=float).reshape(-1)
    s_u = np.asarray(score_u, dtype=float).reshape(-1)
    if len(y) != len(s_l):
        raise ValueError("y_l and score_l must have the same length.")
    if len(y) < 2 or len(s_u) < 2:
        raise ValueError("LOO/two-sample inference requires at least two observations per sample.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(s_l)) or not np.all(np.isfinite(s_u)):
        raise ValueError("Outcome and score arrays must be finite.")
    return y, s_l, s_u


def two_sample_crossfit_se(
    y_l: np.ndarray,
    heldout_predictions: np.ndarray,
    averaged_unlabeled_predictions: np.ndarray,
) -> float:
    """Return the stated cross-fitted two-sample standard error.

    The predictions supplied here are *unscaled*.  The function applies
    ``c = N / (n + N)`` to both labeled and unlabeled prediction components.
    """

    y, pred_l, pred_u = _validate_sample(
        y_l,
        heldout_predictions,
        averaged_unlabeled_predictions,
    )
    n = len(y)
    N = len(pred_u)
    c = N / (n + N)
    labeled_component = y - c * pred_l
    unlabeled_component = c * pred_u
    variance = np.var(labeled_component, ddof=1) / n
    variance += np.var(unlabeled_component, ddof=1) / N
    return float(np.sqrt(max(float(variance), 0.0)))


def loo_aipw_from_predictions(
    y_l: np.ndarray,
    heldout_predictions: np.ndarray,
    averaged_unlabeled_predictions: np.ndarray,
    *,
    singular_fallbacks: int = 0,
) -> LOOAIPWResult:
    """Assemble the exact LOO-AIPW point estimate and two-sample SE."""

    y, pred_l, pred_u = _validate_sample(
        y_l,
        heldout_predictions,
        averaged_unlabeled_predictions,
    )
    n = len(y)
    N = len(pred_u)
    c = N / (n + N)
    estimate = float(np.mean(y - c * pred_l) + np.mean(c * pred_u))
    se = two_sample_crossfit_se(y, pred_l, pred_u)
    return LOOAIPWResult(
        estimate=estimate,
        se=se,
        heldout_predictions=pred_l.copy(),
        averaged_unlabeled_predictions=pred_u.copy(),
        singular_fallbacks=int(singular_fallbacks),
    )


def _loo_linear_predictions(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute exact OLS LOO predictions from sufficient statistics in O(n+N)."""

    y, s_l, s_u = _validate_sample(y_l, score_l, score_u)
    n = len(y)
    train_size = n - 1
    sum_x = float(np.sum(s_l))
    sum_y = float(np.sum(y))
    sum_xx = float(np.dot(s_l, s_l))
    sum_xy = float(np.dot(s_l, y))

    train_sum_x = sum_x - s_l
    train_sum_y = sum_y - y
    train_sum_xx = sum_xx - s_l**2
    train_sum_xy = sum_xy - s_l * y
    train_mean_x = train_sum_x / train_size
    train_mean_y = train_sum_y / train_size
    centered_xx = train_sum_xx - train_sum_x**2 / train_size
    centered_xy = train_sum_xy - train_sum_x * train_sum_y / train_size
    scale = np.maximum(1.0, np.abs(train_sum_xx))
    nonsingular = np.abs(centered_xx) > np.finfo(float).eps * scale * 32.0
    slopes = np.zeros(n, dtype=float)
    slopes[nonsingular] = centered_xy[nonsingular] / centered_xx[nonsingular]
    intercepts = train_mean_y - slopes * train_mean_x
    heldout = intercepts + slopes * s_l

    # qbar(s) = n^{-1} sum_i q_{-i}(s) is affine.  This avoids an n by N
    # prediction matrix and is both time- and memory-linear.
    averaged_unlabeled = np.mean(intercepts) + np.mean(slopes) * s_u
    return heldout, averaged_unlabeled, int(np.sum(~nonsingular))


def loo_linear_aipw(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
) -> LOOAIPWResult:
    heldout, averaged_unlabeled, fallbacks = _loo_linear_predictions(y_l, score_l, score_u)
    return loo_aipw_from_predictions(
        y_l,
        heldout,
        averaged_unlabeled,
        singular_fallbacks=fallbacks,
    )


def loo_isotonic_aipw(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
) -> LOOAIPWResult:
    """Fit exact leave-one-out isotonic calibrators using O(N) memory."""

    y, s_l, s_u = _validate_sample(y_l, score_l, score_u)
    n = len(y)
    heldout = np.empty(n, dtype=float)
    averaged_unlabeled = np.zeros(len(s_u), dtype=float)
    fallbacks = 0
    keep = np.ones(n, dtype=bool)
    for idx in range(n):
        keep[idx] = False
        train_y = y[keep]
        if np.allclose(s_l[keep], s_l[keep][0]) or np.isclose(np.max(train_y), np.min(train_y)):
            prediction = float(np.mean(train_y))
            heldout[idx] = prediction
            averaged_unlabeled += prediction / n
            fallbacks += 1
        else:
            model = fit_sklearn_isotonic_calibration(s_l[keep], train_y)
            heldout[idx] = float(predict_sklearn_isotonic(model, s_l[idx : idx + 1])[0])
            # Only this length-N running accumulator is retained.
            averaged_unlabeled += predict_sklearn_isotonic(model, s_u) / n
        keep[idx] = True
    return loo_aipw_from_predictions(
        y,
        heldout,
        averaged_unlabeled,
        singular_fallbacks=fallbacks,
    )


def deterministic_three_way_folds(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < REPPI_FOLDS:
        raise ValueError("RePPI requires at least three labeled observations.")
    permutation = np.random.default_rng(seed).permutation(n)
    folds = tuple(np.asarray(fold, dtype=int) for fold in np.array_split(permutation, REPPI_FOLDS))
    if any(len(fold) == 0 for fold in folds):
        raise ValueError("RePPI produced an empty fold.")
    return folds  # type: ignore[return-value]


def _covariance_variance_coefficient(y: np.ndarray, prediction: np.ndarray) -> Optional[float]:
    if len(y) < 2:
        return None
    pred_centered = prediction - np.mean(prediction)
    variance = float(np.dot(pred_centered, pred_centered) / (len(prediction) - 1))
    scale = max(1.0, float(np.mean(prediction**2)))
    if not np.isfinite(variance) or variance <= np.finfo(float).eps * scale * 32.0:
        return None
    y_centered = y - np.mean(y)
    covariance = float(np.dot(y_centered, pred_centered) / (len(y) - 1))
    coefficient = covariance / variance
    return float(coefficient) if np.isfinite(coefficient) else None


def reppi_rf_score_only(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    seed: int,
    n_estimators: int = REPPI_RF_TREES,
) -> RePPIRFResult:
    """Three-way cross-fitted scalar-mean RePPI with score-only random forests.

    For rotation ``k``, one fold supplies the preliminary target, the next
    trains the imputation rule, and the third estimates the
    covariance/variance coefficient and correction.  In the scalar-mean
    specialization, fitting the published gradient is algebraically equivalent
    to fitting ``Y`` on the score: the preliminary-target shift and gradient
    scale cancel after centering and safety scaling.  The evaluation-fold
    estimates are weighted by their labeled fold sizes.
    """

    y, s_l, s_u = _validate_sample(y_l, score_l, score_u)
    n = len(y)
    N = len(s_u)
    c = N / (n + N)
    folds = deterministic_three_way_folds(n, seed)
    labeled_components = np.empty(n, dtype=float)
    unlabeled_components = np.zeros(N, dtype=float)
    coefficients: List[float] = []
    evaluation_folds: List[Tuple[int, ...]] = []
    model_seeds: List[int] = []
    singular_fallbacks = 0

    # (preliminary, model-training, coefficient/evaluation) rotations match the
    # published three-way construction.
    rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for rotation, (_, train_fold_id, eval_fold_id) in enumerate(rotations):
        train_idx = folds[train_fold_id]
        eval_idx = folds[eval_fold_id]
        model_seed = int((int(seed) + 104729 * (rotation + 1)) % (2**32 - 1))
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=model_seed,
            n_jobs=1,
        )
        model.fit(s_l[train_idx, None], y[train_idx])
        pred_eval = np.asarray(model.predict(s_l[eval_idx, None]), dtype=float)
        coefficient = _covariance_variance_coefficient(y[eval_idx], pred_eval)
        weight = len(eval_idx) / n
        if coefficient is None:
            # This fold contributes its labeled mean and no unlabeled
            # correction.  If every fold is singular, the full estimator and
            # its standard error exactly reduce to the labeled mean.
            coefficient = 0.0
            singular_fallbacks += 1
        labeled_components[eval_idx] = y[eval_idx] - c * coefficient * pred_eval
        if coefficient != 0.0:
            pred_u = np.asarray(model.predict(s_u[:, None]), dtype=float)
            unlabeled_components += c * weight * coefficient * pred_u
        coefficients.append(float(coefficient))
        evaluation_folds.append(tuple(int(idx) for idx in eval_idx))
        model_seeds.append(model_seed)

    estimate = float(np.mean(labeled_components) + np.mean(unlabeled_components))
    variance = np.var(labeled_components, ddof=1) / n
    variance += np.var(unlabeled_components, ddof=1) / N
    se = float(np.sqrt(max(float(variance), 0.0)))
    return RePPIRFResult(
        estimate=estimate,
        se=se,
        labeled_components=labeled_components,
        unlabeled_components=unlabeled_components,
        coefficients=tuple(coefficients),
        evaluation_folds=tuple(evaluation_folds),
        model_seeds=tuple(model_seeds),
        singular_fallbacks=singular_fallbacks,
    )


def _derived_reppi_seed(seed: int, *components: int) -> int:
    sequence = np.random.SeedSequence(
        [int(seed), 0x52455050, *(int(component) for component in components)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _mlp_candidate_name(candidate: Tuple[Tuple[int, ...], float]) -> str:
    layers, alpha = candidate
    layer_code = "x".join(str(width) for width in layers)
    alpha_code = f"{alpha:.0e}".replace("+", "")
    return f"h{layer_code}_a{alpha_code}"


def _reppi_mlp_features(score: np.ndarray) -> np.ndarray:
    """Return the authors' ``[X, Yhat]`` features for an intercept-only target."""

    values = np.asarray(score, dtype=float).reshape(-1)
    return np.column_stack([np.ones(len(values), dtype=float), values])


def _fit_reppi_mlp(
    score: np.ndarray,
    outcome: np.ndarray,
    *,
    candidate: Tuple[Tuple[int, ...], float],
    seed: int,
) -> Tuple[Optional[MLPRegressor], int, int]:
    """Fit one frozen-grid MLP and return model, warnings, and fit failures."""

    layers, alpha = candidate
    settings = REPPI_BASELINE_CONFIG["mlp_cv"]
    model = MLPRegressor(
        hidden_layer_sizes=layers,
        alpha=float(alpha),
        activation=str(settings["activation"]),
        solver=str(settings["solver"]),
        learning_rate_init=float(settings["learning_rate_init"]),
        max_iter=REPPI_MLP_MAX_ITER,
        tol=float(settings["tol"]),
        early_stopping=bool(settings["early_stopping"]),
        random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            model.fit(_reppi_mlp_features(score), np.asarray(outcome, dtype=float))
        except Exception:
            convergence_count = sum(
                isinstance(item.message, ConvergenceWarning) for item in caught
            )
            return None, int(convergence_count), 1
    convergence_count = sum(
        isinstance(item.message, ConvergenceWarning) for item in caught
    )
    return model, int(convergence_count), 0


def _select_reppi_mlp(
    y: np.ndarray,
    score: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
    rotation: int,
    candidates: Sequence[Tuple[Tuple[int, ...], float]],
) -> Tuple[
    Optional[MLPRegressor],
    str,
    Tuple[float, ...],
    int,
    int,
    int,
    int,
    int,
]:
    """Tune an MLP strictly inside the score-training fold.

    The outer coefficient/evaluation fold is never passed to this function.
    Candidate selection uses only mean held-out outcome MSE, with candidate
    order as the deterministic tie-break.
    """

    split_seed = _derived_reppi_seed(seed, rotation, 0)
    final_seed = _derived_reppi_seed(seed, rotation, 1)
    convergence_warnings = 0
    fit_failures = 0
    nonfinite_failures = 0
    if len(train_idx) < REPPI_MLP_INNER_FOLDS:
        return (
            None,
            "fallback_labeled_mean",
            tuple(float("inf") for _ in candidates),
            split_seed,
            final_seed,
            fit_failures,
            nonfinite_failures,
            convergence_warnings,
        )

    splitter = KFold(
        n_splits=REPPI_MLP_INNER_FOLDS,
        shuffle=True,
        random_state=split_seed,
    )
    inner_splits = list(splitter.split(train_idx))
    cv_mses: List[float] = []
    for candidate_index, candidate in enumerate(candidates):
        squared_error_sum = 0.0
        validation_count = 0
        candidate_valid = True
        for inner_fold, (fit_position, validation_position) in enumerate(inner_splits):
            fit_idx = train_idx[np.asarray(fit_position, dtype=int)]
            validation_idx = train_idx[np.asarray(validation_position, dtype=int)]
            model_seed = _derived_reppi_seed(
                seed,
                rotation,
                2,
                candidate_index,
                inner_fold,
            )
            model, warning_count, failure_count = _fit_reppi_mlp(
                score[fit_idx],
                y[fit_idx],
                candidate=candidate,
                seed=model_seed,
            )
            convergence_warnings += warning_count
            fit_failures += failure_count
            if model is None:
                candidate_valid = False
                break
            try:
                prediction = np.asarray(
                    model.predict(_reppi_mlp_features(score[validation_idx])),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                fit_failures += 1
                candidate_valid = False
                break
            if not np.all(np.isfinite(prediction)):
                nonfinite_failures += 1
                candidate_valid = False
                break
            residual = prediction - y[validation_idx]
            squared_error_sum += float(np.dot(residual, residual))
            validation_count += len(validation_idx)
        cv_mses.append(
            squared_error_sum / validation_count
            if candidate_valid and validation_count
            else float("inf")
        )

    finite_candidates = [
        index for index, value in enumerate(cv_mses) if np.isfinite(value)
    ]
    if not finite_candidates:
        return (
            None,
            "fallback_labeled_mean",
            tuple(cv_mses),
            split_seed,
            final_seed,
            fit_failures,
            nonfinite_failures,
            convergence_warnings,
        )
    selected_index = min(finite_candidates, key=lambda index: (cv_mses[index], index))
    selected_candidate = candidates[selected_index]
    final_seed = _derived_reppi_seed(seed, rotation, 3, selected_index)
    final_model, warning_count, failure_count = _fit_reppi_mlp(
        score[train_idx],
        y[train_idx],
        candidate=selected_candidate,
        seed=final_seed,
    )
    convergence_warnings += warning_count
    fit_failures += failure_count
    return (
        final_model,
        _mlp_candidate_name(selected_candidate)
        if final_model is not None
        else "fallback_labeled_mean",
        tuple(cv_mses),
        split_seed,
        final_seed,
        fit_failures,
        nonfinite_failures,
        convergence_warnings,
    )


def reppi_mlp_cv_score_only(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    seed: int,
    candidates: Sequence[Tuple[Tuple[int, ...], float]] = REPPI_MLP_CANDIDATES,
) -> RePPIMLPCVResult:
    """Paper-v2 scalar-mean RePPI with a nested-CV neural score learner.

    This uses the June-2026 paper's three-way rotations, global ``N/(n+N)``
    scaling, and evaluation-fold-size aggregation.  MLP hyperparameters are
    selected by two-fold CV wholly inside the score-training fold.  The frozen
    grid contains both the paper-compatible two-hidden-layer depth and the
    public January-2025 code's ``(32, 32, 32)`` default architecture.
    """

    if not candidates:
        raise ValueError("RePPI-MLP-CV requires at least one candidate.")
    normalized_candidates = tuple(
        (tuple(int(width) for width in layers), float(alpha))
        for layers, alpha in candidates
    )
    if any(not layers or any(width <= 0 for width in layers) or alpha < 0.0
           for layers, alpha in normalized_candidates):
        raise ValueError("Invalid RePPI-MLP-CV candidate.")

    y, s_l, s_u = _validate_sample(y_l, score_l, score_u)
    n = len(y)
    N = len(s_u)
    c = N / (n + N)
    folds = deterministic_three_way_folds(n, seed)
    labeled_components = np.empty(n, dtype=float)
    unlabeled_components = np.zeros(N, dtype=float)
    evaluation_predictions = np.empty(n, dtype=float)
    coefficients: List[float] = []
    evaluation_folds: List[Tuple[int, ...]] = []
    selected_candidates: List[str] = []
    candidate_cv_mses: List[Tuple[float, ...]] = []
    cv_split_seeds: List[int] = []
    final_model_seeds: List[int] = []
    singular_fallbacks = 0
    model_fit_failures = 0
    nonfinite_prediction_failures = 0
    convergence_warning_count = 0

    rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for rotation, (_, train_fold_id, eval_fold_id) in enumerate(rotations):
        train_idx = folds[train_fold_id]
        eval_idx = folds[eval_fold_id]
        (
            model,
            selected_name,
            cv_mses,
            cv_seed,
            final_seed,
            fit_failures,
            nonfinite_failures,
            warning_count,
        ) = _select_reppi_mlp(
            y,
            s_l,
            train_idx,
            seed=seed,
            rotation=rotation,
            candidates=normalized_candidates,
        )
        model_fit_failures += fit_failures
        nonfinite_prediction_failures += nonfinite_failures
        convergence_warning_count += warning_count
        weight = len(eval_idx) / n
        coefficient: Optional[float] = None
        pred_eval = np.zeros(len(eval_idx), dtype=float)
        pred_u: Optional[np.ndarray] = None
        if model is not None:
            prediction_failed = False
            try:
                pred_eval = np.asarray(
                    model.predict(_reppi_mlp_features(s_l[eval_idx])),
                    dtype=float,
                ).reshape(-1)
                pred_u = np.asarray(
                    model.predict(_reppi_mlp_features(s_u)),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                model_fit_failures += 1
                pred_eval = np.zeros(len(eval_idx), dtype=float)
                pred_u = None
                prediction_failed = True
            if not prediction_failed and (
                not np.all(np.isfinite(pred_eval))
                or pred_u is None
                or not np.all(np.isfinite(pred_u))
            ):
                nonfinite_prediction_failures += 1
                pred_eval = np.zeros(len(eval_idx), dtype=float)
                pred_u = None
            elif not prediction_failed:
                coefficient = _covariance_variance_coefficient(y[eval_idx], pred_eval)
        if coefficient is None:
            coefficient = 0.0
            singular_fallbacks += 1
        evaluation_predictions[eval_idx] = pred_eval
        labeled_components[eval_idx] = y[eval_idx] - c * coefficient * pred_eval
        if coefficient != 0.0 and pred_u is not None:
            unlabeled_components += c * weight * coefficient * pred_u
        coefficients.append(float(coefficient))
        evaluation_folds.append(tuple(int(index) for index in eval_idx))
        selected_candidates.append(selected_name)
        candidate_cv_mses.append(cv_mses)
        cv_split_seeds.append(cv_seed)
        final_model_seeds.append(final_seed)

    estimate = float(np.mean(labeled_components) + np.mean(unlabeled_components))
    variance = np.var(labeled_components, ddof=1) / n
    variance += np.var(unlabeled_components, ddof=1) / N
    se = float(np.sqrt(max(float(variance), 0.0)))
    return RePPIMLPCVResult(
        estimate=estimate,
        se=se,
        labeled_components=labeled_components,
        unlabeled_components=unlabeled_components,
        evaluation_predictions=evaluation_predictions,
        coefficients=tuple(coefficients),
        evaluation_folds=tuple(evaluation_folds),
        selected_candidates=tuple(selected_candidates),
        candidate_cv_mses=tuple(candidate_cv_mses),
        cv_split_seeds=tuple(cv_split_seeds),
        final_model_seeds=tuple(final_model_seeds),
        singular_fallbacks=singular_fallbacks,
        model_fit_failures=model_fit_failures,
        nonfinite_prediction_failures=nonfinite_prediction_failures,
        convergence_warnings=convergence_warning_count,
    )


def _result_row(
    estimator: str,
    estimate: float,
    se: float,
    psi0: float,
    *,
    singular_fallbacks: int = 0,
    model_fit_failures: int = 0,
    nonfinite_prediction_failures: int = 0,
    convergence_warnings: int = 0,
    selected_candidates: Sequence[str] = (),
) -> Dict[str, object]:
    lower, upper = ci_bounds(estimate, se)
    return {
        "estimator": estimator,
        "estimate": float(estimate),
        "se": float(se),
        "bias": float(estimate - psi0),
        "sq_error": float((estimate - psi0) ** 2),
        "covered": float(lower <= psi0 <= upper),
        "ci_length": float(upper - lower),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "fit_status": "ok",
        "singular_fallbacks": int(singular_fallbacks),
        "model_fit_failures": int(model_fit_failures),
        "nonfinite_prediction_failures": int(nonfinite_prediction_failures),
        "convergence_warnings": int(convergence_warnings),
        "selected_candidates": "|".join(str(value) for value in selected_candidates),
    }


def run_rebuttal_methods(
    sample: Dict[str, np.ndarray],
    *,
    seed: int,
    methods: Sequence[str] = REBUTTAL_METHODS,
) -> List[Dict[str, object]]:
    unknown = sorted(set(methods) - set(REBUTTAL_METHODS))
    if unknown:
        raise ValueError("Unknown rebuttal methods: " + ", ".join(unknown))
    y = np.asarray(sample["y_l"], dtype=float)
    s_l = np.asarray(sample["score_l"], dtype=float)
    s_u = np.asarray(sample["score_u"], dtype=float)
    psi0 = float(sample["psi0"])
    rows: List[Dict[str, object]] = []

    for method in methods:
        if method == "labeled_only":
            estimate = labeled_mean(y)
            se = influence_se_labeled_only(y, estimate)
            rows.append(_result_row(method, estimate, se, psi0))
        elif method == "ppi":
            estimate, se, lower, upper = official_ppi_summary(y, s_l, s_u, lam=1.0)
            row = _result_row(method, estimate, se, psi0)
            row.update({"ci_lower": lower, "ci_upper": upper, "ci_length": upper - lower})
            row["covered"] = float(lower <= psi0 <= upper)
            rows.append(row)
        elif method == "ppi_plus_plus":
            estimate, se, lower, upper = official_ppi_summary(y, s_l, s_u, lam=None)
            row = _result_row(method, estimate, se, psi0)
            row.update({"ci_lower": lower, "ci_upper": upper, "ci_length": upper - lower})
            row["covered"] = float(lower <= psi0 <= upper)
            rows.append(row)
        elif method == "aipw":
            estimate = aipp_from_prediction(s_l, s_u, y)
            se = influence_se_from_prediction(estimate, s_l, s_u, y)
            rows.append(_result_row(method, estimate, se, psi0))
        elif method == "linear":
            model = fit_linear_calibration(s_l, y)
            pred_l = predict_linear(model, s_l)
            pred_u = predict_linear(model, s_u)
            estimate = plugin_estimate(pred_l, pred_u)
            se = influence_se_from_prediction(estimate, pred_l, pred_u, y)
            rows.append(_result_row(method, estimate, se, psi0))
        elif method == "linear_loo":
            result = loo_linear_aipw(y, s_l, s_u)
            rows.append(
                _result_row(
                    method,
                    result.estimate,
                    result.se,
                    psi0,
                    singular_fallbacks=result.singular_fallbacks,
                )
            )
        elif method == "isotonic":
            model = fit_sklearn_isotonic_calibration(s_l, y)
            pred_l = predict_sklearn_isotonic(model, s_l)
            pred_u = predict_sklearn_isotonic(model, s_u)
            estimate = plugin_estimate(pred_l, pred_u)
            se = influence_se_from_prediction(estimate, pred_l, pred_u, y)
            rows.append(_result_row(method, estimate, se, psi0))
        elif method == "isotonic_loo":
            result = loo_isotonic_aipw(y, s_l, s_u)
            rows.append(
                _result_row(
                    method,
                    result.estimate,
                    result.se,
                    psi0,
                    singular_fallbacks=result.singular_fallbacks,
                )
            )
        elif method == "monotone_spline":
            model = fit_monotone_spline_calibration(s_l, y)
            pred_l = predict_monotone_spline(model, s_l)
            pred_u = predict_monotone_spline(model, s_u)
            estimate = plugin_estimate(pred_l, pred_u)
            se = influence_se_from_prediction(estimate, pred_l, pred_u, y)
            fallback = int(float(model.metadata.get("fallback", 0.0)) > 0.0)
            rows.append(_result_row(method, estimate, se, psi0, singular_fallbacks=fallback))
        elif method == "reppi_rf":
            result = reppi_rf_score_only(y, s_l, s_u, seed=seed)
            rows.append(
                _result_row(
                    method,
                    result.estimate,
                    result.se,
                    psi0,
                    singular_fallbacks=result.singular_fallbacks,
                )
            )
        elif method == "reppi_mlp_cv":
            result = reppi_mlp_cv_score_only(y, s_l, s_u, seed=seed)
            rows.append(
                _result_row(
                    method,
                    result.estimate,
                    result.se,
                    psi0,
                    singular_fallbacks=result.singular_fallbacks,
                    model_fit_failures=result.model_fit_failures,
                    nonfinite_prediction_failures=(
                        result.nonfinite_prediction_failures
                    ),
                    convergence_warnings=result.convergence_warnings,
                    selected_candidates=result.selected_candidates,
                )
            )
    return rows


def _replicate_seed(base_seed: int, quality_index: int, n: int, replicate: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(quality_index), int(n), int(replicate)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _run_replicate(
    payload: Tuple[int, int, int, str, Dict[str, object], Tuple[str, ...]],
) -> List[Dict[str, object]]:
    base_seed, quality_index, n, quality, setting_record, methods = payload
    replicate = int(setting_record.pop("_replicate"))
    setting = score_setting_from_record(setting_record)
    rng_seed = _replicate_seed(base_seed, quality_index, n, replicate)
    sample = generate_sample(
        n=n,
        N=REBUTTAL_UNLABELED_RATIO * n,
        dgp="monotone",
        setting=setting,
        rng=np.random.default_rng(rng_seed),
    )
    rows = run_rebuttal_methods(sample, seed=rng_seed, methods=methods)
    common = {
        "dgp": "monotone",
        "score_quality": quality,
        "score_setting_id": setting.setting_id,
        "score_family": setting.family,
        "n_labeled": int(n),
        "n_unlabeled": int(REBUTTAL_UNLABELED_RATIO * n),
        "unlabeled_ratio": REBUTTAL_UNLABELED_RATIO,
        "replicate": replicate,
        "replicate_seed": rng_seed,
        "psi0": float(sample["psi0"]),
    }
    return [{**common, **row} for row in rows]


def expected_raw_rows(config: RebuttalExperimentConfig) -> int:
    return (
        len(config.score_qualities)
        * len(config.n_grid)
        * config.replications
        * len(config.methods)
    )


RAW_KEY_COLUMNS = ["score_quality", "n_labeled", "replicate", "estimator"]
RAW_FINITE_COLUMNS = [
    "estimate",
    "se",
    "bias",
    "sq_error",
    "covered",
    "ci_length",
    "ci_lower",
    "ci_upper",
    "psi0",
]


def validate_rebuttal_raw(raw: pd.DataFrame, config: RebuttalExperimentConfig) -> None:
    expected = expected_raw_rows(config)
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} raw rows, found {len(raw)}.")
    if raw.duplicated(RAW_KEY_COLUMNS).any():
        raise ValueError("Raw results contain duplicate score/n/replicate/estimator keys.")
    values = raw[RAW_FINITE_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Raw results contain non-finite statistical outputs.")
    counts = raw.groupby(["score_quality", "n_labeled", "estimator"]).size()
    if not np.all(counts.to_numpy() == config.replications):
        raise ValueError("At least one experiment cell has the wrong replication count.")
    if set(raw["fit_status"].unique()) != {"ok"}:
        raise ValueError("At least one fit did not complete successfully.")


def _summary_group(frame: pd.DataFrame) -> pd.Series:
    estimates = frame["estimate"].to_numpy(dtype=float)
    errors = frame["bias"].to_numpy(dtype=float)
    squared_errors = frame["sq_error"].to_numpy(dtype=float)
    coverage = frame["covered"].to_numpy(dtype=float)
    widths = frame["ci_length"].to_numpy(dtype=float)
    replications = len(frame)
    rmse = float(np.sqrt(np.mean(squared_errors)))
    mse_mcse = float(np.std(squared_errors, ddof=1) / np.sqrt(replications))
    return pd.Series(
        {
            "replications": replications,
            "mean_estimate": float(np.mean(estimates)),
            "mean_bias": float(np.mean(errors)),
            "mean_bias_mcse": float(np.std(estimates, ddof=1) / np.sqrt(replications)),
            "rmse": rmse,
            "rmse_mcse": float(mse_mcse / (2.0 * rmse)) if rmse > 0.0 else 0.0,
            "empirical_variance": float(np.var(estimates, ddof=1)),
            "mean_se": float(np.mean(frame["se"])),
            "coverage": float(np.mean(coverage)),
            "coverage_mcse": float(np.sqrt(np.mean(coverage) * (1.0 - np.mean(coverage)) / replications)),
            "mean_ci_length": float(np.mean(widths)),
            "mean_ci_length_mcse": float(np.std(widths, ddof=1) / np.sqrt(replications)),
            "singular_fallbacks": int(np.sum(frame["singular_fallbacks"])),
            "model_fit_failures": int(np.sum(frame["model_fit_failures"])),
            "nonfinite_prediction_failures": int(
                np.sum(frame["nonfinite_prediction_failures"])
            ),
            "convergence_warnings": int(np.sum(frame["convergence_warnings"])),
        }
    )


def _paired_variance_ratio_mcse(
    cell: pd.DataFrame,
    *,
    reference: str,
    estimator: str,
) -> float:
    reference_estimates = (
        cell[cell["estimator"] == reference]
        .sort_values("replicate")["estimate"]
        .to_numpy(dtype=float)
    )
    method_estimates = (
        cell[cell["estimator"] == estimator]
        .sort_values("replicate")["estimate"]
        .to_numpy(dtype=float)
    )
    if len(reference_estimates) == 0 or len(reference_estimates) != len(method_estimates):
        return float("nan")
    reference_sq = (reference_estimates - np.mean(reference_estimates)) ** 2
    method_sq = (method_estimates - np.mean(method_estimates)) ** 2
    reference_variance = float(np.mean(reference_sq))
    method_variance = float(np.mean(method_sq))
    if method_variance <= 0.0:
        return 0.0
    influence = (
        reference_sq / method_variance
        - (reference_variance / method_variance**2) * method_sq
    )
    return float(np.std(influence, ddof=1) / np.sqrt(len(influence)))


def summarise_rebuttal(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dgp",
        "score_quality",
        "score_setting_id",
        "score_family",
        "n_labeled",
        "n_unlabeled",
        "unlabeled_ratio",
        "estimator",
    ]
    summary = (
        raw.groupby(group_columns, as_index=False, sort=True)
        .apply(_summary_group, include_groups=False)
        .reset_index(drop=True)
    )
    label = summary[summary["estimator"] == "labeled_only"][
        ["score_quality", "n_labeled", "empirical_variance"]
    ].rename(columns={"empirical_variance": "labeled_empirical_variance"})
    ppi_plus_plus = summary[summary["estimator"] == "ppi_plus_plus"][
        ["score_quality", "n_labeled", "empirical_variance"]
    ].rename(columns={"empirical_variance": "ppi_plus_plus_empirical_variance"})
    summary = summary.merge(label, on=["score_quality", "n_labeled"], how="left")
    summary = summary.merge(
        ppi_plus_plus,
        on=["score_quality", "n_labeled"],
        how="left",
    )
    summary["relative_efficiency_vs_labeled_mean"] = (
        summary["labeled_empirical_variance"] / summary["empirical_variance"]
    )
    summary["relative_efficiency_vs_ppi_plus_plus"] = (
        summary["ppi_plus_plus_empirical_variance"] / summary["empirical_variance"]
    )

    # Paired delta-method Monte Carlo SEs for the two explicitly named
    # variance-ratio references.
    label_ratio_mcses: List[float] = []
    ppi_plus_plus_ratio_mcses: List[float] = []
    for row in summary.itertuples(index=False):
        cell = raw[
            (raw["score_quality"] == row.score_quality)
            & (raw["n_labeled"] == row.n_labeled)
        ]
        label_ratio_mcses.append(
            _paired_variance_ratio_mcse(
                cell,
                reference="labeled_only",
                estimator=row.estimator,
            )
        )
        ppi_plus_plus_ratio_mcses.append(
            _paired_variance_ratio_mcse(
                cell,
                reference="ppi_plus_plus",
                estimator=row.estimator,
            )
        )
    summary["relative_efficiency_vs_labeled_mean_mcse"] = label_ratio_mcses
    summary["relative_efficiency_vs_ppi_plus_plus_mcse"] = ppi_plus_plus_ratio_mcses
    # Stable aliases consumed by rebuttal placeholder resolution.
    summary["score_regime"] = summary["score_quality"]
    summary["n"] = summary["n_labeled"]
    summary["method"] = summary["estimator"]
    summary["mse"] = summary["rmse"] ** 2
    summary["bias"] = summary["mean_bias"]
    summary["mean_interval_width"] = summary["mean_ci_length"]
    return summary


def _method_styles() -> Dict[str, Dict[str, object]]:
    return {
        "labeled_only": {"label": "Labeled mean", "color": "#202020", "linestyle": ":"},
        "ppi": {"label": "PPI", "color": "#7f7f7f", "linestyle": "-"},
        "ppi_plus_plus": {"label": "PPI++", "color": "#8c4fb0", "linestyle": "-"},
        "aipw": {"label": "Raw pooled AIPW", "color": "#e68613", "linestyle": "-"},
        "linear": {"label": "LinearCal", "color": "#2878b5", "linestyle": "-"},
        "linear_loo": {"label": "LinearCal-LOO", "color": "#2878b5", "linestyle": "--"},
        "isotonic": {"label": "IsoCal", "color": "#3a923a", "linestyle": "-"},
        "isotonic_loo": {"label": "IsoCal-LOO", "color": "#3a923a", "linestyle": "--"},
        "monotone_spline": {"label": "Smooth monotone", "color": "#d1495b", "linestyle": "-"},
        "reppi_rf": {"label": "RePPI-RF", "color": "#4b4ba3", "linestyle": "-."},
        "reppi_mlp_cv": {
            "label": "RePPI-MLP-CV",
            "color": "#008b8b",
            "linestyle": "-.",
        },
    }


def _quality_title(quality: str) -> str:
    return {
        "well_calibrated": "Mildly miscalibrated score",
        "poorly_calibrated": "Strongly miscalibrated score",
    }.get(quality, quality.replace("_", " ").title())


def plot_rebuttal_figures(summary: pd.DataFrame, output_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = _method_styles()
    qualities = [
        quality
        for quality in REBUTTAL_SCORE_QUALITIES
        if quality in set(summary["score_quality"])
    ]
    reppi_supplement_methods = {"reppi_rf", "reppi_mlp_cv"}
    main_methods = [
        method for method in REBUTTAL_METHODS
        if method not in reppi_supplement_methods
    ]

    fig, axes = plt.subplots(1, len(qualities), figsize=(11.2, 4.1), sharey=True, squeeze=False)
    for axis, quality in zip(axes[0], qualities):
        frame = summary[summary["score_quality"] == quality]
        for method in main_methods:
            method_frame = frame[frame["estimator"] == method].sort_values("n_labeled")
            if method_frame.empty:
                continue
            style = styles[method]
            axis.plot(
                method_frame["n_labeled"],
                method_frame["rmse"],
                marker="o",
                markersize=3.2,
                linewidth=1.45,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
            )
        ticks = sorted(int(value) for value in frame["n_labeled"].unique())
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(ticks, labels=[str(value) for value in ticks])
        axis.set_title(_quality_title(quality))
        axis.set_xlabel("Labeled sample size n")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0, 0].set_ylabel("RMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    rmse_path = output_dir / "fig_small_n_rmse.pdf"
    fig.savefig(rmse_path, bbox_inches="tight")
    plt.close(fig)

    reppi_methods = (
        "labeled_only",
        "ppi",
        "ppi_plus_plus",
        "aipw",
        "linear",
        "isotonic",
        "monotone_spline",
        "reppi_rf",
        "reppi_mlp_cv",
    )
    fig, axes = plt.subplots(
        1,
        len(qualities),
        figsize=(10.8, 3.9),
        sharey=True,
        squeeze=False,
    )
    for axis, quality in zip(axes[0], qualities):
        frame = summary[summary["score_quality"] == quality]
        for method in reppi_methods:
            method_frame = frame[frame["estimator"] == method].sort_values("n_labeled")
            if method_frame.empty:
                continue
            style = styles[method]
            axis.plot(
                method_frame["n_labeled"],
                method_frame["rmse"],
                marker="o",
                markersize=3.0,
                linewidth=1.35,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
            )
        ticks = sorted(int(value) for value in frame["n_labeled"].unique())
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(ticks, labels=[str(value) for value in ticks])
        axis.set_title(_quality_title(quality))
        axis.set_xlabel("Labeled sample size n")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0, 0].set_ylabel("RMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    reppi_path = output_dir / "fig_small_n_reppi.pdf"
    fig.savefig(reppi_path, bbox_inches="tight")
    plt.close(fig)

    metrics = (
        ("mean_bias", "Signed bias", 0.0),
        ("coverage", "95% coverage", 0.95),
        ("mean_ci_length", "Mean interval width", None),
        (
            "relative_efficiency_vs_ppi_plus_plus",
            "Relative efficiency vs PPI++",
            1.0,
        ),
    )
    fig, axes = plt.subplots(
        len(qualities),
        len(metrics),
        figsize=(13.2, 6.3),
        sharex="col",
        squeeze=False,
    )
    for row_index, quality in enumerate(qualities):
        frame = summary[summary["score_quality"] == quality]
        for column_index, (metric, title, reference) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for method in REBUTTAL_METHODS:
                method_frame = frame[frame["estimator"] == method].sort_values("n_labeled")
                if method_frame.empty:
                    continue
                style = styles[method]
                axis.plot(
                    method_frame["n_labeled"],
                    method_frame[metric],
                    linewidth=1.1,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                )
            if reference is not None:
                axis.axhline(reference, color="#555555", linewidth=0.65, linestyle=":")
            axis.set_xscale("log")
            ticks = sorted(int(value) for value in frame["n_labeled"].unique())
            axis.set_xticks(ticks, labels=[str(value) for value in ticks])
            axis.grid(alpha=0.2, linewidth=0.55)
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(quality.replace("_", " ").title())
            if row_index == len(qualities) - 1:
                axis.set_xlabel("n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    diagnostics_path = output_dir / "fig_small_n_diagnostics.pdf"
    fig.savefig(diagnostics_path, bbox_inches="tight")
    plt.close(fig)
    return [rmse_path, reppi_path, diagnostics_path]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> Dict[str, str]:
    experiment_dir = Path(__file__).resolve().parent
    paths = {
        "rebuttal_sample_size.py": Path(__file__).resolve(),
        "reppi_baseline_config.json": REPPI_BASELINE_CONFIG_PATH.resolve(),
        "simulate.py": experiment_dir / "simulate.py",
        "estimators.py": experiment_dir / "estimators.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance() -> Dict[str, object]:
    def run_git(*args: str) -> Optional[str]:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def _validate_config(config: RebuttalExperimentConfig) -> None:
    if config.replications < 2:
        raise ValueError("At least two replications are required for Monte Carlo uncertainty.")
    if config.jobs <= 0:
        raise ValueError("jobs must be positive.")
    if not config.n_grid or any(n < 3 for n in config.n_grid):
        raise ValueError("n_grid must contain labeled sample sizes of at least three.")
    if config.unlabeled_ratio != REBUTTAL_UNLABELED_RATIO:
        raise ValueError("The locked rebuttal design requires N = 16n.")
    unknown_qualities = sorted(set(config.score_qualities) - set(REBUTTAL_SCORE_QUALITIES))
    if unknown_qualities:
        raise ValueError("Unknown score qualities: " + ", ".join(unknown_qualities))
    unknown_methods = sorted(set(config.methods) - set(REBUTTAL_METHODS))
    if unknown_methods:
        raise ValueError("Unknown methods: " + ", ".join(unknown_methods))


def build_number_ledger(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a long-form, machine-checkable map from claims to summary rows."""

    metric_columns = (
        ("mean_bias", "mean_bias_mcse"),
        ("rmse", "rmse_mcse"),
        ("coverage", "coverage_mcse"),
        ("mean_ci_length", "mean_ci_length_mcse"),
        (
            "relative_efficiency_vs_labeled_mean",
            "relative_efficiency_vs_labeled_mean_mcse",
        ),
        (
            "relative_efficiency_vs_ppi_plus_plus",
            "relative_efficiency_vs_ppi_plus_plus_mcse",
        ),
    )
    rows: List[Dict[str, object]] = []
    for record in summary.to_dict(orient="records"):
        for metric, mcse_column in metric_columns:
            claim_key = (
                f"{record['score_quality']}.n{int(record['n_labeled'])}."
                f"{record['estimator']}.{metric}"
            )
            rows.append(
                {
                    "claim_key": claim_key,
                    "source_file": "summary.csv",
                    "score_quality": record["score_quality"],
                    "n_labeled": int(record["n_labeled"]),
                    "n_unlabeled": int(record["n_unlabeled"]),
                    "estimator": record["estimator"],
                    "metric": metric,
                    "value": float(record[metric]),
                    "mcse": float(record[mcse_column]),
                    "replications": int(record["replications"]),
                }
            )
    return pd.DataFrame(rows).sort_values("claim_key").reset_index(drop=True)


def _checkpoint_path(
    checkpoint_dir: Path,
    config: RebuttalExperimentConfig,
    quality: str,
    n: int,
) -> Path:
    signature_payload = {
        "seed": config.seed,
        "replications": config.replications,
        "quality": quality,
        "n": n,
        "methods": list(config.methods),
        "ratio": config.unlabeled_ratio,
        "recommendations_sha256": _file_sha256(config.recommendations_file),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "implementation_hashes": _implementation_hashes(),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return checkpoint_dir / f"{quality}_n{n}_{signature}.csv"


def _valid_checkpoint(
    frame: pd.DataFrame,
    config: RebuttalExperimentConfig,
    quality: str,
    n: int,
) -> bool:
    expected = config.replications * len(config.methods)
    if len(frame) != expected or frame.duplicated(RAW_KEY_COLUMNS).any():
        return False
    if set(frame["score_quality"]) != {quality} or set(frame["n_labeled"]) != {n}:
        return False
    if set(frame["estimator"]) != set(config.methods):
        return False
    if set(frame["replicate"]) != set(range(config.replications)):
        return False
    return bool(np.all(np.isfinite(frame[RAW_FINITE_COLUMNS].to_numpy(dtype=float))))


def run_rebuttal_experiment(config: RebuttalExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run, validate, summarize, plot, and manifest the locked design."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with config.recommendations_file.open("r", encoding="utf-8") as handle:
        recommendations = json.load(handle)
    setting_records = {
        quality: dict(recommendations[quality])
        for quality in config.score_qualities
    }
    started = time.time()
    raw_chunks: List[pd.DataFrame] = []
    cell_runtimes: List[Dict[str, object]] = []
    checkpoint_dir = config.output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for quality in config.score_qualities:
        quality_index = REBUTTAL_SCORE_QUALITIES.index(quality)
        for n in config.n_grid:
            cell_started = time.time()
            checkpoint_path = _checkpoint_path(checkpoint_dir, config, quality, int(n))
            if config.resume and checkpoint_path.exists():
                checkpoint = pd.read_csv(checkpoint_path, keep_default_na=False)
                if _valid_checkpoint(checkpoint, config, quality, int(n)):
                    raw_chunks.append(checkpoint)
                    cell_runtimes.append(
                        {
                            "score_quality": quality,
                            "n_labeled": int(n),
                            "runtime_seconds": 0.0,
                            "resumed_from": str(checkpoint_path),
                        }
                    )
                    continue
            payloads = []
            for replicate in range(config.replications):
                record = dict(setting_records[quality])
                record["_replicate"] = replicate
                payloads.append(
                    (
                        config.seed,
                        quality_index,
                        int(n),
                        quality,
                        record,
                        tuple(config.methods),
                    )
                )
            if config.jobs == 1:
                replicate_rows = [_run_replicate(payload) for payload in payloads]
            else:
                # Threads avoid platform semaphore restrictions while the
                # scikit-learn/NumPy kernels release the GIL.  executor.map
                # preserves payload order, and every stochastic fit has an
                # explicit per-replication seed.
                with ThreadPoolExecutor(max_workers=config.jobs) as executor:
                    chunk_size = max(1, config.replications // (config.jobs * 4))
                    replicate_rows = list(
                        executor.map(_run_replicate, payloads, chunksize=chunk_size)
                    )
            rows = [row for replicate in replicate_rows for row in replicate]
            checkpoint = pd.DataFrame(rows)
            if not _valid_checkpoint(checkpoint, config, quality, int(n)):
                raise ValueError(f"Invalid completed checkpoint for {quality}, n={n}.")
            checkpoint.to_csv(checkpoint_path, index=False)
            raw_chunks.append(checkpoint)
            cell_runtimes.append(
                {
                    "score_quality": quality,
                    "n_labeled": int(n),
                    "runtime_seconds": time.time() - cell_started,
                    "checkpoint": str(checkpoint_path),
                }
            )

    raw = pd.concat(raw_chunks, ignore_index=True)
    validate_rebuttal_raw(raw, config)
    summary = summarise_rebuttal(raw)
    raw_path = config.output_dir / "raw_results.csv"
    summary_path = config.output_dir / "summary.csv"
    number_ledger_path = config.output_dir / "number_ledger.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    build_number_ledger(summary).to_csv(number_ledger_path, index=False)
    figure_paths = plot_rebuttal_figures(summary, config.output_dir) if config.make_figures else []

    output_paths = [raw_path, summary_path, number_ledger_path, *figure_paths]
    parity_path = config.output_dir / "official_reppi_parity.json"
    official_parity: Optional[Dict[str, object]] = None
    if parity_path.exists():
        with parity_path.open("r", encoding="utf-8") as stream:
            official_parity = json.load(stream)
        if official_parity.get("status") != "passed":
            raise ValueError("Pinned official RePPI parity validation did not pass.")
        output_paths.append(parity_path)
    selected_candidate_counts: Dict[str, int] = {}
    for encoded in raw.loc[
        raw["estimator"] == "reppi_mlp_cv",
        "selected_candidates",
    ].astype(str):
        for candidate in encoded.split("|"):
            if candidate:
                selected_candidate_counts[candidate] = (
                    selected_candidate_counts.get(candidate, 0) + 1
                )
    manifest = {
        "study": "neurips_2026_rebuttal_sample_size",
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "implementation_hashes": _implementation_hashes(),
        "status": "complete",
        "confirmatory_defaults": {
            "replications": REBUTTAL_REPLICATIONS,
            "seed": REBUTTAL_SEED,
            "n_grid": list(REBUTTAL_N_GRID),
            "unlabeled_ratio": REBUTTAL_UNLABELED_RATIO,
            "methods": list(REBUTTAL_METHODS),
            "score_qualities": list(REBUTTAL_SCORE_QUALITIES),
            "reppi_rf_trees": REPPI_RF_TREES,
            "reppi_folds": REPPI_FOLDS,
            "reppi_mlp_inner_folds": REPPI_MLP_INNER_FOLDS,
            "reppi_mlp_candidates": [
                {
                    "hidden_layer_sizes": list(layers),
                    "alpha": alpha,
                    "name": _mlp_candidate_name((layers, alpha)),
                }
                for layers, alpha in REPPI_MLP_CANDIDATES
            ],
            "reppi_mlp_max_iter": REPPI_MLP_MAX_ITER,
            "reppi_official_commit": REPPI_BASELINE_CONFIG[
                "official_repository"
            ]["commit"],
            "reppi_official_reppi_py_sha256": REPPI_BASELINE_CONFIG[
                "official_repository"
            ]["reppi_py_sha256"],
            "reppi_paper_arxiv_id": REPPI_BASELINE_CONFIG["paper"]["arxiv_id"],
            "reppi_paper_algorithm": REPPI_BASELINE_CONFIG["paper"]["algorithm"],
        },
        "run_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "recommendations_file": str(config.recommendations_file),
            "n_grid": list(config.n_grid),
            "score_qualities": list(config.score_qualities),
            "methods": list(config.methods),
        },
        "expected_raw_rows": expected_raw_rows(config),
        "observed_raw_rows": len(raw),
        "unique_keys": int(raw[RAW_KEY_COLUMNS].drop_duplicates().shape[0]),
        "failed_fits": int(np.sum(raw["fit_status"] != "ok")),
        "singular_fallbacks": {
            str(method): int(value)
            for method, value in raw.groupby("estimator")["singular_fallbacks"].sum().items()
        },
        "model_fit_failures": {
            str(method): int(value)
            for method, value in raw.groupby("estimator")["model_fit_failures"].sum().items()
        },
        "nonfinite_prediction_failures": {
            str(method): int(value)
            for method, value in raw.groupby("estimator")[
                "nonfinite_prediction_failures"
            ].sum().items()
        },
        "convergence_warnings": {
            str(method): int(value)
            for method, value in raw.groupby("estimator")[
                "convergence_warnings"
            ].sum().items()
        },
        "reppi_mlp_selected_candidate_counts": selected_candidate_counts,
        "reppi_baseline_provenance": REPPI_BASELINE_CONFIG,
        "official_code_parity": official_parity,
        "cell_runtimes": cell_runtimes,
        "total_runtime_seconds": time.time() - started,
        "dependencies": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
            "ppi-python": _package_version("ppi-python"),
            "matplotlib": _package_version("matplotlib"),
            "xgboost": _package_version("xgboost"),
            "joblib": _package_version("joblib"),
            "statsmodels": _package_version("statsmodels"),
            "numba": _package_version("numba"),
        },
        "git": _git_provenance(),
        "input_hashes": {
            str(config.recommendations_file): _file_sha256(config.recommendations_file),
            str(REPPI_BASELINE_CONFIG_PATH): _file_sha256(
                REPPI_BASELINE_CONFIG_PATH
            ),
        },
        "output_hashes": {
            path.name: _file_sha256(path)
            for path in output_paths
        },
        "mc_uncertainty_columns": [
            "mean_bias_mcse",
            "rmse_mcse",
            "coverage_mcse",
            "mean_ci_length_mcse",
            "relative_efficiency_vs_labeled_mean_mcse",
            "relative_efficiency_vs_ppi_plus_plus_mcse",
        ],
    }
    with (config.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return raw, summary
