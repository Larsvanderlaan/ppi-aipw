from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families import Gaussian
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppi_aipw import mean_inference

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised only when xgboost is unavailable
    xgb = None


def clip_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)


def clip_range(x: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), float(y_min), float(y_max))


@dataclass
class TuningGrid:
    min_data_in_leaf: Iterable[int]
    num_leaves: Iterable[int]
    max_depth: Iterable[int]


@dataclass
class CalibrationModel:
    name: str
    fitted: object
    metadata: Dict[str, float]


_AUTO_AIPW_EFFICIENCY_LABEL = "aipw_efficiency_maximization"
_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS = 6
_MONOTONE_SPLINE_DERIVATIVE_DEGREE = 2
_MONOTONE_SPLINE_PENALTY = 1e-3


@dataclass(frozen=True)
class AutoCandidateSpec:
    method: str
    label: str
    efficiency_maximization: bool


@dataclass
class XGBoostMonotoneRegressor:
    booster: Any

    def predict(self, score: np.ndarray) -> np.ndarray:
        if xgb is None:  # pragma: no cover - guarded during fitting as well
            raise ImportError(
                "xgboost is required for the vendored auto-calibration baseline. "
                "Install xgboost before running these experiments."
            )
        score = np.asarray(score, dtype=float).reshape(-1, 1)
        return np.asarray(self.booster.predict(xgb.DMatrix(data=score)), dtype=float)


def canonical_auto_method(method: str) -> str:
    aliases = {
        "aipw": "aipw",
        "identity": "aipw",
        "none": "aipw",
        "linear": "linear",
        "linear_calibration": "linear",
        "monotone_spline": "monotone_spline",
        "isotonic_spline": "monotone_spline",
        "smooth_spline": "monotone_spline",
        "mspline": "monotone_spline",
        "isocal": "isocal",
        "isotonic": "isocal",
        "isotonic_calibration": "isocal",
    }
    try:
        return aliases[method.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown auto-calibration method '{method}'. Expected one of: {choices}.") from exc


def fit_xgboost_isocal_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    max_depth: int = 20,
    min_child_weight: float = 10.0,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))

    if np.allclose(score_l, score_l[0]) or np.isclose(y_max, y_min):
        mean_y = float(np.mean(y_l))
        return CalibrationModel(
            name="xgboost_isocal_fallback",
            fitted={"a": 0.0, "b": mean_y},
            metadata={
                "fallback": 1.0,
                "y_min": y_min,
                "y_max": y_max,
                "max_depth": float(max_depth),
                "min_child_weight": float(min_child_weight),
            },
        )

    if xgb is None:
        raise ImportError(
            "xgboost is required for the vendored auto-calibration baseline. "
            "Install xgboost before running these experiments."
        )

    data = xgb.DMatrix(
        data=score_l.reshape(-1, 1),
        label=y_l.reshape(-1),
    )
    params = {
        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "monotone_constraints": "(1)",
        "eta": 1.0,
        "gamma": 0.0,
        "lambda": 0.0,
        "objective": "reg:squarederror",
        "verbosity": 0,
    }
    model = XGBoostMonotoneRegressor(
        booster=xgb.train(params=params, dtrain=data, num_boost_round=1)
    )
    return CalibrationModel(
        name="xgboost_isocal",
        fitted=model,
        metadata={
            "y_min": y_min,
            "y_max": y_max,
            "max_depth": float(max_depth),
            "min_child_weight": float(min_child_weight),
        },
    )


def predict_xgboost_isocal(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    if isinstance(model.fitted, dict):
        pred = model.fitted["a"] * np.asarray(score, dtype=float) + model.fitted["b"]
    else:
        pred = model.fitted.predict(score)
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(pred, y_min, y_max)


def _choose_monotone_spline_knots(
    scores_scaled: np.ndarray,
    *,
    max_internal_knots: int,
    degree: int,
) -> np.ndarray:
    unique_scores = np.unique(scores_scaled)
    max_allowed = max(0, unique_scores.size - degree - 1)
    n_internal = min(max_internal_knots, max_allowed)
    if n_internal <= 0:
        internal = np.array([], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
        internal = np.quantile(scores_scaled, quantiles)
        internal = np.unique(np.asarray(internal, dtype=float))
        internal = internal[(internal > 1e-8) & (internal < 1.0 - 1e-8)]
    return np.concatenate(
        [
            np.repeat(0.0, degree + 1),
            internal,
            np.repeat(1.0, degree + 1),
        ]
    )


def _integrated_bspline_design(
    scores_scaled: np.ndarray,
    *,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    scores_scaled = np.clip(np.asarray(scores_scaled, dtype=float), 0.0, 1.0)
    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid spline specification: expected at least one basis function.")

    columns = []
    for idx in range(n_basis):
        coef = np.zeros(n_basis, dtype=float)
        coef[idx] = 1.0
        derivative_basis = BSpline(knots, coef, degree, extrapolate=False)
        integral_basis = derivative_basis.antiderivative()
        baseline = float(np.asarray(integral_basis(0.0), dtype=float))
        column = np.asarray(integral_basis(scores_scaled), dtype=float) - baseline
        columns.append(column)
    return np.column_stack(columns)


def _evaluate_monotone_spline(
    scores_scaled: np.ndarray,
    *,
    knots: np.ndarray,
    coef: np.ndarray,
    degree: int,
    intercept: float,
) -> np.ndarray:
    scores_scaled = np.clip(np.asarray(scores_scaled, dtype=float), 0.0, 1.0)
    derivative_spline = BSpline(knots, np.asarray(coef, dtype=float), degree, extrapolate=False)
    integrated_spline = derivative_spline.antiderivative()
    baseline = float(np.asarray(integrated_spline(0.0), dtype=float))
    return float(intercept) + np.asarray(integrated_spline(scores_scaled), dtype=float) - baseline


def _second_difference_penalty(n_basis: int) -> np.ndarray:
    if n_basis < 3:
        return np.eye(n_basis, dtype=float)
    penalty = np.zeros((n_basis - 2, n_basis), dtype=float)
    for idx in range(n_basis - 2):
        penalty[idx, idx : idx + 3] = np.array([1.0, -2.0, 1.0], dtype=float)
    return penalty


def fit_monotone_spline_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))

    if np.allclose(score_l, score_l[0]) or np.isclose(y_max, y_min):
        mean_y = float(np.mean(y_l))
        return CalibrationModel(
            name="monotone_spline_fallback",
            fitted={"slope": 0.0, "intercept": mean_y},
            metadata={"fallback": 1.0, "y_min": y_min, "y_max": y_max},
        )

    score_min = float(np.min(score_l))
    score_max = float(np.max(score_l))
    score_scale = float(score_max - score_min)
    if np.isclose(score_scale, 0.0) or np.unique(score_l).size < 4:
        return fit_linear_calibration(score_l, y_l)

    scores_scaled = np.clip((score_l - score_min) / score_scale, 0.0, 1.0)
    basis_degree = _MONOTONE_SPLINE_DERIVATIVE_DEGREE
    knots = _choose_monotone_spline_knots(
        scores_scaled,
        max_internal_knots=_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS,
        degree=basis_degree,
    )
    basis = _integrated_bspline_design(scores_scaled, knots=knots, degree=basis_degree)
    design = np.column_stack([np.ones_like(scores_scaled), basis])
    penalty_block = _second_difference_penalty(basis.shape[1])
    penalty_design = np.column_stack(
        [
            np.zeros((penalty_block.shape[0], 1), dtype=float),
            penalty_block,
        ]
    )
    augmented_design = np.vstack(
        [
            design,
            np.sqrt(_MONOTONE_SPLINE_PENALTY) * penalty_design,
        ]
    )
    augmented_response = np.concatenate(
        [
            y_l,
            np.zeros(penalty_design.shape[0], dtype=float),
        ]
    )

    lower = np.concatenate([[-np.inf], np.zeros(basis.shape[1], dtype=float)])
    upper = np.full(lower.shape, np.inf, dtype=float)
    result = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return fit_linear_calibration(score_l, y_l)

    return CalibrationModel(
        name="monotone_spline",
        fitted={
            "intercept": float(result.x[0]),
            "coef": np.asarray(result.x[1:], dtype=float),
            "knots": np.asarray(knots, dtype=float),
            "basis_degree": float(basis_degree),
            "score_min": score_min,
            "score_scale": score_scale,
        },
        metadata={
            "max_internal_knots": float(_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS),
            "basis_degree": float(basis_degree + 1),
            "penalty": float(_MONOTONE_SPLINE_PENALTY),
            "y_min": y_min,
            "y_max": y_max,
        },
    )


def predict_monotone_spline(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if "slope" in getattr(model, "fitted", {}):
        pred = model.fitted["slope"] * score + model.fitted["intercept"]
    elif "a" in getattr(model, "fitted", {}):
        pred = model.fitted["a"] * score + model.fitted["b"]
    else:
        score_min = float(model.fitted["score_min"])
        score_scale = float(model.fitted["score_scale"])
        if np.isclose(score_scale, 0.0):
            pred = np.full_like(score, float(model.fitted["intercept"]), dtype=float)
        else:
            z = np.clip((score - score_min) / score_scale, 0.0, 1.0)
            pred = _evaluate_monotone_spline(
                z,
                knots=np.asarray(model.fitted["knots"], dtype=float),
                coef=np.asarray(model.fitted["coef"], dtype=float),
                degree=int(model.fitted["basis_degree"]),
                intercept=float(model.fitted["intercept"]),
            )
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(pred, y_min, y_max)


def _estimate_efficiency_lambda_unweighted(
    y_l: np.ndarray,
    pred_l: np.ndarray,
    pred_u: np.ndarray,
) -> float:
    n = len(y_l)
    N = len(pred_u)
    rho = n / float(n + N)
    labeled_score = (1.0 - rho) * np.asarray(pred_l, dtype=float)
    unlabeled_score = (1.0 - rho) * np.asarray(pred_u, dtype=float)
    centered_outcome = np.asarray(y_l, dtype=float) - float(np.mean(y_l))
    centered_labeled_score = labeled_score - float(np.mean(labeled_score))
    numerator = float(np.mean(centered_outcome * centered_labeled_score))
    denominator = float(np.var(labeled_score) + (n / float(N)) * np.var(unlabeled_score))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _apply_efficiency_scaling_unweighted(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_unlabeled: np.ndarray,
    predictions_to_scale: tuple[np.ndarray, ...],
) -> tuple[float, list[np.ndarray]]:
    lambda_hat = _estimate_efficiency_lambda_unweighted(y_train, pred_train, pred_unlabeled)
    return lambda_hat, [np.asarray(pred, dtype=float) * lambda_hat for pred in predictions_to_scale]


def _resolve_auto_unlabeled_subsample_size(
    n_labeled: int,
    n_unlabeled: int,
    auto_unlabeled_subsample_size: int | None,
) -> int:
    if auto_unlabeled_subsample_size is None:
        requested_size = 10 * n_labeled
    else:
        requested_size = int(auto_unlabeled_subsample_size)
        if requested_size < 1:
            raise ValueError("auto_unlabeled_subsample_size must be at least 1 when provided.")
    return min(n_unlabeled, requested_size)


def _subset_unlabeled_for_auto(
    score_u: np.ndarray,
    *,
    n_labeled: int,
    auto_unlabeled_subsample_size: int | None,
    subset_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_u = np.asarray(score_u, dtype=float)
    subset_size = _resolve_auto_unlabeled_subsample_size(
        n_labeled,
        len(score_u),
        auto_unlabeled_subsample_size,
    )
    if subset_size >= len(score_u):
        return score_u, {
            "auto_unlabeled_subsample_size": len(score_u),
            "auto_unlabeled_subsample_default": auto_unlabeled_subsample_size is None,
            "unlabeled_strategy": "all_unlabeled_rows_in_each_fold",
        }

    rng = np.random.default_rng(subset_seed)
    subset_idx = np.sort(rng.choice(len(score_u), size=subset_size, replace=False))
    return score_u[subset_idx], {
        "auto_unlabeled_subsample_size": subset_size,
        "auto_unlabeled_subsample_default": auto_unlabeled_subsample_size is None,
        "auto_unlabeled_subsample_seed": subset_seed,
        "unlabeled_strategy": "subsampled_unlabeled_rows_in_each_fold",
    }


def _auto_candidate_specs(candidate_methods: tuple[str, ...]) -> tuple[list[str], list[AutoCandidateSpec]]:
    canonical_candidates: list[str] = []
    for method_name in candidate_methods:
        canonical = canonical_auto_method(method_name)
        if canonical not in canonical_candidates:
            canonical_candidates.append(canonical)
    if not canonical_candidates:
        raise ValueError("candidate_methods must contain at least one valid method.")

    specs = [
        AutoCandidateSpec(method=method_name, label=method_name, efficiency_maximization=False)
        for method_name in canonical_candidates
    ]
    if "aipw" in canonical_candidates:
        specs.append(
            AutoCandidateSpec(
                method="aipw",
                label=_AUTO_AIPW_EFFICIENCY_LABEL,
                efficiency_maximization=True,
            )
        )
    return canonical_candidates, specs


def _predict_auto_candidate(
    method_name: str,
    y_train: np.ndarray,
    score_train: np.ndarray,
    score_targets: tuple[np.ndarray, ...],
    *,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
) -> list[np.ndarray]:
    y_train = np.asarray(y_train, dtype=float)
    score_train = np.asarray(score_train, dtype=float)
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))

    if method_name == "aipw":
        return [np.asarray(score_target, dtype=float) for score_target in score_targets]
    if method_name == "linear":
        model = fit_linear_calibration(score_train, y_train)
        return [
            clip_range(predict_linear(model, score_target), y_min, y_max)
            for score_target in score_targets
        ]
    if method_name == "monotone_spline":
        model = fit_monotone_spline_calibration(score_train, y_train)
        return [
            clip_range(predict_monotone_spline(model, score_target), y_min, y_max)
            for score_target in score_targets
        ]
    if method_name == "isocal":
        model = fit_xgboost_isocal_calibration(
            score_train,
            y_train,
            max_depth=isocal_max_depth,
            min_child_weight=isocal_min_child_weight,
        )
        return [
            clip_range(predict_xgboost_isocal(model, score_target), y_min, y_max)
            for score_target in score_targets
        ]
    raise ValueError(f"Unsupported auto-calibration candidate '{method_name}'.")


def _candidate_cv_predictions(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    method_name: str,
    splitter: KFold,
    efficiency_maximization: bool,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_oof = np.zeros_like(np.asarray(y_l, dtype=float), dtype=float)
    pred_unlabeled_folds = []

    for train_idx, val_idx in splitter.split(np.asarray(y_l, dtype=float)):
        y_train = np.asarray(y_l, dtype=float)[train_idx]
        score_train = np.asarray(score_l, dtype=float)[train_idx]
        score_val = np.asarray(score_l, dtype=float)[val_idx]

        pred_train, pred_val, pred_unlabeled = _predict_auto_candidate(
            method_name,
            y_train,
            score_train,
            (score_train, score_val, score_u),
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )

        if efficiency_maximization:
            _, scaled_predictions = _apply_efficiency_scaling_unweighted(
                y_train,
                pred_train,
                pred_unlabeled,
                (pred_val, pred_unlabeled),
            )
            pred_val, pred_unlabeled = scaled_predictions

        pred_oof[val_idx] = pred_val
        pred_unlabeled_folds.append(pred_unlabeled)

    mean_pred_unlabeled = np.mean(np.stack(pred_unlabeled_folds, axis=0), axis=0)
    return pred_oof, mean_pred_unlabeled


def _cv_selection_score(
    y_l: np.ndarray,
    pred_l: np.ndarray,
    pred_u: np.ndarray,
) -> float:
    n = len(y_l)
    N = len(pred_u)
    rho = n / float(n + N)
    labeled_component = np.asarray(y_l, dtype=float) - (1.0 - rho) * np.asarray(pred_l, dtype=float)
    unlabeled_component = (1.0 - rho) * np.asarray(pred_u, dtype=float)
    return float(np.var(labeled_component) / n + np.var(unlabeled_component) / N)


def select_auto_aipw_method_cv(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "monotone_spline", "isocal"),
    num_folds: int = 20,
    random_state: int = 0,
    auto_unlabeled_subsample_size: int | None = None,
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> tuple[str, dict[str, Any], np.ndarray, np.ndarray]:
    y_l = np.asarray(y_l, dtype=float)
    score_l = np.asarray(score_l, dtype=float)
    score_u = np.asarray(score_u, dtype=float)
    if y_l.ndim != 1 or score_l.ndim != 1 or score_u.ndim != 1:
        raise ValueError("The vendored auto baseline expects one-dimensional labeled and unlabeled predictions.")
    if len(y_l) < 2:
        raise ValueError("Need at least two labeled observations for the auto baseline.")

    canonical_candidates, candidate_specs = _auto_candidate_specs(candidate_methods)
    n_splits = min(int(num_folds), len(y_l))
    if n_splits < 2:
        raise ValueError("num_folds is too large relative to the labeled sample size.")

    score_u_selection, unlabeled_metadata = _subset_unlabeled_for_auto(
        score_u,
        n_labeled=len(y_l),
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        subset_seed=int(random_state),
    )
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    base_cv_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_name in canonical_candidates:
        base_cv_predictions[method_name] = _candidate_cv_predictions(
            y_l,
            score_l,
            score_u_selection,
            method_name=method_name,
            splitter=splitter,
            efficiency_maximization=False,
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )

    scores: dict[str, float] = {}
    for candidate_spec in candidate_specs:
        if candidate_spec.efficiency_maximization:
            pred_l_cf, pred_u_cf = _candidate_cv_predictions(
                y_l,
                score_l,
                score_u_selection,
                method_name=candidate_spec.method,
                splitter=splitter,
                efficiency_maximization=True,
                isocal_max_depth=isocal_max_depth,
                isocal_min_child_weight=isocal_min_child_weight,
            )
        else:
            pred_l_cf, pred_u_cf = base_cv_predictions[candidate_spec.method]
        scores[candidate_spec.label] = _cv_selection_score(y_l, pred_l_cf, pred_u_cf)

    selected_label = min(scores, key=scores.get)
    selected_candidate = next(spec for spec in candidate_specs if spec.label == selected_label)
    selected_pred_labeled, selected_pred_unlabeled = base_cv_predictions[selected_candidate.method]
    diagnostics = {
        "selected_method": selected_candidate.method,
        "selected_candidate": selected_label,
        "selected_efficiency_maximization": selected_candidate.efficiency_maximization,
        "num_folds": n_splits,
        **unlabeled_metadata,
        "scores": scores,
    }
    return selected_candidate.method, diagnostics, selected_pred_labeled, selected_pred_unlabeled


def auto_aipw_pointestimate_and_se(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    *,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "monotone_spline", "isocal"),
    num_folds: int = 20,
    random_state: int = 0,
    auto_unlabeled_subsample_size: int | None = None,
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> Dict[str, Any]:
    selected_method, diagnostics, pred_labeled_cf, pred_unlabeled_cf = select_auto_aipw_method_cv(
        y_l,
        score_l,
        score_u,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        random_state=random_state,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )

    pred_labeled_point, pred_unlabeled_point = _predict_auto_candidate(
        selected_method,
        np.asarray(y_l, dtype=float),
        np.asarray(score_l, dtype=float),
        (score_l, score_u),
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    pred_labeled_variance = pred_labeled_cf
    pred_unlabeled_variance = pred_unlabeled_cf

    if bool(diagnostics["selected_efficiency_maximization"]):
        lambda_hat = _estimate_efficiency_lambda_unweighted(y_l, pred_labeled_cf, pred_unlabeled_cf)
        pred_labeled_point = pred_labeled_point * lambda_hat
        pred_unlabeled_point = pred_unlabeled_point * lambda_hat
        pred_labeled_variance = pred_labeled_variance * lambda_hat
        pred_unlabeled_variance = pred_unlabeled_variance * lambda_hat
        diagnostics = dict(diagnostics)
        diagnostics["efficiency_lambda"] = float(lambda_hat)

    estimate = float(aipp_from_prediction(pred_labeled_point, pred_unlabeled_point, np.asarray(y_l, dtype=float)))
    se = float(
        influence_se_from_prediction(
            estimate,
            pred_labeled_variance,
            pred_unlabeled_variance,
            np.asarray(y_l, dtype=float),
        )
    )
    return {
        "estimate": estimate,
        "se": se,
        **diagnostics,
    }


def fit_linear_calibration(score_l: np.ndarray, y_l: np.ndarray) -> CalibrationModel:
    design = np.column_stack([score_l, np.ones_like(score_l)])
    coef, _, _, _ = np.linalg.lstsq(design, y_l, rcond=None)
    model = {"a": float(coef[0]), "b": float(coef[1])}
    return CalibrationModel(name="linear", fitted=model, metadata=model)


def predict_linear(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    return model.fitted["a"] * score + model.fitted["b"]


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = clip_unit(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_platt_calibration(score_l: np.ndarray, y_l: np.ndarray) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))
    if np.isclose(y_max, y_min):
        metadata = {"a": 0.0, "b": 0.0, "y_min": y_min, "y_max": y_max}
        return CalibrationModel(name="platt", fitted=metadata, metadata=metadata)
    if not np.all(np.isin(np.unique(y_l), [y_min, y_max])) or not np.isclose(y_min, 0.0) or not np.isclose(y_max, 1.0):
        raise ValueError("Platt calibration is currently implemented only for binary outcomes in {0,1}.")
    if np.any(score_l <= 0.0) or np.any(score_l >= 1.0):
        raise ValueError("Platt calibration requires scores strictly in (0,1).")
    logit_score = np.log(score_l / (1.0 - score_l))
    mean_y = float(np.clip(np.mean(y_l), 1e-8, 1.0 - 1e-8))
    if np.allclose(logit_score, logit_score[0]):
        metadata = {
            "a": 0.0,
            "b": float(_safe_logit(np.array([mean_y], dtype=float))[0]),
            "y_min": y_min,
            "y_max": y_max,
        }
        return CalibrationModel(name="platt", fitted=metadata, metadata=metadata)

    start_intercept = float(_safe_logit(np.array([mean_y], dtype=float))[0])
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )
        model.fit(logit_score.reshape(-1, 1), y_l.astype(int))
        beta = np.array(
            [
                float(model.coef_.reshape(-1)[0]),
                float(model.intercept_.reshape(-1)[0]),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(beta)):
            beta = np.array([1.0, start_intercept], dtype=float)
    except Exception:
        beta = np.array([1.0, start_intercept], dtype=float)
    metadata = {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "y_min": y_min,
        "y_max": y_max,
    }
    return CalibrationModel(name="platt", fitted=metadata, metadata=metadata)


def predict_platt(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if np.any(score <= 0.0) or np.any(score >= 1.0):
        raise ValueError("Platt calibration requires scores strictly in (0,1).")
    logit_score = np.log(score / (1.0 - score))
    pred_scaled = clip_unit(_sigmoid(model.fitted["a"] * logit_score + model.fitted["b"]))
    y_min = float(model.metadata.get("y_min", 0.0))
    y_max = float(model.metadata.get("y_max", 1.0))
    return clip_range(y_min + (y_max - y_min) * pred_scaled, y_min, y_max)


def fit_sklearn_isotonic_calibration(score_l: np.ndarray, y_l: np.ndarray) -> CalibrationModel:
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))
    model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True, out_of_bounds="clip")
    model.fit(score_l, y_l)
    return CalibrationModel(
        name="sklearn_isotonic",
        fitted=model,
        metadata={"y_min": y_min, "y_max": y_max},
    )


def predict_sklearn_isotonic(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(model.fitted.predict(score), y_min, y_max)


def _compress_sorted_bins(
    score_l: np.ndarray,
    y_l: np.ndarray,
    min_bin_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    order = np.argsort(score_l)
    score_sorted = score_l[order]
    y_sorted = y_l[order]
    n = len(score_sorted)
    if min_bin_size <= 1 or n <= min_bin_size:
        return score_sorted, y_sorted, np.ones(n, dtype=float)

    bins = []
    start = 0
    while start < n:
        end = min(start + min_bin_size, n)
        bins.append((start, end))
        start = end
    if len(bins) >= 2:
        last_start, last_end = bins[-1]
        if last_end - last_start < min_bin_size:
            prev_start, _ = bins[-2]
            bins[-2] = (prev_start, last_end)
            bins.pop()

    score_bin = np.array([score_sorted[s:e].mean() for s, e in bins], dtype=float)
    y_bin = np.array([y_sorted[s:e].mean() for s, e in bins], dtype=float)
    weight_bin = np.array([e - s for s, e in bins], dtype=float)
    return score_bin, y_bin, weight_bin


def fit_binned_isotonic_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    min_bin_size: int = 15,
) -> CalibrationModel:
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))
    score_bin, y_bin, weight_bin = _compress_sorted_bins(score_l, y_l, min_bin_size=min_bin_size)
    model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True, out_of_bounds="clip")
    model.fit(score_bin, y_bin, sample_weight=weight_bin)
    return CalibrationModel(
        name="binned_isotonic",
        fitted=model,
        metadata={
            "y_min": y_min,
            "y_max": y_max,
            "min_bin_size": float(min_bin_size),
        },
    )


def predict_binned_isotonic(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(model.fitted.predict(score), y_min, y_max)


def fit_tuned_binned_isotonic_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    min_bin_size_grid: Iterable[int] = (5, 10, 15, 20, 30),
    cv_folds: int = 5,
    random_state: int = 0,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    candidates = tuple(sorted({int(value) for value in min_bin_size_grid if int(value) >= 1}))
    if len(candidates) == 0:
        raise ValueError("min_bin_size_grid must contain at least one positive integer.")

    best_min_bin_size = candidates[0]
    best_score = np.inf
    effective_folds = min(int(cv_folds), len(score_l))
    if effective_folds >= 2:
        splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        for min_bin_size in candidates:
            fold_scores = []
            for train_idx, val_idx in splitter.split(score_l):
                model = fit_binned_isotonic_calibration(
                    score_l[train_idx],
                    y_l[train_idx],
                    min_bin_size=min_bin_size,
                )
                pred_val = predict_binned_isotonic(model, score_l[val_idx])
                fold_scores.append(float(np.mean((y_l[val_idx] - pred_val) ** 2)))
            cv_score = float(np.mean(fold_scores))
            if cv_score < best_score:
                best_score = cv_score
                best_min_bin_size = min_bin_size

    fitted = fit_binned_isotonic_calibration(score_l, y_l, min_bin_size=best_min_bin_size)
    metadata = dict(fitted.metadata)
    metadata.update(
        {
            "selected_min_bin_size": float(best_min_bin_size),
            "cv_mse": best_score if np.isfinite(best_score) else np.nan,
        }
    )
    return CalibrationModel(
        name="tuned_binned_isotonic",
        fitted=fitted.fitted,
        metadata=metadata,
    )


def fit_venn_abers_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    round_digits: int = 2,
) -> CalibrationModel:
    y_l = np.asarray(y_l, dtype=float)
    unique = np.unique(y_l)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError("Venn-Abers calibration currently requires binary outcomes in {0, 1}.")
    order = np.argsort(score_l)
    fitted = {
        "score_l": np.asarray(score_l, dtype=float)[order],
        "y_l": y_l[order],
        "round_digits": int(round_digits),
        "cache": {},
    }
    return CalibrationModel(
        name="venn_abers",
        fitted=fitted,
        metadata={"round_digits": int(round_digits)},
    )


def _fit_isotonic_with_extra(
    score_l: np.ndarray,
    y_l: np.ndarray,
    score_value: float,
    extra_label: float,
) -> float:
    x = np.append(score_l, score_value)
    y = np.append(y_l, extra_label)
    order = np.argsort(x)
    model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, increasing=True, out_of_bounds="clip")
    model.fit(x[order], y[order])
    return float(clip_unit(model.predict(np.array([score_value], dtype=float)))[0])


def predict_venn_abers(
    model: CalibrationModel,
    score: np.ndarray,
    reference_pred: np.ndarray,
) -> np.ndarray:
    train_score = model.fitted["score_l"]
    train_y = model.fitted["y_l"]
    round_digits = model.fitted["round_digits"]
    cache = model.fitted["cache"]

    rounded_score = np.round(np.asarray(score, dtype=float), round_digits)
    reference_pred = clip_unit(np.asarray(reference_pred, dtype=float))
    pred = np.empty_like(rounded_score, dtype=float)
    for value in np.unique(rounded_score):
        key = float(value)
        if key not in cache:
            p0 = _fit_isotonic_with_extra(train_score, train_y, key, 0.0)
            p1 = _fit_isotonic_with_extra(train_score, train_y, key, 1.0)
            cache[key] = (p0, p1)
        p0, p1 = cache[key]
        midpoint = 0.5 * (p0 + p1)
        mask = rounded_score == value
        pred[mask] = midpoint + (p1 - p0) * (reference_pred[mask] - midpoint)
    return clip_unit(pred)


def fit_tuned_venn_abers_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    round_digits_grid: Iterable[int] = (1, 2, 3),
    cv_folds: int = 5,
    random_state: int = 0,
    min_bin_size: int = 15,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    unique = np.unique(y_l)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError("Tuned Venn-Abers calibration currently requires binary outcomes in {0, 1}.")

    round_digits_grid = tuple(int(value) for value in round_digits_grid)
    if len(round_digits_grid) == 0:
        raise ValueError("round_digits_grid must contain at least one candidate.")

    n = len(score_l)
    effective_folds = min(int(cv_folds), n)
    best_round_digits = round_digits_grid[0]
    best_score = np.inf

    if effective_folds >= 2:
        splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        for round_digits in round_digits_grid:
            fold_scores = []
            for train_idx, val_idx in splitter.split(score_l):
                reference_model = fit_binned_isotonic_calibration(
                    score_l[train_idx],
                    y_l[train_idx],
                    min_bin_size=min_bin_size,
                )
                reference_val = predict_binned_isotonic(reference_model, score_l[val_idx])
                va_model = fit_venn_abers_calibration(
                    score_l[train_idx],
                    y_l[train_idx],
                    round_digits=round_digits,
                )
                pred_val = predict_venn_abers(va_model, score_l[val_idx], reference_val)
                fold_scores.append(float(np.mean((y_l[val_idx] - pred_val) ** 2)))
            cv_score = float(np.mean(fold_scores))
            if cv_score < best_score:
                best_score = cv_score
                best_round_digits = round_digits

    reference_model = fit_binned_isotonic_calibration(score_l, y_l, min_bin_size=min_bin_size)
    va_model = fit_venn_abers_calibration(score_l, y_l, round_digits=best_round_digits)
    return CalibrationModel(
        name="tuned_venn_abers",
        fitted={
            "reference_model": reference_model,
            "venn_abers_model": va_model,
        },
        metadata={
            "round_digits": float(best_round_digits),
            "cv_mse": best_score if np.isfinite(best_score) else np.nan,
            "min_bin_size": float(min_bin_size),
            "y_min": float(np.min(y_l)),
            "y_max": float(np.max(y_l)),
        },
    )


def predict_tuned_venn_abers(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    reference_model = model.fitted["reference_model"]
    va_model = model.fitted["venn_abers_model"]
    reference_pred = predict_binned_isotonic(reference_model, score)
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(predict_venn_abers(va_model, score, reference_pred), y_min, y_max)


def fit_venn_abers_shrinkage_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    round_digits: int = 2,
    reference_model: Optional[CalibrationModel] = None,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    if reference_model is None:
        reference_model = fit_binned_isotonic_calibration(score_l, y_l, min_bin_size=15)
    va_model = fit_venn_abers_calibration(score_l, y_l, round_digits=round_digits)
    return CalibrationModel(
        name="venn_abers_shrinkage",
        fitted={
            "reference_model": reference_model,
            "venn_abers_model": va_model,
        },
        metadata={
            "round_digits": float(round_digits),
            "y_min": float(np.min(y_l)),
            "y_max": float(np.max(y_l)),
        },
    )


def predict_venn_abers_shrinkage(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    reference_model = model.fitted["reference_model"]
    va_model = model.fitted["venn_abers_model"]
    reference_pred = predict_binned_isotonic(reference_model, score)
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(predict_venn_abers(va_model, score, reference_pred), y_min, y_max)


def fit_gam_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    spline_df: Optional[int] = None,
    spline_degree: int = 3,
) -> CalibrationModel:
    score_l = np.asarray(score_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    n = len(score_l)
    if n <= spline_degree:
        raise ValueError("Need more observations than spline degree for GAM calibration.")

    if spline_df is None:
        spline_df = max(spline_degree + 1, min(12, max(6, n // 25)))
    spline_df = int(min(max(spline_df, spline_degree + 1), max(spline_degree + 1, n - 1)))

    smooth_x = score_l.reshape(-1, 1)
    smoother = BSplines(smooth_x, df=[spline_df], degree=[spline_degree])
    exog = np.ones((n, 1), dtype=float)
    model = GLMGam(y_l, smoother=smoother, exog=exog, family=Gaussian())

    initial_result = model.fit()
    alpha = np.ones(model.k_smooths, dtype=float)
    try:
        tuned_alpha, _, _ = model.select_penweight(start_model_params=np.asarray(initial_result.params))
        alpha = np.asarray(tuned_alpha, dtype=float)
        model.alpha = alpha
        result = model.fit(start_params=np.asarray(initial_result.params))
    except Exception:
        result = initial_result

    return CalibrationModel(
        name="gam_spline",
        fitted=result,
        metadata={
            "alpha": float(np.asarray(alpha).reshape(-1)[0]),
            "spline_df": float(spline_df),
            "spline_degree": float(spline_degree),
            "score_min": float(np.min(score_l)),
            "score_max": float(np.max(score_l)),
            "y_min": float(np.min(y_l)),
            "y_max": float(np.max(y_l)),
        },
    )


def predict_gam(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    score_min = model.metadata.get("score_min", -np.inf)
    score_max = model.metadata.get("score_max", np.inf)
    score = np.clip(np.asarray(score, dtype=float), score_min, score_max).reshape(-1, 1)
    exog = np.ones((score.shape[0], 1), dtype=float)
    pred = model.fitted.predict(exog=exog, exog_smooth=score)
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(pred, y_min, y_max)


def _build_lgbm_model(
    min_data_in_leaf: int,
    num_leaves: int,
    max_depth: int,
    random_state: int,
    n_estimators: int,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=n_estimators,
        monotone_constraints=[1],
        min_data_in_leaf=min_data_in_leaf,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        verbosity=-1,
        random_state=random_state,
        n_jobs=1,
    )


def fit_monotone_lgbm_calibration(
    score_l: np.ndarray,
    y_l: np.ndarray,
    tuning_grid: TuningGrid,
    cv_folds: int,
    random_state: int,
    n_estimators: int,
) -> CalibrationModel:
    y_l = np.asarray(y_l, dtype=float)
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))
    x_l = pd.DataFrame({"score": score_l})
    if cv_folds <= 1:
        params = {
            "min_data_in_leaf": int(next(iter(tuning_grid.min_data_in_leaf))),
            "num_leaves": int(next(iter(tuning_grid.num_leaves))),
            "max_depth": int(next(iter(tuning_grid.max_depth))),
        }
        final_model = _build_lgbm_model(
            min_data_in_leaf=params["min_data_in_leaf"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            random_state=random_state,
            n_estimators=n_estimators,
        )
        final_model.fit(x_l, y_l)
        return CalibrationModel(
            name="monotone_lgbm",
            fitted=final_model,
            metadata={**params, "cv_mse": np.nan, "y_min": y_min, "y_max": y_max},
        )
    kfold = KFold(
        n_splits=min(cv_folds, len(score_l)),
        shuffle=True,
        random_state=random_state,
    )
    best_score = np.inf
    best_params: Optional[Dict[str, int]] = None

    for min_data in tuning_grid.min_data_in_leaf:
        for num_leaves in tuning_grid.num_leaves:
            for max_depth in tuning_grid.max_depth:
                fold_scores = []
                valid = True
                for fold_id, (train_idx, val_idx) in enumerate(kfold.split(x_l)):
                    if len(train_idx) <= min_data:
                        valid = False
                        break
                    model = _build_lgbm_model(
                        min_data_in_leaf=min_data,
                        num_leaves=num_leaves,
                        max_depth=max_depth,
                        random_state=random_state + fold_id,
                        n_estimators=n_estimators,
                    )
                    model.fit(x_l.iloc[train_idx], y_l[train_idx])
                    pred = clip_unit(model.predict(x_l.iloc[val_idx]))
                    fold_scores.append(np.mean((y_l[val_idx] - pred) ** 2))
                if not valid or not fold_scores:
                    continue
                cv_score = float(np.mean(fold_scores))
                if cv_score < best_score:
                    best_score = cv_score
                    best_params = {
                        "min_data_in_leaf": int(min_data),
                        "num_leaves": int(num_leaves),
                        "max_depth": int(max_depth),
                    }

    if best_params is None:
        best_params = {
            "min_data_in_leaf": int(min(tuning_grid.min_data_in_leaf)),
            "num_leaves": int(next(iter(tuning_grid.num_leaves))),
            "max_depth": int(next(iter(tuning_grid.max_depth))),
        }
        best_score = np.nan

    final_model = _build_lgbm_model(
        min_data_in_leaf=best_params["min_data_in_leaf"],
        num_leaves=best_params["num_leaves"],
        max_depth=best_params["max_depth"],
        random_state=random_state,
        n_estimators=n_estimators,
    )
    final_model.fit(x_l, y_l)
    metadata = {
        **best_params,
        "cv_mse": best_score,
        "y_min": y_min,
        "y_max": y_max,
    }
    return CalibrationModel(name="monotone_lgbm", fitted=final_model, metadata=metadata)


def predict_monotone_lgbm(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    pred = model.fitted.predict(pd.DataFrame({"score": score}))
    y_min = model.metadata.get("y_min", 1e-6)
    y_max = model.metadata.get("y_max", 1.0 - 1e-6)
    return clip_range(pred, y_min, y_max)


def _scalarize_result(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected a scalar-like value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def aipw_em_result(
    y_l: np.ndarray,
    pred_l: np.ndarray,
    pred_u: np.ndarray,
    *,
    alpha: float,
) -> Dict[str, float]:
    result = mean_inference(
        np.asarray(y_l, dtype=float),
        np.asarray(pred_l, dtype=float),
        np.asarray(pred_u, dtype=float),
        method="aipw",
        alpha=float(alpha),
        efficiency_maximization=True,
    )
    lower, upper = result.ci
    return {
        "estimate": _scalarize_result(result.pointestimate),
        "se": _scalarize_result(result.se),
        "ci_lower": _scalarize_result(lower),
        "ci_upper": _scalarize_result(upper),
    }


def plugin_estimate(pred_l: np.ndarray, pred_u: np.ndarray) -> float:
    n = len(pred_l)
    N = len(pred_u)
    rho = n / (n + N)
    return float(rho * np.mean(pred_l) + (1.0 - rho) * np.mean(pred_u))


def plugin_se_from_prediction(pred_l: np.ndarray, pred_u: np.ndarray) -> float:
    n = len(pred_l)
    N = len(pred_u)
    M = n + N
    rho = n / M
    psi_hat = plugin_estimate(pred_l, pred_u)
    d_l = pred_l - psi_hat
    d_u = pred_u - psi_hat
    sigma2 = rho * np.mean(d_l**2) + (1.0 - rho) * np.mean(d_u**2)
    return float(np.sqrt(sigma2 / M))


def aipp_from_prediction(pred_l: np.ndarray, pred_u: np.ndarray, y_l: np.ndarray) -> float:
    return float(plugin_estimate(pred_l, pred_u) + np.mean(y_l - pred_l))


def ppi_mean_from_prediction(pred_l: np.ndarray, pred_u: np.ndarray, y_l: np.ndarray) -> float:
    return float(np.mean(pred_u) + np.mean(y_l - pred_l))


def labeled_mean(y_l: np.ndarray) -> float:
    return float(np.mean(y_l))


def ppi_mean_se_from_prediction(
    pred_l: np.ndarray,
    pred_u: np.ndarray,
    y_l: np.ndarray,
) -> float:
    n = len(y_l)
    N = len(pred_u)
    resid_l = y_l - pred_l
    centered_l = resid_l - np.mean(resid_l)
    centered_u = pred_u - np.mean(pred_u)
    sigma2 = np.mean(centered_l**2) / n + np.mean(centered_u**2) / N
    return float(np.sqrt(sigma2))


def influence_se_from_prediction(
    psi_hat: float,
    pred_l: np.ndarray,
    pred_u: np.ndarray,
    y_l: np.ndarray,
) -> float:
    n = len(y_l)
    N = len(pred_u)
    M = n + N
    rho = n / M
    # For the pooled plug-in estimator
    #   rho E_n[m(X)] + (1-rho) E_N[m(X)] + E_n[Y-m(X)],
    # the two-sample influence decomposition is
    #   Y - (1-rho)m(X)   on labeled points, and
    #   (1-rho)m(X)       on unlabeled points.
    labeled_component = y_l - (1.0 - rho) * pred_l
    unlabeled_component = (1.0 - rho) * pred_u
    centered_l = labeled_component - np.mean(labeled_component)
    centered_u = unlabeled_component - np.mean(unlabeled_component)
    sigma2 = np.mean(centered_l**2) / n + np.mean(centered_u**2) / N
    return float(np.sqrt(sigma2))


def influence_se_labeled_only(y_l: np.ndarray, psi_hat: float) -> float:
    n = len(y_l)
    return float(np.sqrt(np.mean((y_l - psi_hat) ** 2) / n))


def linear_residual_identity(pred_l: np.ndarray, pred_u: np.ndarray, y_l: np.ndarray) -> float:
    n = len(y_l)
    N = len(pred_u)
    rho = n / (n + N)
    return float(rho * np.mean(y_l) + (1.0 - rho) * np.mean(pred_u))
