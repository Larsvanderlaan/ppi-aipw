from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ._calibration import CalibrationModel, calibrate_predictions, canonical_method, fit_calibrator
from ._utils import (
    construct_weight_vector,
    reshape_to_2d,
    restore_shape,
    validate_mean_inputs,
    z_interval,
)


_AUTO_AIPW_EFFICIENCY_LABEL = "aipw_efficiency_maximization"
_PROGNOSTIC_LINEAR_RIDGE_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=float)


@dataclass(frozen=True)
class _AutoCandidateSpec:
    method: str
    label: str
    efficiency_maximization: bool


@dataclass(frozen=True)
class _PreparedMeanInferenceInputs:
    Y_2d: np.ndarray
    pred_labeled_point: np.ndarray
    pred_unlabeled_point: np.ndarray
    pred_labeled_variance: np.ndarray
    pred_unlabeled_variance: np.ndarray
    weights: np.ndarray
    weights_unlabeled_point: np.ndarray
    weights_unlabeled_variance: np.ndarray
    calibrator: Any
    method: str
    selected_candidate: str
    selected_efficiency_maximization: bool
    final_efficiency_maximization: bool
    diagnostics: dict[str, Any]
    efficiency_lambda: float | np.ndarray | None


@dataclass(frozen=True)
class _MeanInferenceState:
    pointestimate: np.ndarray
    se: np.ndarray | None
    ci: tuple[np.ndarray, np.ndarray] | None
    method: str
    selected_candidate: str
    selected_efficiency_maximization: bool
    final_efficiency_maximization: bool
    efficiency_lambda: float | np.ndarray | None
    inference: str
    diagnostics: dict[str, Any]
    calibrator: Any


@dataclass(frozen=True)
class MeanInferenceResult:
    pointestimate: float | np.ndarray
    se: float | np.ndarray
    ci: tuple[float | np.ndarray, float | np.ndarray]
    method: str
    selected_candidate: str
    selected_efficiency_maximization: bool
    efficiency_lambda: float | np.ndarray | None
    inference: str
    diagnostics: dict[str, Any]
    calibrator: Any


@dataclass
class PrognosticLinearModel:
    coefficients: list[np.ndarray]
    x_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)
    method: str = "prognostic_linear"

    def predict(
        self,
        scores: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        scores_2d = reshape_to_2d(np.asarray(scores, dtype=float))
        if scores_2d.shape[1] != len(self.coefficients):
            raise ValueError(
                f"Expected {len(self.coefficients)} score columns, got {scores_2d.shape[1]}."
            )
        X_2d = _coerce_covariates(X, n_obs=scores_2d.shape[0], name="X")
        if self.x_dim == 0:
            if X_2d is not None and X_2d.shape[1] != 0:
                raise ValueError("This prognostic_linear model was fit without extra covariates.")
            X_design = np.zeros((scores_2d.shape[0], 0), dtype=float)
        else:
            if X_2d is None:
                raise ValueError(
                    "This prognostic_linear model was fit with extra covariates; "
                    "pass X when calling predict(...)."
                )
            if X_2d.shape[1] != self.x_dim:
                raise ValueError(f"Expected X with {self.x_dim} columns, got {X_2d.shape[1]}.")
            X_design = X_2d

        pred = np.column_stack(
            [
                np.column_stack([np.ones(scores_2d.shape[0]), scores_2d[:, idx], X_design]) @ coef
                for idx, coef in enumerate(self.coefficients)
            ]
        )
        return restore_shape(pred, np.asarray(scores))


def _labeled_fraction(n_labeled: int, n_unlabeled: int) -> float:
    return n_labeled / float(n_labeled + n_unlabeled)


def _coerce_covariates(
    X: np.ndarray | None,
    *,
    n_obs: int,
    name: str,
) -> np.ndarray | None:
    if X is None:
        return None
    X_2d = reshape_to_2d(np.asarray(X, dtype=float))
    if X_2d.shape[0] != n_obs:
        raise ValueError(f"{name} must have {n_obs} rows, got {X_2d.shape[0]}.")
    return X_2d


def _validate_prognostic_covariates(
    X: np.ndarray | None,
    X_unlabeled: np.ndarray | None,
    *,
    n_labeled: int,
    n_unlabeled: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    X_2d = _coerce_covariates(X, n_obs=n_labeled, name="X")
    X_unlabeled_2d = _coerce_covariates(X_unlabeled, n_obs=n_unlabeled, name="X_unlabeled")
    if (X_2d is None) != (X_unlabeled_2d is None):
        raise ValueError("Provide both X and X_unlabeled together, or neither.")
    if X_2d is not None and X_unlabeled_2d is not None and X_2d.shape[1] != X_unlabeled_2d.shape[1]:
        raise ValueError(
            "X and X_unlabeled must have the same number of columns. "
            f"Got {X_2d.shape[1]} and {X_unlabeled_2d.shape[1]}."
        )
    return X_2d, X_unlabeled_2d


def _solve_prognostic_linear_system(
    score: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    z = np.column_stack([np.ones(score.shape[0]), score])
    z_w = z * sample_weight[:, None]
    if X.shape[1] == 0:
        gram = z.T @ z_w
        rhs = z_w.T @ y
        return np.asarray(np.linalg.pinv(gram) @ rhs, dtype=float)

    x_w = X * sample_weight[:, None]
    zz = z.T @ z_w
    zx = z.T @ x_w
    xx = X.T @ x_w + float(alpha) * np.eye(X.shape[1], dtype=float)
    gram = np.block([[zz, zx], [zx.T, xx]])
    rhs = np.concatenate([z_w.T @ y, x_w.T @ y])
    return np.asarray(np.linalg.pinv(gram) @ rhs, dtype=float)


def _predict_prognostic_linear_from_coef(
    coef: np.ndarray,
    score: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(score.shape[0]), score, X])
    return np.asarray(design @ coef, dtype=float)


def _weighted_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    normalized_weight = sample_weight / np.sum(sample_weight)
    return float(np.sum(normalized_weight * (y_true - y_pred) ** 2))


def _select_prognostic_linear_alpha(
    y: np.ndarray,
    score: np.ndarray,
    X: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    if X.shape[1] == 0:
        return 0.0
    n_obs = y.shape[0]
    n_splits = min(5, n_obs)
    if n_splits < 2:
        return 1.0
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_alpha = float(_PROGNOSTIC_LINEAR_RIDGE_GRID[0])
    best_score = np.inf
    for alpha in _PROGNOSTIC_LINEAR_RIDGE_GRID:
        fold_errors = []
        for train_idx, val_idx in splitter.split(score):
            coef = _solve_prognostic_linear_system(
                score[train_idx],
                X[train_idx],
                y[train_idx],
                sample_weight[train_idx],
                alpha=float(alpha),
            )
            pred_val = _predict_prognostic_linear_from_coef(coef, score[val_idx], X[val_idx])
            fold_errors.append(
                _weighted_prediction_error(y[val_idx], pred_val, sample_weight[val_idx])
            )
        mean_error = float(np.mean(fold_errors))
        if mean_error < best_score:
            best_score = mean_error
            best_alpha = float(alpha)
    return best_alpha


def _fit_prognostic_linear(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    X: np.ndarray | None,
    X_unlabeled: np.ndarray | None,
    w: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PrognosticLinearModel]:
    Y_2d, Yhat_2d, Yhat_unlabeled_2d = validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
    X_2d, X_unlabeled_2d = _validate_prognostic_covariates(
        X,
        X_unlabeled,
        n_labeled=Y_2d.shape[0],
        n_unlabeled=Yhat_unlabeled_2d.shape[0],
    )
    X_design = np.zeros((Y_2d.shape[0], 0), dtype=float) if X_2d is None else X_2d
    X_unlabeled_design = (
        np.zeros((Yhat_unlabeled_2d.shape[0], 0), dtype=float) if X_unlabeled_2d is None else X_unlabeled_2d
    )
    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=False)

    pred_labeled = np.zeros_like(Y_2d, dtype=float)
    pred_unlabeled = np.zeros_like(Yhat_unlabeled_2d, dtype=float)
    coefficients: list[np.ndarray] = []
    selected_alphas: list[float] = []
    for idx in range(Y_2d.shape[1]):
        score_labeled = Yhat_2d[:, idx]
        score_unlabeled = Yhat_unlabeled_2d[:, idx]
        alpha = _select_prognostic_linear_alpha(
            Y_2d[:, idx],
            score_labeled,
            X_design,
            weights,
        )
        coef = _solve_prognostic_linear_system(
            score_labeled,
            X_design,
            Y_2d[:, idx],
            weights,
            alpha=alpha,
        )
        pred_labeled[:, idx] = _predict_prognostic_linear_from_coef(coef, score_labeled, X_design)
        pred_unlabeled[:, idx] = _predict_prognostic_linear_from_coef(
            coef,
            score_unlabeled,
            X_unlabeled_design,
        )
        coefficients.append(coef)
        selected_alphas.append(alpha)

    model = PrognosticLinearModel(
        coefficients=coefficients,
        x_dim=X_design.shape[1],
        metadata={
            "n_outputs": Y_2d.shape[1],
            "x_dim": X_design.shape[1],
            "uses_covariates": bool(X_design.shape[1] > 0),
            "ridge_alpha": selected_alphas[0] if len(selected_alphas) == 1 else np.asarray(selected_alphas),
            "ridge_penalizes_only_covariates": True,
        },
    )
    return Y_2d, Yhat_2d, pred_labeled, pred_unlabeled, model


def _fit_and_calibrate(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
    X: np.ndarray | None,
    X_unlabeled: np.ndarray | None,
    isocal_backend: str,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    if method == "prognostic_linear":
        return _fit_prognostic_linear(
            Y,
            Yhat,
            Yhat_unlabeled,
            X=X,
            X_unlabeled=X_unlabeled,
            w=w,
        )
    Y_2d, Yhat_2d, Yhat_unlabeled_2d = validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
    model = fit_calibrator(
        Y_2d,
        Yhat_2d,
        method=method,
        w=w,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    pred_labeled = np.asarray(model.predict(Yhat_2d), dtype=float)
    pred_unlabeled = np.asarray(model.predict(Yhat_unlabeled_2d), dtype=float)
    return Y_2d, Yhat_2d, pred_labeled, pred_unlabeled, model


def _coerce_generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _bootstrap_pointestimates(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool,
    candidate_methods: tuple[str, ...],
    num_folds: int,
    auto_unlabeled_subsample_size: int | None,
    selection_random_state: int | np.random.Generator | None,
    isocal_backend: str,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
    n_resamples: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2 when inference='bootstrap'.")

    Y_arr = np.asarray(Y)
    Yhat_arr = np.asarray(Yhat)
    Yhat_unlabeled_arr = np.asarray(Yhat_unlabeled)
    X_arr = None if X is None else np.asarray(X, dtype=float)
    X_unlabeled_arr = None if X_unlabeled is None else np.asarray(X_unlabeled, dtype=float)
    if w is not None:
        w = np.asarray(w, dtype=float)
    if w_unlabeled is not None:
        w_unlabeled = np.asarray(w_unlabeled, dtype=float)

    n_labeled = Y_arr.shape[0]
    n_unlabeled = Yhat_unlabeled_arr.shape[0]
    rng = _coerce_generator(random_state)

    bootstrap_estimates = []
    for _ in range(n_resamples):
        labeled_idx = rng.integers(0, n_labeled, size=n_labeled)
        unlabeled_idx = rng.integers(0, n_unlabeled, size=n_unlabeled)
        prepared = _prepare_inference_inputs(
            Y_arr[labeled_idx],
            Yhat_arr[labeled_idx],
            Yhat_unlabeled_arr[unlabeled_idx],
            method=method,
            w=None if w is None else w[labeled_idx],
            w_unlabeled=None if w_unlabeled is None else w_unlabeled[unlabeled_idx],
            X=None if X_arr is None else X_arr[labeled_idx],
            X_unlabeled=None if X_unlabeled_arr is None else X_unlabeled_arr[unlabeled_idx],
            efficiency_maximization=efficiency_maximization,
            candidate_methods=candidate_methods,
            num_folds=num_folds,
            auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
            selection_random_state=selection_random_state,
            isocal_backend=isocal_backend,
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )
        estimate = _aipw_mean_pointestimate_from_predictions(
            prepared.Y_2d,
            prepared.pred_labeled_point,
            prepared.pred_unlabeled_point,
            w=prepared.weights,
            w_unlabeled=prepared.weights_unlabeled_point,
        ).reshape(1, -1)
        bootstrap_estimates.append(estimate)

    return np.concatenate(bootstrap_estimates, axis=0)


def _aipw_mean_pointestimate_from_predictions(
    Y: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    *,
    w: np.ndarray,
    w_unlabeled: np.ndarray,
) -> np.ndarray:
    n_labeled = Y.shape[0]
    n_unlabeled = pred_unlabeled.shape[0]
    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)

    labeled_term = (w * (Y - c * pred_labeled)).mean(axis=0)
    unlabeled_term = (w_unlabeled * (c * pred_unlabeled)).mean(axis=0)
    return labeled_term + unlabeled_term


def _resolve_efficiency_maximization(
    efficiency_maximization: bool | None,
) -> bool:
    return bool(efficiency_maximization)


def _resolve_num_folds(
    num_folds: int | None,
) -> int:
    if num_folds is None:
        return 100
    return int(num_folds)


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


def _resolve_selection_seeds(
    selection_random_state: int | np.random.Generator | None,
) -> tuple[int, int]:
    if isinstance(selection_random_state, np.random.Generator):
        return (
            int(selection_random_state.integers(0, np.iinfo(np.int32).max)),
            int(selection_random_state.integers(0, np.iinfo(np.int32).max)),
        )
    if selection_random_state is None:
        rng = np.random.default_rng()
        return (
            int(rng.integers(0, np.iinfo(np.int32).max)),
            int(rng.integers(0, np.iinfo(np.int32).max)),
        )
    seed = int(selection_random_state)
    return seed, seed


def _subset_unlabeled_for_auto(
    Yhat_unlabeled_2d: np.ndarray,
    w_unlabeled: np.ndarray | None,
    X_unlabeled: np.ndarray | None,
    *,
    n_labeled: int,
    auto_unlabeled_subsample_size: int | None,
    subset_seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    n_unlabeled = Yhat_unlabeled_2d.shape[0]
    subset_size = _resolve_auto_unlabeled_subsample_size(
        n_labeled,
        n_unlabeled,
        auto_unlabeled_subsample_size,
    )
    if subset_size >= n_unlabeled:
        return (
            Yhat_unlabeled_2d,
            w_unlabeled,
            X_unlabeled,
            {
                "auto_unlabeled_subsample_size": n_unlabeled,
                "auto_unlabeled_subsample_default": auto_unlabeled_subsample_size is None,
                "unlabeled_strategy": "all_unlabeled_rows_in_each_fold",
            },
        )

    rng = np.random.default_rng(subset_seed)
    subset_idx = np.sort(rng.choice(n_unlabeled, size=subset_size, replace=False))
    subset_weights = None if w_unlabeled is None else w_unlabeled[subset_idx]
    subset_covariates = None if X_unlabeled is None else X_unlabeled[subset_idx]
    return (
        Yhat_unlabeled_2d[subset_idx],
        subset_weights,
        subset_covariates,
        {
            "auto_unlabeled_subsample_size": subset_size,
            "auto_unlabeled_subsample_default": auto_unlabeled_subsample_size is None,
            "auto_unlabeled_subsample_seed": subset_seed,
            "unlabeled_strategy": "subsampled_unlabeled_rows_in_each_fold",
        },
    )


def _format_coordinate_parameter(x: np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x[0]) if x.size == 1 else x


def _estimate_efficiency_lambda(
    Y: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    *,
    w: np.ndarray,
    w_unlabeled: np.ndarray,
) -> np.ndarray:
    n_labeled = Y.shape[0]
    n_unlabeled = pred_unlabeled.shape[0]
    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)

    labeled_outcome = w * Y
    labeled_score = c * w * pred_labeled
    unlabeled_score = c * w_unlabeled * pred_unlabeled

    numerator = np.mean(
        (labeled_outcome - labeled_outcome.mean(axis=0))
        * (labeled_score - labeled_score.mean(axis=0)),
        axis=0,
    )
    denominator = np.var(labeled_score, axis=0) + (n_labeled / float(n_unlabeled)) * np.var(
        unlabeled_score,
        axis=0,
    )
    lambda_hat = np.divide(
        numerator,
        denominator,
        out=np.zeros(Y.shape[1], dtype=float),
        where=denominator > 0,
    )
    return lambda_hat


def _apply_efficiency_scaling(
    Y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_unlabeled: np.ndarray,
    predictions_to_scale: tuple[np.ndarray, ...],
    *,
    w_train: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    weights_train = construct_weight_vector(Y_train.shape[0], w_train, vectorized=True)
    weights_unlabeled = construct_weight_vector(pred_unlabeled.shape[0], w_unlabeled, vectorized=True)
    lambda_hat = _estimate_efficiency_lambda(
        Y_train,
        pred_train,
        pred_unlabeled,
        w=weights_train,
        w_unlabeled=weights_unlabeled,
    )
    scaled_predictions = [pred * lambda_hat.reshape(1, -1) for pred in predictions_to_scale]
    return lambda_hat, scaled_predictions


def _auto_candidate_specs(candidate_methods: tuple[str, ...]) -> tuple[list[str], list[_AutoCandidateSpec]]:
    canonical_candidates: list[str] = []
    for method_name in candidate_methods:
        canonical = canonical_method(method_name)
        if canonical not in canonical_candidates:
            canonical_candidates.append(canonical)
    if not canonical_candidates:
        raise ValueError("candidate_methods must contain at least one valid method.")

    specs = [
        _AutoCandidateSpec(
            method=method_name,
            label=method_name,
            efficiency_maximization=False,
        )
        for method_name in canonical_candidates
    ]
    if "aipw" in canonical_candidates:
        specs.append(
            _AutoCandidateSpec(
                method="aipw",
                label=_AUTO_AIPW_EFFICIENCY_LABEL,
                efficiency_maximization=True,
            )
        )
    return canonical_candidates, specs


def _cv_selection_score(
    Y: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    *,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
) -> float:
    n_labeled = Y.shape[0]
    n_unlabeled = pred_unlabeled.shape[0]
    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)
    weights = construct_weight_vector(n_labeled, w, vectorized=True)
    weights_unlabeled = construct_weight_vector(n_unlabeled, w_unlabeled, vectorized=True)
    labeled_component = weights * (Y - c * pred_labeled)
    unlabeled_component = weights_unlabeled * (c * pred_unlabeled)
    componentwise_score = np.var(labeled_component, axis=0) / n_labeled + np.var(
        unlabeled_component,
        axis=0,
    ) / n_unlabeled
    return float(np.sum(componentwise_score))


def _candidate_cv_predictions(
    Y_2d: np.ndarray,
    Yhat_2d: np.ndarray,
    Yhat_unlabeled_2d: np.ndarray,
    *,
    method_name: str,
    splitter: KFold,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    X_2d: np.ndarray | None,
    X_unlabeled_2d: np.ndarray | None,
    efficiency_maximization: bool,
    isocal_backend: str,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_oof = np.zeros_like(Y_2d, dtype=float)
    pred_unlabeled_folds = []

    for train_idx, val_idx in splitter.split(Y_2d):
        y_train = Y_2d[train_idx]
        yhat_train = Yhat_2d[train_idx]
        yhat_val = Yhat_2d[val_idx]
        w_train = None if w is None else w[train_idx]
        X_train = None if X_2d is None else X_2d[train_idx]
        X_val = None if X_2d is None else X_2d[val_idx]
        if method_name == "prognostic_linear":
            _, _, pred_train, _, calibrator = _fit_prognostic_linear(
                y_train,
                yhat_train,
                Yhat_unlabeled_2d,
                X=X_train,
                X_unlabeled=X_unlabeled_2d,
                w=w_train,
            )
            pred_val = np.asarray(calibrator.predict(yhat_val, X=X_val), dtype=float)
            pred_unlabeled = np.asarray(calibrator.predict(Yhat_unlabeled_2d, X=X_unlabeled_2d), dtype=float)
        else:
            calibrator = fit_calibrator(
                y_train,
                yhat_train,
                method=method_name,
                w=w_train,
                isocal_backend=isocal_backend,
                isocal_max_depth=isocal_max_depth,
                isocal_min_child_weight=isocal_min_child_weight,
            )
            pred_train = np.asarray(calibrator.predict(yhat_train), dtype=float)
            pred_val = np.asarray(calibrator.predict(yhat_val), dtype=float)
            pred_unlabeled = np.asarray(calibrator.predict(Yhat_unlabeled_2d), dtype=float)

        if efficiency_maximization:
            _, scaled_predictions = _apply_efficiency_scaling(
                y_train,
                pred_train,
                pred_unlabeled,
                (pred_val, pred_unlabeled),
                w_train=w_train,
                w_unlabeled=w_unlabeled,
            )
            pred_val, pred_unlabeled = scaled_predictions

        pred_oof[val_idx] = pred_val
        pred_unlabeled_folds.append(pred_unlabeled)

    mean_pred_unlabeled = np.mean(np.stack(pred_unlabeled_folds, axis=0), axis=0)
    return pred_oof, mean_pred_unlabeled


def _select_mean_method_cv_internal(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    num_folds: int = 5,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> tuple[str, dict[str, Any], np.ndarray, np.ndarray]:
    requested_efficiency_maximization = _resolve_efficiency_maximization(efficiency_maximization)
    Y_2d, Yhat_2d, Yhat_unlabeled_2d = validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
    X_2d, X_unlabeled_2d = _validate_prognostic_covariates(
        X,
        X_unlabeled,
        n_labeled=Y_2d.shape[0],
        n_unlabeled=Yhat_unlabeled_2d.shape[0],
    )
    n_labeled = Y_2d.shape[0]

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if n_labeled < 2:
        raise ValueError("Need at least two labeled observations for CV-based method selection.")

    canonical_candidates, candidate_specs = _auto_candidate_specs(candidate_methods)

    n_splits = min(num_folds, n_labeled)
    if n_splits < 2:
        raise ValueError("num_folds is too large relative to the number of labeled observations.")

    subset_seed, cv_seed = _resolve_selection_seeds(selection_random_state)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    if w is not None:
        w = np.asarray(w, dtype=float).reshape(-1)
    if w_unlabeled is not None:
        w_unlabeled = np.asarray(w_unlabeled, dtype=float).reshape(-1)
    Yhat_unlabeled_selection_2d, w_unlabeled_selection, X_unlabeled_selection_2d, unlabeled_metadata = (
        _subset_unlabeled_for_auto(
        Yhat_unlabeled_2d,
        w_unlabeled,
        X_unlabeled_2d,
        n_labeled=n_labeled,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        subset_seed=subset_seed,
        )
    )

    base_cv_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_name in canonical_candidates:
        base_cv_predictions[method_name] = _candidate_cv_predictions(
            Y_2d,
            Yhat_2d,
            Yhat_unlabeled_selection_2d,
            method_name=method_name,
            splitter=splitter,
            w=w,
            w_unlabeled=w_unlabeled_selection,
            X_2d=X_2d,
            X_unlabeled_2d=X_unlabeled_selection_2d,
            efficiency_maximization=False,
            isocal_backend=isocal_backend,
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )

    scores: dict[str, float] = {}
    for candidate_spec in candidate_specs:
        if candidate_spec.efficiency_maximization:
            pred_oof, mean_pred_unlabeled = _candidate_cv_predictions(
                Y_2d,
                Yhat_2d,
                Yhat_unlabeled_selection_2d,
                method_name=candidate_spec.method,
                splitter=splitter,
                w=w,
                w_unlabeled=w_unlabeled_selection,
                X_2d=X_2d,
                X_unlabeled_2d=X_unlabeled_selection_2d,
                efficiency_maximization=True,
                isocal_backend=isocal_backend,
                isocal_max_depth=isocal_max_depth,
                isocal_min_child_weight=isocal_min_child_weight,
            )
        else:
            pred_oof, mean_pred_unlabeled = base_cv_predictions[candidate_spec.method]
        scores[candidate_spec.label] = _cv_selection_score(
            Y_2d,
            pred_oof,
            mean_pred_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled_selection,
        )

    selected_label = min(scores, key=scores.get)
    selected_candidate = next(
        candidate_spec for candidate_spec in candidate_specs if candidate_spec.label == selected_label
    )
    selected_method = selected_candidate.method
    diagnostics = {
        "selected_method": selected_method,
        "selected_candidate": selected_label,
        "selected_efficiency_maximization": selected_candidate.efficiency_maximization,
        "selection_metric": "cross_validated_if_variance",
        "requested_efficiency_maximization": requested_efficiency_maximization,
        "selection_ignores_requested_efficiency_maximization": True,
        "num_folds": n_splits,
        **unlabeled_metadata,
        "scores": scores,
    }
    selected_pred_labeled, selected_pred_unlabeled = base_cv_predictions[selected_method]
    return selected_method, diagnostics, selected_pred_labeled, selected_pred_unlabeled


def select_mean_method_cv(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> tuple[str, dict[str, Any]]:
    """Selects a mean-estimation method by cross-validated IF-variance minimization.

    The selector compares candidate methods using out-of-fold labeled predictions
    together with fold-averaged unlabeled predictions, then chooses the method
    with the smallest estimated influence-function variance. When ``"aipw"``
    appears in ``candidate_methods``, the selector also compares against an
    efficiency-maximized AIPW candidate. By default, the foldwise objective uses
    a fixed unlabeled subsample of size ``min(n_unlabeled, 10 * n_labeled)``;
    set ``auto_unlabeled_subsample_size`` to override this.
    """
    resolved_num_folds = _resolve_num_folds(num_folds)
    selected_method, diagnostics, _, _ = _select_mean_method_cv_internal(
        Y,
        Yhat,
        Yhat_unlabeled,
        candidate_methods=candidate_methods,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        num_folds=resolved_num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    return selected_method, diagnostics


def _prepare_inference_inputs(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int = 5,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> _PreparedMeanInferenceInputs:
    if method.lower() == "auto":
        selected_method, selection_metadata, pred_labeled_cf_base, pred_unlabeled_cf_base = (
            _select_mean_method_cv_internal(
                Y,
                Yhat,
                Yhat_unlabeled,
                candidate_methods=candidate_methods,
                w=w,
                w_unlabeled=w_unlabeled,
                X=X,
                X_unlabeled=X_unlabeled,
                efficiency_maximization=efficiency_maximization,
                num_folds=num_folds,
                auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
                selection_random_state=selection_random_state,
                isocal_backend=isocal_backend,
                isocal_max_depth=isocal_max_depth,
                isocal_min_child_weight=isocal_min_child_weight,
            )
        )
        Y_2d, _, pred_labeled_full, pred_unlabeled_full, calibrator = _fit_and_calibrate(
            Y,
            Yhat,
            Yhat_unlabeled,
            method=selected_method,
            w=w,
            X=X,
            X_unlabeled=X_unlabeled,
            isocal_backend=isocal_backend,
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )
        weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=True)
        weights_unlabeled_point = construct_weight_vector(
            pred_unlabeled_full.shape[0],
            w_unlabeled,
            vectorized=True,
        )
        _, _, Yhat_unlabeled_2d = validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
        _, X_unlabeled_2d = _validate_prognostic_covariates(
            X,
            X_unlabeled,
            n_labeled=Y_2d.shape[0],
            n_unlabeled=Yhat_unlabeled_2d.shape[0],
        )
        w_unlabeled_arr = None if w_unlabeled is None else np.asarray(w_unlabeled, dtype=float).reshape(-1)
        _, w_unlabeled_selection, _, _ = _subset_unlabeled_for_auto(
            Yhat_unlabeled_2d,
            w_unlabeled_arr,
            X_unlabeled_2d,
            n_labeled=Y_2d.shape[0],
            auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
            subset_seed=int(selection_metadata["auto_unlabeled_subsample_seed"])
            if "auto_unlabeled_subsample_seed" in selection_metadata
            else 0,
        )
        weights_unlabeled_variance = construct_weight_vector(
            pred_unlabeled_cf_base.shape[0],
            w_unlabeled_selection,
            vectorized=True,
        )

        pred_labeled_point = pred_labeled_full
        pred_unlabeled_point = pred_unlabeled_full
        pred_labeled_variance = pred_labeled_cf_base
        pred_unlabeled_variance = pred_unlabeled_cf_base
        final_efficiency_maximization = bool(
            efficiency_maximization or selection_metadata.get("selected_efficiency_maximization", False)
        )

        efficiency_lambda_out: float | np.ndarray | None = None
        diagnostics = dict(selection_metadata)
        diagnostics["final_efficiency_maximization"] = final_efficiency_maximization
        diagnostics["lambda_from_cross_fitted_estimates"] = final_efficiency_maximization
        if final_efficiency_maximization:
            lambda_hat = _estimate_efficiency_lambda(
                Y_2d,
                pred_labeled_cf_base,
                pred_unlabeled_cf_base,
                w=weights,
                w_unlabeled=weights_unlabeled_variance,
            )
            scale = lambda_hat.reshape(1, -1)
            pred_labeled_point = pred_labeled_point * scale
            pred_unlabeled_point = pred_unlabeled_point * scale
            pred_labeled_variance = pred_labeled_variance * scale
            pred_unlabeled_variance = pred_unlabeled_variance * scale
            efficiency_lambda_out = _format_coordinate_parameter(lambda_hat)
            diagnostics["efficiency_lambda_source"] = "cross_fitted"
            calibrator.metadata["efficiency_lambda"] = efficiency_lambda_out
            calibrator.metadata["efficiency_lambda_source"] = "cross_fitted"

        calibrator.metadata.update(diagnostics)
        return _PreparedMeanInferenceInputs(
            Y_2d=Y_2d,
            pred_labeled_point=pred_labeled_point,
            pred_unlabeled_point=pred_unlabeled_point,
            pred_labeled_variance=pred_labeled_variance,
            pred_unlabeled_variance=pred_unlabeled_variance,
            weights=weights,
            weights_unlabeled_point=weights_unlabeled_point,
            weights_unlabeled_variance=weights_unlabeled_variance,
            calibrator=calibrator,
            method=selected_method,
            selected_candidate=str(diagnostics["selected_candidate"]),
            selected_efficiency_maximization=bool(diagnostics["selected_efficiency_maximization"]),
            final_efficiency_maximization=final_efficiency_maximization,
            diagnostics=diagnostics,
            efficiency_lambda=efficiency_lambda_out,
        )

    canonical = canonical_method(method)
    Y_2d, _, pred_labeled, pred_unlabeled, calibrator = _fit_and_calibrate(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=canonical,
        w=w,
        X=X,
        X_unlabeled=X_unlabeled,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=True)
    weights_unlabeled = construct_weight_vector(pred_unlabeled.shape[0], w_unlabeled, vectorized=True)
    efficiency_lambda_out: float | np.ndarray | None = None
    final_efficiency_maximization = bool(efficiency_maximization)
    diagnostics: dict[str, Any] = {
        "selected_method": canonical,
        "selected_candidate": canonical,
        "selected_efficiency_maximization": final_efficiency_maximization,
        "final_efficiency_maximization": final_efficiency_maximization,
        "lambda_from_cross_fitted_estimates": False,
    }
    if final_efficiency_maximization:
        lambda_hat = _estimate_efficiency_lambda(
            Y_2d,
            pred_labeled,
            pred_unlabeled,
            w=weights,
            w_unlabeled=weights_unlabeled,
        )
        scale = lambda_hat.reshape(1, -1)
        pred_labeled = pred_labeled * scale
        pred_unlabeled = pred_unlabeled * scale
        efficiency_lambda_out = _format_coordinate_parameter(lambda_hat)
        diagnostics["efficiency_lambda_source"] = "full_sample"
        calibrator.metadata["efficiency_lambda"] = efficiency_lambda_out
        calibrator.metadata["efficiency_lambda_source"] = "full_sample"

    calibrator.metadata.update(diagnostics)
    return _PreparedMeanInferenceInputs(
        Y_2d=Y_2d,
        pred_labeled_point=pred_labeled,
        pred_unlabeled_point=pred_unlabeled,
        pred_labeled_variance=pred_labeled,
        pred_unlabeled_variance=pred_unlabeled,
        weights=weights,
        weights_unlabeled_point=weights_unlabeled,
        weights_unlabeled_variance=weights_unlabeled,
        calibrator=calibrator,
        method=canonical,
        selected_candidate=canonical,
        selected_efficiency_maximization=final_efficiency_maximization,
        final_efficiency_maximization=final_efficiency_maximization,
        diagnostics=diagnostics,
        efficiency_lambda=efficiency_lambda_out,
    )


def _prepare_mean_estimation_inputs(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int = 5,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    prepared = _prepare_inference_inputs(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    return (
        prepared.Y_2d,
        prepared.pred_labeled_point,
        prepared.pred_unlabeled_point,
        prepared.weights,
        prepared.weights_unlabeled_point,
        prepared.calibrator,
    )


def _prepare_auto_variance_inputs(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool,
    candidate_methods: tuple[str, ...],
    num_folds: int,
    auto_unlabeled_subsample_size: int | None,
    selection_random_state: int | np.random.Generator | None,
    isocal_backend: str,
    isocal_max_depth: int,
    isocal_min_child_weight: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
]:
    prepared = _prepare_inference_inputs(
        Y,
        Yhat,
        Yhat_unlabeled,
        method="auto",
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    return (
        prepared.Y_2d,
        prepared.pred_labeled_point,
        prepared.pred_unlabeled_point,
        prepared.pred_labeled_variance,
        prepared.pred_unlabeled_variance,
        prepared.weights,
        prepared.weights_unlabeled_point,
        prepared.weights_unlabeled_variance,
        prepared.calibrator,
    )


def _wald_standard_error(
    prepared: _PreparedMeanInferenceInputs,
) -> np.ndarray:
    labeled_component, unlabeled_component = _wald_influence_components(prepared)
    n_labeled = prepared.Y_2d.shape[0]
    n_unlabeled = prepared.pred_unlabeled_variance.shape[0]
    return np.sqrt(
        np.var(labeled_component, axis=0) / n_labeled + np.var(unlabeled_component, axis=0) / n_unlabeled
    )


def _wald_influence_components(
    prepared: _PreparedMeanInferenceInputs,
) -> tuple[np.ndarray, np.ndarray]:
    n_labeled = prepared.Y_2d.shape[0]
    n_unlabeled = prepared.pred_unlabeled_variance.shape[0]
    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)
    labeled_component = prepared.weights * (prepared.Y_2d - c * prepared.pred_labeled_variance)
    unlabeled_component = prepared.weights_unlabeled_variance * (c * prepared.pred_unlabeled_variance)
    return labeled_component, unlabeled_component


def _mean_inference_result_from_prepared(
    prepared: _PreparedMeanInferenceInputs,
    *,
    alpha: float,
    alternative: str,
    reference: np.ndarray,
) -> MeanInferenceResult:
    pointestimate = _aipw_mean_pointestimate_from_predictions(
        prepared.Y_2d,
        prepared.pred_labeled_point,
        prepared.pred_unlabeled_point,
        w=prepared.weights,
        w_unlabeled=prepared.weights_unlabeled_point,
    )
    standard_error = _wald_standard_error(prepared)
    ci = z_interval(pointestimate, standard_error, alpha, alternative)
    diagnostics = dict(prepared.diagnostics)
    diagnostics["inference"] = "wald"
    return MeanInferenceResult(
        pointestimate=restore_shape(pointestimate, reference),
        se=restore_shape(standard_error, reference),
        ci=(
            restore_shape(ci[0], reference),
            restore_shape(ci[1], reference),
        ),
        method=prepared.method,
        selected_candidate=prepared.selected_candidate,
        selected_efficiency_maximization=prepared.selected_efficiency_maximization,
        efficiency_lambda=prepared.efficiency_lambda,
        inference="wald",
        diagnostics=diagnostics,
        calibrator=prepared.calibrator,
    )


def _bootstrap_interval(
    bootstrap_estimates: np.ndarray,
    *,
    alpha: float,
    alternative: str,
) -> tuple[np.ndarray, np.ndarray]:
    if alternative == "two-sided":
        lower = np.quantile(bootstrap_estimates, alpha / 2.0, axis=0)
        upper = np.quantile(bootstrap_estimates, 1.0 - alpha / 2.0, axis=0)
    elif alternative == "larger":
        lower = np.quantile(bootstrap_estimates, alpha, axis=0)
        upper = np.full(lower.shape, np.inf, dtype=float)
    elif alternative == "smaller":
        lower = np.full(bootstrap_estimates.shape[1], -np.inf, dtype=float)
        upper = np.quantile(bootstrap_estimates, 1.0 - alpha, axis=0)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'.")
    return lower, upper


def _fit_mean_inference(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
    compute_se: bool = True,
    compute_ci: bool = True,
) -> _MeanInferenceState:
    inference = inference.lower()
    if inference not in {"wald", "bootstrap"}:
        raise ValueError("inference must be either 'wald' or 'bootstrap'.")
    if compute_ci:
        compute_se = True

    resolved_efficiency_maximization = _resolve_efficiency_maximization(efficiency_maximization)
    resolved_num_folds = _resolve_num_folds(num_folds)
    prepared = _prepare_inference_inputs(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=resolved_efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=resolved_num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    pointestimate = _aipw_mean_pointestimate_from_predictions(
        prepared.Y_2d,
        prepared.pred_labeled_point,
        prepared.pred_unlabeled_point,
        w=prepared.weights,
        w_unlabeled=prepared.weights_unlabeled_point,
    )
    diagnostics = dict(prepared.diagnostics)
    diagnostics["inference"] = inference

    standard_error: np.ndarray | None = None
    ci: tuple[np.ndarray, np.ndarray] | None = None
    if inference == "wald":
        if compute_se:
            standard_error = _wald_standard_error(prepared)
        if compute_ci and standard_error is not None:
            ci = z_interval(pointestimate, standard_error, alpha, alternative)
    else:
        if method.lower() == "auto":
            diagnostics["bootstrap_selected_once"] = True
            diagnostics["bootstrap_method"] = prepared.method
            diagnostics["bootstrap_efficiency_maximization"] = prepared.final_efficiency_maximization
        if compute_se:
            bootstrap_estimates = _bootstrap_pointestimates(
                Y,
                Yhat,
                Yhat_unlabeled,
                method=prepared.method,
                w=w,
                w_unlabeled=w_unlabeled,
                X=X,
                X_unlabeled=X_unlabeled,
                efficiency_maximization=prepared.final_efficiency_maximization,
                candidate_methods=candidate_methods,
                num_folds=resolved_num_folds,
                auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
                selection_random_state=selection_random_state,
                isocal_backend=isocal_backend,
                isocal_max_depth=isocal_max_depth,
                isocal_min_child_weight=isocal_min_child_weight,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            standard_error = np.std(bootstrap_estimates, axis=0, ddof=1)
            if compute_ci:
                ci = _bootstrap_interval(bootstrap_estimates, alpha=alpha, alternative=alternative)

    return _MeanInferenceState(
        pointestimate=pointestimate,
        se=standard_error,
        ci=ci,
        method=prepared.method,
        selected_candidate=prepared.selected_candidate,
        selected_efficiency_maximization=prepared.selected_efficiency_maximization,
        final_efficiency_maximization=prepared.final_efficiency_maximization,
        efficiency_lambda=prepared.efficiency_lambda,
        inference=inference,
        diagnostics=diagnostics,
        calibrator=prepared.calibrator,
    )


def aipw_mean_inference(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> MeanInferenceResult:
    """Computes mean estimation and uncertainty in one shared pass.

    This is the recommended one-call API when you want the point estimate,
    standard error, confidence interval, fitted calibrator, and any automatic
    method-selection diagnostics together. Optional ``X`` and
    ``X_unlabeled`` are used by ``method="prognostic_linear"``.
    """
    state = _fit_mean_inference(
        Y,
        Yhat,
        Yhat_unlabeled,
        alpha=alpha,
        alternative=alternative,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
        inference=inference,
        n_resamples=n_resamples,
        random_state=random_state,
        compute_se=True,
        compute_ci=True,
    )
    if state.se is None or state.ci is None:
        raise RuntimeError("Internal error: mean inference did not compute uncertainty outputs.")
    return MeanInferenceResult(
        pointestimate=restore_shape(state.pointestimate, np.asarray(Y)),
        se=restore_shape(state.se, np.asarray(Y)),
        ci=(
            restore_shape(state.ci[0], np.asarray(Y)),
            restore_shape(state.ci[1], np.asarray(Y)),
        ),
        method=state.method,
        selected_candidate=state.selected_candidate,
        selected_efficiency_maximization=state.selected_efficiency_maximization,
        efficiency_lambda=state.efficiency_lambda,
        inference=state.inference,
        diagnostics=state.diagnostics,
        calibrator=state.calibrator,
    )


def aipw_mean_pointestimate(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    return_calibrator: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, CalibrationModel]:
    """Computes an AIPW-style point estimate for the population mean.

    The calling convention mirrors :mod:`ppi_py`: pass labeled outcomes ``Y``,
    labeled predictions ``Yhat``, and unlabeled predictions ``Yhat_unlabeled``.
    Set ``method`` to ``"aipw"``, ``"linear"``, ``"prognostic_linear"``,
    ``"sigmoid"``, ``"monotone_spline"``, or ``"isotonic"``.

    Args:
        Y:
            Gold-standard outcomes for the labeled sample. Shape ``(n_labeled,)``
            or ``(n_labeled, d)``.
        Yhat:
            Model predictions for the same labeled rows as ``Y``. Must have the
            same shape as ``Y``.
        Yhat_unlabeled:
            Model predictions for the unlabeled sample. Shape
            ``(n_unlabeled,)`` or ``(n_unlabeled, d)`` with the same output
            dimension as ``Y``.
        method:
            Calibration method applied before the AIPW augmentation step.
            Available choices are:

            - ``"aipw"``: use the raw predictions directly, with no calibration.
            - ``"linear"``: fit an affine map from predictions to outcomes.
            - ``"prognostic_linear"``: fit a semisupervised linear adjustment
              using the score plus optional extra covariates ``X``; the
              intercept and score coefficient are unpenalized, while the extra
              covariates are ridge-tuned on the labeled sample.
            - ``"sigmoid"``: fit a sigmoid calibration map after rescaling outcomes
              into the observed labeled range.
            - ``"monotone_spline"``: fit a smooth monotone spline calibration map.
            - ``"isotonic"``: fit a monotone isotonic calibration map.
            - ``"auto"``: choose among ``candidate_methods`` by cross-validated
              IF-variance minimization, while also comparing against an
              efficiency-maximized AIPW candidate when ``"aipw"`` is included.
        w:
            Optional sample weights for labeled data. If provided, must have
            length ``n_labeled``.
        w_unlabeled:
            Optional sample weights for unlabeled data. If provided, must have
            length ``n_unlabeled``.
        X:
            Optional extra covariates for the labeled sample. These are used by
            ``method="prognostic_linear"``.
        X_unlabeled:
            Optional extra covariates for the unlabeled sample. Must be passed
            together with ``X`` when using ``method="prognostic_linear"``.
        efficiency_maximization:
            If ``True``, rescales the predictor used by the estimator to
            ``lambda m(X)``, where ``m(X)`` is the raw score for
            ``method="aipw"`` and the calibrated score for the other methods.
            The coefficient ``lambda`` is chosen coordinatewise by empirical
            influence-function variance minimization. With ``method="aipw"``,
            this gives the unrestricted efficiency-maximized raw-score
            correction often associated with PPI++. Under ``method="auto"``,
            this flag does not change the method-choice stage; instead, after
            selection it applies a cross-fitted lambda to the selected method.
        candidate_methods:
            Candidate methods considered when ``method="auto"``. If
            ``"aipw"`` is included, the selector also evaluates an
            efficiency-maximized AIPW candidate.
        num_folds:
            Number of labeled-data folds used when ``method="auto"``. The
            default is ``100`` and is capped at the labeled sample size.
        auto_unlabeled_subsample_size:
            Unlabeled subset size used by ``method="auto"`` during foldwise
            selection and cross-fitted lambda estimation. The default is
            ``min(n_unlabeled, 10 * n_labeled)``.
        selection_random_state:
            Optional random seed or ``numpy.random.Generator`` controlling the
            fold split and unlabeled subsample when ``method="auto"``.
        isocal_backend:
            Backend used when ``method="isotonic"``. Choose ``"xgboost"`` or
            ``"sklearn"``. The default is ``"xgboost"``.
        isocal_max_depth:
            Maximum depth used by the one-round monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        isocal_min_child_weight:
            ``min_child_weight`` used by the monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        return_calibrator:
            If ``True``, also returns the fitted calibration model so it can be
            inspected or reused.

    Returns:
        The estimated population mean, with scalar or vector shape matching ``Y``.

        Examples:
        Basic usage::

            estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled)

        Use isotonic calibration::

            estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled, method="isotonic")

        Turn on empirical efficiency maximization for raw-score AIPW::

            estimate = aipw_mean_pointestimate(
                Y,
                Yhat,
                Yhat_unlabeled,
                method="aipw",
                efficiency_maximization=True,
            )

        Choose among AIPW, linear calibration, and isotonic calibration by CV::

            estimate = aipw_mean_pointestimate(
                Y,
                Yhat,
                Yhat_unlabeled,
                method="auto",
                candidate_methods=("aipw", "linear", "isotonic"),
            )
    """
    state = _fit_mean_inference(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
        inference="wald",
        compute_se=False,
        compute_ci=False,
    )
    estimate_out = restore_shape(state.pointestimate, np.asarray(Y))
    if return_calibrator:
        return estimate_out, state.calibrator
    return estimate_out


def aipw_mean_se(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> float | np.ndarray:
    """Computes a standard error for :func:`aipw_mean_pointestimate`.

    This uses the same arguments as :func:`aipw_mean_pointestimate` and returns a
    scalar or vector standard error matching the shape of ``Y``.

    Use ``inference="wald"`` for the analytic standard error or
    ``inference="bootstrap"`` to estimate the standard error from bootstrap
    resamples. When ``method="auto"`` and ``inference="wald"``, the point
    estimate uses the selected method refit on the full labeled sample, while
    any final lambda scaling is learned from the selected method's cross-fitted
    labeled predictions and unlabeled-subsample predictions and then reused
    for the Wald variance. When ``method="auto"`` and
    ``inference="bootstrap"``, the method is selected once on the original
    sample and bootstrap resamples refit only that chosen method.
    """

    state = _fit_mean_inference(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
        inference=inference,
        n_resamples=n_resamples,
        random_state=random_state,
        compute_se=True,
        compute_ci=False,
    )
    if state.se is None:
        raise RuntimeError("Internal error: mean SE computation did not produce a standard error.")
    return restore_shape(state.se, np.asarray(Y))


def aipw_mean_ci(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    X: np.ndarray | None = None,
    X_unlabeled: np.ndarray | None = None,
    efficiency_maximization: bool | None = None,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "isotonic"),
    num_folds: int | None = None,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Computes a confidence interval for the mean.

    Args:
        Y:
            Observed outcomes on the labeled sample.
        Yhat:
            Predictions for the labeled sample.
        Yhat_unlabeled:
            Predictions for the unlabeled sample.
        alpha:
            Miscoverage level. ``alpha=0.1`` gives a 90% interval and
            ``alpha=0.05`` gives a 95% interval.
        alternative:
            One of ``"two-sided"``, ``"larger"``, or ``"smaller"``.
        method:
            One of ``"aipw"``, ``"linear"``, ``"prognostic_linear"``, ``"sigmoid"``, ``"monotone_spline"``, ``"isotonic"``, or
            ``"auto"``. The automatic option chooses among
            ``candidate_methods`` by cross-validated IF-variance minimization,
            while also comparing against an efficiency-maximized AIPW candidate
            when ``"aipw"`` is included.
        w:
            Optional labeled-sample weights.
        w_unlabeled:
            Optional unlabeled-sample weights.
        efficiency_maximization:
            If ``True``, rescales the predictor used by the estimator to
            ``lambda m(X)``, where ``m(X)`` is the raw score for
            ``method="aipw"`` and the calibrated score for the other methods.
            The coefficient ``lambda`` is chosen coordinatewise by empirical
            influence-function variance minimization before computing the
            estimate and interval. Under ``method="auto"``, this flag is
            applied only after method selection, using a lambda learned from
            cross-fitted predictions.
        candidate_methods:
            Candidate methods considered when ``method="auto"``. If
            ``"aipw"`` is included, the selector also evaluates an
            efficiency-maximized AIPW candidate.
        num_folds:
            Number of labeled-data folds used when ``method="auto"``. The
            default is ``100`` and is capped at the labeled sample size.
        auto_unlabeled_subsample_size:
            Unlabeled subset size used by ``method="auto"`` during foldwise
            selection and cross-fitted lambda estimation. The default is
            ``min(n_unlabeled, 10 * n_labeled)``.
        selection_random_state:
            Optional random seed or ``numpy.random.Generator`` controlling the
            fold split and unlabeled subsample when ``method="auto"``.
        isocal_backend:
            Backend used when ``method="isotonic"``. Choose ``"xgboost"`` or
            ``"sklearn"``. The default is ``"xgboost"``.
        isocal_max_depth:
            Maximum depth used by the one-round monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        isocal_min_child_weight:
            ``min_child_weight`` used by the monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        inference:
            ``"wald"`` for the analytic normal-approximation interval or
            ``"bootstrap"`` for a percentile bootstrap interval that resamples
            labeled and unlabeled rows and refits the calibration step in each
            resample while holding the prediction model fixed.
        n_resamples:
            Number of bootstrap resamples when ``inference="bootstrap"``.
        random_state:
            Optional random seed or ``numpy.random.Generator`` for bootstrap
            reproducibility.

    Returns:
        A pair ``(lower, upper)`` with the same scalar or vector shape as ``Y``.

    Notes:
        ``inference="wald"`` returns the analytic normal-approximation interval.
        When ``method="auto"``, the point estimate is computed from the selected
        method refit on the full labeled sample, while any final lambda scaling
        is learned from the selected method's cross-fitted labeled predictions
        and unlabeled-subsample predictions and reused for the standard
        error.
        ``inference="bootstrap"`` returns a percentile bootstrap interval.
        When ``method="auto"``, bootstrap selects the method once on the
        original sample and then bootstraps only that chosen method.
    """

    state = _fit_mean_inference(
        Y,
        Yhat,
        Yhat_unlabeled,
        alpha=alpha,
        alternative=alternative,
        method=method,
        w=w,
        w_unlabeled=w_unlabeled,
        X=X,
        X_unlabeled=X_unlabeled,
        efficiency_maximization=efficiency_maximization,
        candidate_methods=candidate_methods,
        num_folds=num_folds,
        auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
        selection_random_state=selection_random_state,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
        inference=inference,
        n_resamples=n_resamples,
        random_state=random_state,
        compute_se=True,
        compute_ci=True,
    )
    if state.ci is None:
        raise RuntimeError("Internal error: mean CI computation did not produce an interval.")
    return restore_shape(state.ci[0], np.asarray(Y)), restore_shape(state.ci[1], np.asarray(Y))


def linear_calibration_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="linear"``."""

    kwargs.setdefault("method", "linear")
    return aipw_mean_pointestimate(*args, **kwargs)


def linear_calibration_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="linear"``."""

    kwargs.setdefault("method", "linear")
    return aipw_mean_ci(*args, **kwargs)


def sigmoid_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="sigmoid"``."""

    kwargs.setdefault("method", "sigmoid")
    return aipw_mean_pointestimate(*args, **kwargs)


def sigmoid_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="sigmoid"``."""

    kwargs.setdefault("method", "sigmoid")
    return aipw_mean_ci(*args, **kwargs)


def isotonic_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="isotonic"``."""

    kwargs.setdefault("method", "isotonic")
    return aipw_mean_pointestimate(*args, **kwargs)


def isotonic_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="isotonic"``."""

    kwargs.setdefault("method", "isotonic")
    return aipw_mean_ci(*args, **kwargs)


def platt_scaling_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Legacy shortcut for :func:`aipw_mean_pointestimate` with ``method="sigmoid"``."""

    kwargs.setdefault("method", "sigmoid")
    return aipw_mean_pointestimate(*args, **kwargs)


def platt_scaling_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Legacy shortcut for :func:`aipw_mean_ci` with ``method="sigmoid"``."""

    kwargs.setdefault("method", "sigmoid")
    return aipw_mean_ci(*args, **kwargs)


def isocal_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Legacy shortcut for :func:`aipw_mean_pointestimate` with ``method="isotonic"``."""

    kwargs.setdefault("method", "isotonic")
    return aipw_mean_pointestimate(*args, **kwargs)


def isocal_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Legacy shortcut for :func:`aipw_mean_ci` with ``method="isotonic"``."""

    kwargs.setdefault("method", "isotonic")
    return aipw_mean_ci(*args, **kwargs)


mean_pointestimate = aipw_mean_pointestimate
mean_ci = aipw_mean_ci
mean_se = aipw_mean_se
mean_inference = aipw_mean_inference
ppi_aipw_mean_pointestimate = aipw_mean_pointestimate
ppi_aipw_mean_ci = aipw_mean_ci
ppi_aipw_mean_inference = aipw_mean_inference
pi_aipw_mean_pointestimate = aipw_mean_pointestimate
pi_aipw_mean_ci = aipw_mean_ci
pi_aipw_mean_inference = aipw_mean_inference
