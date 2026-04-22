from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear, minimize
from sklearn.isotonic import IsotonicRegression
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised only when xgboost is unavailable
    xgb = None

from ._utils import construct_weight_vector, restore_shape, validate_pair_inputs


METHOD_ALIASES = {
    "aipw": "aipw",
    "identity": "aipw",
    "none": "aipw",
    "linear": "linear",
    "linear_calibration": "linear",
    "prognostic_linear": "prognostic_linear",
    "linear_adjustment": "prognostic_linear",
    "prognostic": "prognostic_linear",
    "sigmoid": "sigmoid",
    "platt": "sigmoid",
    "platt_scaling": "sigmoid",
    "isotonic": "isotonic",
    "isocal": "isotonic",
    "isotonic_calibration": "isotonic",
    "monotone_spline": "monotone_spline",
    "isotonic_spline": "monotone_spline",
    "smooth_spline": "monotone_spline",
    "mspline": "monotone_spline",
}


_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS = 6
_MONOTONE_SPLINE_DERIVATIVE_DEGREE = 2
_MONOTONE_SPLINE_PENALTY = 1e-3


def canonical_method(method: str) -> str:
    try:
        return METHOD_ALIASES[method.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown calibration method '{method}'. Expected one of: {choices}.") from exc


def clip_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def clip_range(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), float(lower), float(upper))


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = clip_unit(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


@dataclass
class _CoordinateCalibrator:
    method: str
    fitted: Any
    y_min: float
    y_max: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)

        if self.method == "aipw":
            return scores

        if np.isclose(self.y_max, self.y_min):
            return np.full_like(scores, self.y_min, dtype=float)

        if self.method == "linear":
            pred = self.fitted["slope"] * scores + self.fitted["intercept"]
            return clip_range(pred, self.y_min, self.y_max)

        if self.method == "sigmoid":
            scores_scaled = clip_unit((scores - self.y_min) / (self.y_max - self.y_min))
            logits = safe_logit(scores_scaled)
            calibrated = sigmoid(self.fitted["slope"] * logits + self.fitted["intercept"])
            pred = self.y_min + (self.y_max - self.y_min) * calibrated
            return clip_range(pred, self.y_min, self.y_max)

        if self.method == "isotonic":
            pred = self.fitted.predict(scores)
            return clip_range(pred, self.y_min, self.y_max)

        if self.method == "monotone_spline":
            fitted = self.fitted
            score_min = float(fitted["score_min"])
            score_scale = float(fitted["score_scale"])
            if np.isclose(score_scale, 0.0):
                return np.full_like(scores, float(fitted["intercept"]), dtype=float)
            z = np.clip((scores - score_min) / score_scale, 0.0, 1.0)
            pred = _evaluate_monotone_spline(
                z,
                knots=np.asarray(fitted["knots"], dtype=float),
                coef=np.asarray(fitted["coef"], dtype=float),
                degree=int(fitted["basis_degree"]),
                intercept=float(fitted["intercept"]),
            )
            return clip_range(pred, self.y_min, self.y_max)

        raise ValueError(f"Unsupported calibrator method '{self.method}'.")


@dataclass
class _XGBoostMonotoneRegressor:
    booster: Any

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if xgb is None:  # pragma: no cover - guarded during fitting as well
            raise ImportError(
                "xgboost is required for isocal_backend='xgboost'. "
                "Install xgboost or switch to isocal_backend='sklearn'."
            )
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        return np.asarray(self.booster.predict(xgb.DMatrix(data=scores)), dtype=float)


@dataclass
class CalibrationModel:
    method: str
    calibrators: list[_CoordinateCalibrator]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_outputs = len(self.calibrators)
        parts = [f"method={self.method!r}", f"n_outputs={n_outputs}"]
        backend = self.metadata.get("isocal_backend")
        if backend:
            parts.append(f"isocal_backend={backend!r}")
        lambda_source = self.metadata.get("efficiency_lambda_source")
        if lambda_source:
            parts.append(f"efficiency_lambda_source={lambda_source!r}")
        return f"CalibrationModel({', '.join(parts)})"

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores_2d = scores.reshape(-1, 1) if scores.ndim == 1 else scores.copy()
        if scores_2d.shape[1] != len(self.calibrators):
            raise ValueError(
                f"Expected {len(self.calibrators)} columns, got {scores_2d.shape[1]}."
            )

        pred = np.column_stack(
            [calibrator.predict(scores_2d[:, idx]) for idx, calibrator in enumerate(self.calibrators)]
        )
        return restore_shape(pred, scores)


def _fit_linear_coordinate(
    y: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
) -> _CoordinateCalibrator:
    design = np.column_stack([scores, np.ones_like(scores)])
    sqrt_weight = np.sqrt(sample_weight)
    coef, _, _, _ = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        y * sqrt_weight,
        rcond=None,
    )
    return _CoordinateCalibrator(
        method="linear",
        fitted={"slope": float(coef[0]), "intercept": float(coef[1])},
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
    )


def _fit_platt_coordinate(
    y: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
) -> _CoordinateCalibrator:
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if np.isclose(y_max, y_min):
        return _CoordinateCalibrator(
            method="sigmoid",
            fitted={"slope": 0.0, "intercept": 0.0},
            y_min=y_min,
            y_max=y_max,
        )

    y_scaled = clip_unit((y - y_min) / (y_max - y_min))
    scores_scaled = clip_unit((scores - y_min) / (y_max - y_min))
    logits = safe_logit(scores_scaled)

    mean_y = float(np.average(y_scaled, weights=sample_weight))
    start = np.array([1.0, float(safe_logit(np.array([mean_y]))[0])], dtype=float)

    if np.allclose(logits, logits[0]):
        return _CoordinateCalibrator(
            method="sigmoid",
            fitted={"slope": 0.0, "intercept": float(start[1])},
            y_min=y_min,
            y_max=y_max,
        )

    sample_weight = sample_weight / sample_weight.sum()

    def objective(beta: np.ndarray) -> float:
        p = clip_unit(sigmoid(beta[0] * logits + beta[1]))
        loss = -(y_scaled * np.log(p) + (1.0 - y_scaled) * np.log(1.0 - p))
        return float(np.sum(sample_weight * loss))

    result = minimize(objective, x0=start, method="BFGS")
    beta = result.x if result.success and np.all(np.isfinite(result.x)) else start

    return _CoordinateCalibrator(
        method="sigmoid",
        fitted={"slope": float(beta[0]), "intercept": float(beta[1])},
        y_min=y_min,
        y_max=y_max,
    )


def _fit_isocal_coordinate(
    y: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    *,
    backend: str,
    max_depth: int,
    min_child_weight: float,
) -> _CoordinateCalibrator:
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if np.allclose(scores, scores[0]) or np.isclose(y_max, y_min):
        mean_y = float(np.average(y, weights=sample_weight))
        return _CoordinateCalibrator(
            method="linear",
            fitted={"slope": 0.0, "intercept": mean_y},
            y_min=y_min,
            y_max=y_max,
            metadata={"fallback": 1.0},
        )

    if backend == "sklearn":
        model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True, out_of_bounds="clip")
        model.fit(scores, y, sample_weight=sample_weight)
    elif backend == "xgboost":
        if xgb is None:
            raise ImportError(
                "xgboost is required for isocal_backend='xgboost'. "
                "Install xgboost or switch to isocal_backend='sklearn'."
            )
        data = xgb.DMatrix(
            data=np.asarray(scores, dtype=float).reshape(-1, 1),
            label=np.asarray(y, dtype=float).reshape(-1),
            weight=np.asarray(sample_weight, dtype=float),
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
        model = _XGBoostMonotoneRegressor(
            booster=xgb.train(params=params, dtrain=data, num_boost_round=1)
        )
    else:
        raise ValueError("isocal_backend must be either 'xgboost' or 'sklearn'.")
    return _CoordinateCalibrator(
        method="isotonic",
        fitted=model,
        y_min=y_min,
        y_max=y_max,
        metadata={
            "backend": backend,
            "max_depth": float(max_depth),
            "min_child_weight": float(min_child_weight),
        },
    )


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


def _fit_monotone_spline_coordinate(
    y: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
) -> _CoordinateCalibrator:
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if np.allclose(scores, scores[0]) or np.isclose(y_max, y_min):
        mean_y = float(np.average(y, weights=sample_weight))
        return _CoordinateCalibrator(
            method="linear",
            fitted={"slope": 0.0, "intercept": mean_y},
            y_min=y_min,
            y_max=y_max,
            metadata={"fallback": 1.0},
        )

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_scale = float(score_max - score_min)
    if np.isclose(score_scale, 0.0) or np.unique(scores).size < 4:
        return _fit_linear_coordinate(y, scores, sample_weight)

    scores_scaled = np.clip((scores - score_min) / score_scale, 0.0, 1.0)
    basis_degree = _MONOTONE_SPLINE_DERIVATIVE_DEGREE
    knots = _choose_monotone_spline_knots(
        scores_scaled,
        max_internal_knots=_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS,
        degree=basis_degree,
    )
    basis = _integrated_bspline_design(scores_scaled, knots=knots, degree=basis_degree)

    design = np.column_stack([np.ones_like(scores_scaled), basis])
    sqrt_weight = np.sqrt(np.asarray(sample_weight, dtype=float))
    weighted_design = design * sqrt_weight[:, None]
    weighted_response = y * sqrt_weight

    penalty_block = _second_difference_penalty(basis.shape[1])
    penalty_design = np.column_stack(
        [
            np.zeros((penalty_block.shape[0], 1), dtype=float),
            penalty_block,
        ]
    )
    augmented_design = np.vstack(
        [
            weighted_design,
            np.sqrt(_MONOTONE_SPLINE_PENALTY) * penalty_design,
        ]
    )
    augmented_response = np.concatenate(
        [
            weighted_response,
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
        return _fit_linear_coordinate(y, scores, sample_weight)

    return _CoordinateCalibrator(
        method="monotone_spline",
        fitted={
            "intercept": float(result.x[0]),
            "coef": np.asarray(result.x[1:], dtype=float),
            "knots": np.asarray(knots, dtype=float),
            "basis_degree": float(basis_degree),
            "score_min": score_min,
            "score_scale": score_scale,
        },
        y_min=y_min,
        y_max=y_max,
        metadata={
            "max_internal_knots": float(_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS),
            "basis_degree": float(basis_degree + 1),
            "penalty": float(_MONOTONE_SPLINE_PENALTY),
        },
    )


def fit_calibrator(
    Y: np.ndarray,
    Yhat: np.ndarray,
    *,
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
) -> CalibrationModel:
    """Fits a calibrator from labeled outcomes and predictions.

    Args:
        Y:
            Observed outcomes on the labeled sample.
        Yhat:
            Predictions for the same labeled rows.
        method:
            One of:

            - ``"aipw"``: no calibration, just pass predictions through.
            - ``"linear"``: fit a straight-line calibration map.
            - ``"sigmoid"``: fit a sigmoid-shaped calibration map.
            - ``"isotonic"``: fit a monotone isotonic calibration map.
            - ``"monotone_spline"``: fit a smooth monotone spline calibration map.
        w:
            Optional sample weights for the labeled sample.
        isocal_backend:
            Backend used when ``method="isotonic"``. Choose ``"xgboost"`` or
            ``"sklearn"``. The default is ``"xgboost"``.
        isocal_max_depth:
            Maximum depth used by the one-round monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        isocal_min_child_weight:
            ``min_child_weight`` used by the monotone XGBoost fit when
            ``isocal_backend="xgboost"``.

    Returns:
        A :class:`CalibrationModel` that can be reused on new predictions.
    """

    method = canonical_method(method)
    if method == "prognostic_linear":
        raise ValueError(
            "method='prognostic_linear' requires optional covariates X/X_unlabeled "
            "and is available through mean_inference(...) and causal_inference(...), "
            "not fit_calibrator(...)."
        )
    isocal_backend = isocal_backend.lower()
    Y_2d, Yhat_2d = validate_pair_inputs(Y, Yhat)
    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=False)

    calibrators: list[_CoordinateCalibrator] = []
    for idx in range(Y_2d.shape[1]):
        y_coord = Y_2d[:, idx]
        score_coord = Yhat_2d[:, idx]

        if method == "aipw":
            calibrator = _CoordinateCalibrator(
                method="aipw",
                fitted=None,
                y_min=float(np.min(y_coord)),
                y_max=float(np.max(y_coord)),
            )
        elif method == "linear":
            calibrator = _fit_linear_coordinate(y_coord, score_coord, weights)
        elif method == "sigmoid":
            calibrator = _fit_platt_coordinate(y_coord, score_coord, weights)
        elif method == "isotonic":
            calibrator = _fit_isocal_coordinate(
                y_coord,
                score_coord,
                weights,
                backend=isocal_backend,
                max_depth=isocal_max_depth,
                min_child_weight=isocal_min_child_weight,
            )
        elif method == "monotone_spline":
            calibrator = _fit_monotone_spline_coordinate(y_coord, score_coord, weights)
        else:
            raise ValueError(f"Unsupported method '{method}'.")

        calibrators.append(calibrator)

    return CalibrationModel(
        method=method,
        calibrators=calibrators,
        metadata={
            "n_outputs": Y_2d.shape[1],
            "isocal_backend": isocal_backend if method == "isotonic" else None,
        },
    )


def calibrate_predictions(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray | None = None,
    *,
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    return_model: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray | None, CalibrationModel]:
    """Calibrates labeled and unlabeled predictions in one call.

    Args:
        Y:
            Observed outcomes on the labeled sample.
        Yhat:
            Predictions on the labeled sample.
        Yhat_unlabeled:
            Optional predictions on the unlabeled sample.
        method:
            Calibration method to fit. Choose from ``"aipw"``, ``"linear"``,
            ``"sigmoid"``, ``"isotonic"``, or ``"monotone_spline"``.
        w:
            Optional labeled-sample weights.
        isocal_backend:
            Backend used when ``method="isotonic"``. Choose ``"xgboost"`` or
            ``"sklearn"``. The default is ``"xgboost"``.
        isocal_max_depth:
            Maximum depth used by the one-round monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        isocal_min_child_weight:
            ``min_child_weight`` used by the monotone XGBoost fit when
            ``isocal_backend="xgboost"``.
        return_model:
            If ``True``, also returns the fitted :class:`CalibrationModel`.

    Returns:
        Calibrated labeled predictions, calibrated unlabeled predictions, and
        optionally the fitted model.
    """

    model = fit_calibrator(
        Y,
        Yhat,
        method=method,
        w=w,
        isocal_backend=isocal_backend,
        isocal_max_depth=isocal_max_depth,
        isocal_min_child_weight=isocal_min_child_weight,
    )
    pred_labeled = model.predict(Yhat)
    pred_unlabeled = None if Yhat_unlabeled is None else model.predict(Yhat_unlabeled)

    if return_model:
        return pred_labeled, pred_unlabeled, model
    return pred_labeled, pred_unlabeled
