from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from ._utils import construct_weight_vector, restore_shape, validate_pair_inputs


METHOD_ALIASES = {
    "aipw": "aipw",
    "identity": "aipw",
    "none": "aipw",
    "linear": "linear",
    "linear_calibration": "linear",
    "platt": "platt",
    "platt_scaling": "platt",
    "isocal": "isocal",
    "isotonic": "isocal",
    "isotonic_calibration": "isocal",
}


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
    metadata: dict[str, float] = field(default_factory=dict)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)

        if self.method == "aipw":
            return scores

        if np.isclose(self.y_max, self.y_min):
            return np.full_like(scores, self.y_min, dtype=float)

        if self.method == "linear":
            pred = self.fitted["slope"] * scores + self.fitted["intercept"]
            return clip_range(pred, self.y_min, self.y_max)

        if self.method == "platt":
            scores_scaled = clip_unit((scores - self.y_min) / (self.y_max - self.y_min))
            logits = safe_logit(scores_scaled)
            calibrated = sigmoid(self.fitted["slope"] * logits + self.fitted["intercept"])
            pred = self.y_min + (self.y_max - self.y_min) * calibrated
            return clip_range(pred, self.y_min, self.y_max)

        if self.method == "isocal":
            pred = self.fitted.predict(scores)
            return clip_range(pred, self.y_min, self.y_max)

        raise ValueError(f"Unsupported calibrator method '{self.method}'.")


@dataclass
class CalibrationModel:
    method: str
    calibrators: list[_CoordinateCalibrator]
    metadata: dict[str, Any] = field(default_factory=dict)

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
            method="platt",
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
            method="platt",
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
        method="platt",
        fitted={"slope": float(beta[0]), "intercept": float(beta[1])},
        y_min=y_min,
        y_max=y_max,
    )


def _fit_isocal_coordinate(
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

    model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True, out_of_bounds="clip")
    model.fit(scores, y, sample_weight=sample_weight)
    return _CoordinateCalibrator(
        method="isocal",
        fitted=model,
        y_min=y_min,
        y_max=y_max,
    )


def fit_calibrator(
    Y: np.ndarray,
    Yhat: np.ndarray,
    *,
    method: str = "linear",
    w: np.ndarray | None = None,
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
            - ``"linear"``: fit a straight-line recalibration map.
            - ``"platt"``: fit a sigmoid-shaped calibration map.
            - ``"isocal"``: fit a monotone isotonic calibration map.
        w:
            Optional sample weights for the labeled sample.

    Returns:
        A :class:`CalibrationModel` that can be reused on new predictions.
    """

    method = canonical_method(method)
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
        elif method == "platt":
            calibrator = _fit_platt_coordinate(y_coord, score_coord, weights)
        elif method == "isocal":
            calibrator = _fit_isocal_coordinate(y_coord, score_coord, weights)
        else:
            raise ValueError(f"Unsupported method '{method}'.")

        calibrators.append(calibrator)

    return CalibrationModel(
        method=method,
        calibrators=calibrators,
        metadata={"n_outputs": Y_2d.shape[1]},
    )


def calibrate_predictions(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray | None = None,
    *,
    method: str = "linear",
    w: np.ndarray | None = None,
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
            ``"platt"``, or ``"isocal"``.
        w:
            Optional labeled-sample weights.
        return_model:
            If ``True``, also returns the fitted :class:`CalibrationModel`.

    Returns:
        Calibrated labeled predictions, calibrated unlabeled predictions, and
        optionally the fitted model.
    """

    model = fit_calibrator(Y, Yhat, method=method, w=w)
    pred_labeled = model.predict(Yhat)
    pred_unlabeled = None if Yhat_unlabeled is None else model.predict(Yhat_unlabeled)

    if return_model:
        return pred_labeled, pred_unlabeled, model
    return pred_labeled, pred_unlabeled
