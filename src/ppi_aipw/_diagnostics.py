from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ._api import (
    MeanInferenceResult,
    PrognosticLinearModel,
    _compute_wald_statistics,
    _fit_and_calibrate,
    _fit_prognostic_linear,
    _summary_value,
)
from ._calibration import CalibrationModel
from ._utils import construct_weight_vector, reshape_to_2d, validate_pair_inputs, z_interval


@dataclass(frozen=True)
class CalibrationBLPDiagnostics:
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    intercept_ci: tuple[float, float]
    slope_ci: tuple[float, float]
    slope_null: float
    slope_wald_t: float
    slope_p_value: float


@dataclass(frozen=True)
class CalibrationCurveDiagnostics:
    raw_labeled_scores: np.ndarray
    calibrated_labeled_scores: np.ndarray
    observed_outcomes: np.ndarray
    bin_centers: np.ndarray
    bin_mean_raw_score: np.ndarray
    bin_mean_calibrated_score: np.ndarray
    bin_mean_outcome: np.ndarray
    bin_counts: np.ndarray
    grid_scores: np.ndarray
    fitted_curve: np.ndarray
    blp: CalibrationBLPDiagnostics


@dataclass(frozen=True)
class CalibrationDiagnostics:
    method: str
    diagnostic_mode: str
    n_outputs: int
    n_labeled: int
    num_bins: int
    per_output: tuple[CalibrationCurveDiagnostics, ...]
    effective_num_folds: int | None = None
    reference_covariates: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            "CalibrationDiagnostics("
            f"method={self.method!r}, "
            f"diagnostic_mode={self.diagnostic_mode!r}, "
            f"n_outputs={self.n_outputs}, "
            f"n_labeled={self.n_labeled}, "
            f"num_bins={self.num_bins}"
            ")"
        )

    def summary(
        self,
        *,
        null: float = 1.0,
        alternative: str = "two-sided",
    ) -> str:
        null_arr = np.asarray(null, dtype=float).reshape(-1)
        if null_arr.size != 1:
            raise ValueError("CalibrationDiagnostics.summary currently expects a scalar null.")

        lines = [
            "CalibrationDiagnostics summary",
            f"method: {self.method}",
            f"diagnostic_mode: {self.diagnostic_mode}",
            f"n_outputs: {self.n_outputs}",
            f"n_labeled: {self.n_labeled}",
            f"num_bins: {self.num_bins}",
            f"blp_slope_null: {_summary_value(float(null_arr[0]))}",
            f"blp_slope_alternative: {alternative}",
        ]
        if self.effective_num_folds is not None:
            lines.append(f"effective_num_folds: {self.effective_num_folds}")

        for idx, record in enumerate(self.per_output):
            _, wald_t, p_value = _compute_wald_statistics(
                record.blp.slope,
                record.blp.slope_se,
                null=float(null_arr[0]),
                alternative=alternative,
            )
            lower, upper = z_interval(
                np.array([record.blp.slope], dtype=float),
                np.array([record.blp.slope_se], dtype=float),
                alpha=0.05,
                alternative=alternative,
            )
            prefix = "calibrated_blp_slope" if self.n_outputs == 1 else f"output[{idx}] calibrated_blp_slope"
            lines.append(
                (
                    f"{prefix}: estimate={_summary_value(record.blp.slope)}, "
                    f"se={_summary_value(record.blp.slope_se)}, "
                    f"ci=({_summary_value(float(lower[0]))}, {_summary_value(float(upper[0]))}), "
                    f"wald_t={_summary_value(float(wald_t[0]))}, "
                    f"p_value={_summary_value(float(p_value[0]))}"
                )
            )
        return "\n".join(lines)


def _coerce_covariates(
    X: np.ndarray | None,
    *,
    n_obs: int,
) -> np.ndarray | None:
    if X is None:
        return None
    X_2d = reshape_to_2d(np.asarray(X, dtype=float), name="X")
    if X_2d.shape[0] != n_obs:
        raise ValueError(f"X must have {n_obs} rows, got {X_2d.shape[0]}.")
    return X_2d


def _resolve_model(
    obj: MeanInferenceResult | CalibrationModel | PrognosticLinearModel,
) -> CalibrationModel | PrognosticLinearModel:
    if isinstance(obj, MeanInferenceResult):
        return obj.calibrator
    if isinstance(obj, (CalibrationModel, PrognosticLinearModel)):
        return obj
    raise TypeError(
        "Expected a MeanInferenceResult, CalibrationModel, or PrognosticLinearModel."
    )


def _weighted_average(x: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(weights, dtype=float)))


def _normalize_diagnostic_mode(diagnostic_mode: str) -> str:
    mode = diagnostic_mode.lower().replace("-", "_")
    aliases = {
        "out_of_fold": "out_of_fold",
        "oof": "out_of_fold",
        "in_sample": "in_sample",
        "insample": "in_sample",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError("diagnostic_mode must be 'out_of_fold' or 'in_sample'.") from exc


def _fit_blp_diagnostics(
    outcomes: np.ndarray,
    calibrated_scores: np.ndarray,
    weights: np.ndarray,
) -> CalibrationBLPDiagnostics:
    outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
    calibrated_scores = np.asarray(calibrated_scores, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    predictor_centered = calibrated_scores - np.average(calibrated_scores, weights=weights)
    if np.all(np.isclose(predictor_centered, 0.0)):
        intercept = float(np.average(outcomes, weights=weights))
        design_intercept = np.ones((outcomes.shape[0], 1), dtype=float)
        xtwx = design_intercept.T @ (weights[:, None] * design_intercept)
        resid = outcomes - intercept
        meat = (design_intercept * (weights * resid)[:, None]).T @ (
            design_intercept * (weights * resid)[:, None]
        )
        intercept_var = float((np.linalg.pinv(xtwx) @ meat @ np.linalg.pinv(xtwx))[0, 0])
        intercept_se = float(np.sqrt(max(intercept_var, 0.0)))
        intercept_ci = z_interval(
            np.array([intercept], dtype=float),
            np.array([intercept_se], dtype=float),
            alpha=0.05,
            alternative="two-sided",
        )
        return CalibrationBLPDiagnostics(
            intercept=intercept,
            slope=float("nan"),
            intercept_se=intercept_se,
            slope_se=float("nan"),
            intercept_ci=(float(intercept_ci[0][0]), float(intercept_ci[1][0])),
            slope_ci=(float("nan"), float("nan")),
            slope_null=1.0,
            slope_wald_t=float("nan"),
            slope_p_value=float("nan"),
        )

    design = np.column_stack([np.ones(outcomes.shape[0]), calibrated_scores])
    sqrt_weight = np.sqrt(weights)
    coef, _, _, _ = np.linalg.lstsq(design * sqrt_weight[:, None], outcomes * sqrt_weight, rcond=None)
    resid = outcomes - design @ coef
    xtwx = design.T @ (weights[:, None] * design)
    xtwx_inv = np.linalg.pinv(xtwx)
    meat = (design * (weights * resid)[:, None]).T @ (design * (weights * resid)[:, None])
    covariance = xtwx_inv @ meat @ xtwx_inv
    variance = np.maximum(np.diag(covariance), 0.0)
    se = np.sqrt(variance)
    lower, upper = z_interval(coef, se, alpha=0.05, alternative="two-sided")
    _, slope_t, slope_p_value = _compute_wald_statistics(
        coef[1],
        se[1],
        null=1.0,
        alternative="two-sided",
    )

    return CalibrationBLPDiagnostics(
        intercept=float(coef[0]),
        slope=float(coef[1]),
        intercept_se=float(se[0]),
        slope_se=float(se[1]),
        intercept_ci=(float(lower[0]), float(upper[0])),
        slope_ci=(float(lower[1]), float(upper[1])),
        slope_null=1.0,
        slope_wald_t=float(slope_t[0]),
        slope_p_value=float(slope_p_value[0]),
    )


def _bin_diagnostics(
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    *,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(raw_scores, kind="mergesort")
    groups = [group for group in np.array_split(order, min(num_bins, raw_scores.shape[0])) if group.size > 0]

    bin_centers = []
    bin_mean_raw = []
    bin_mean_calibrated = []
    bin_mean_outcome = []
    bin_counts = []

    for group in groups:
        scores_group = raw_scores[group]
        calibrated_group = calibrated_scores[group]
        outcomes_group = outcomes[group]
        weights_group = weights[group]
        bin_centers.append(0.5 * (float(np.min(scores_group)) + float(np.max(scores_group))))
        bin_mean_raw.append(_weighted_average(scores_group, weights_group))
        bin_mean_calibrated.append(_weighted_average(calibrated_group, weights_group))
        bin_mean_outcome.append(_weighted_average(outcomes_group, weights_group))
        bin_counts.append(int(group.size))

    return (
        np.asarray(bin_centers, dtype=float),
        np.asarray(bin_mean_raw, dtype=float),
        np.asarray(bin_mean_calibrated, dtype=float),
        np.asarray(bin_mean_outcome, dtype=float),
        np.asarray(bin_counts, dtype=int),
    )


def calibration_diagnostics(
    obj: MeanInferenceResult | CalibrationModel | PrognosticLinearModel,
    Y: np.ndarray,
    Yhat: np.ndarray,
    *,
    X: np.ndarray | None = None,
    w: np.ndarray | None = None,
    diagnostic_mode: str = "out_of_fold",
    num_folds: int = 10,
    num_bins: int = 10,
) -> CalibrationDiagnostics:
    """Builds calibration diagnostics for a fitted result or model.

    By default, diagnostics use out-of-fold labeled predictions for a more
    honest calibration check. Set ``diagnostic_mode="in_sample"`` if you want
    a purely descriptive fit-on-fit view instead.
    """

    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    model = _resolve_model(obj)
    diagnostic_mode = _normalize_diagnostic_mode(diagnostic_mode)
    Y_2d, Yhat_2d = validate_pair_inputs(Y, Yhat)
    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=False)
    X_2d = _coerce_covariates(X, n_obs=Y_2d.shape[0])

    if isinstance(model, PrognosticLinearModel) and model.x_dim > 0 and X_2d is None:
        raise ValueError(
            "X is required for calibration diagnostics when the fitted model uses prognostic covariates."
        )

    def _predict_with_model(current_model: CalibrationModel | PrognosticLinearModel) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(current_model, PrognosticLinearModel):
            if current_model.x_dim > 0:
                assert X_2d is not None
                reference_covariates_local = np.average(X_2d, axis=0, weights=weights)
                calibrated_labeled_local = np.asarray(current_model.predict(Yhat_2d, X=X_2d), dtype=float)
            else:
                reference_covariates_local = None
                calibrated_labeled_local = np.asarray(current_model.predict(Yhat_2d), dtype=float)
        else:
            reference_covariates_local = None
            calibrated_labeled_local = np.asarray(current_model.predict(Yhat_2d), dtype=float)
        return calibrated_labeled_local, reference_covariates_local

    if diagnostic_mode == "in_sample":
        calibrated_labeled, reference_covariates = _predict_with_model(model)
        effective_num_folds = None
    else:
        n_splits = min(int(num_folds), Y_2d.shape[0])
        if n_splits < 2:
            raise ValueError(
                "out_of_fold calibration diagnostics require at least two labeled observations."
            )
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        calibrated_labeled = np.zeros_like(Y_2d, dtype=float)
        if isinstance(model, PrognosticLinearModel) and model.x_dim > 0:
            assert X_2d is not None
            reference_covariates = np.average(X_2d, axis=0, weights=weights)
        else:
            reference_covariates = None

        for train_idx, val_idx in splitter.split(Y_2d):
            train_weight = None if w is None else np.asarray(w, dtype=float).reshape(-1)[train_idx]
            if isinstance(model, PrognosticLinearModel):
                _, _, _, pred_val, _ = _fit_prognostic_linear(
                    Y_2d[train_idx],
                    Yhat_2d[train_idx],
                    Yhat_2d[val_idx],
                    X=None if X_2d is None else X_2d[train_idx],
                    X_unlabeled=None if X_2d is None else X_2d[val_idx],
                    w=train_weight,
                )
            else:
                _, _, _, pred_val, _ = _fit_and_calibrate(
                    Y_2d[train_idx],
                    Yhat_2d[train_idx],
                    Yhat_2d[val_idx],
                    method=model.method,
                    w=train_weight,
                    X=None,
                    X_unlabeled=None,
                    isocal_backend=str(model.metadata.get("isocal_backend") or "xgboost"),
                    isocal_max_depth=int(model.metadata.get("isocal_max_depth", 20)),
                    isocal_min_child_weight=float(model.metadata.get("isocal_min_child_weight", 10.0)),
                )
            calibrated_labeled[val_idx] = np.asarray(pred_val, dtype=float)
        effective_num_folds = n_splits

    per_output: list[CalibrationCurveDiagnostics] = []
    for idx in range(Y_2d.shape[1]):
        raw_scores = np.asarray(Yhat_2d[:, idx], dtype=float)
        calibrated_scores = np.asarray(calibrated_labeled[:, idx], dtype=float)
        outcomes = np.asarray(Y_2d[:, idx], dtype=float)
        bin_centers, bin_mean_raw, bin_mean_calibrated, bin_mean_outcome, bin_counts = _bin_diagnostics(
            raw_scores,
            calibrated_scores,
            outcomes,
            weights,
            num_bins=num_bins,
        )

        grid_scores = np.linspace(float(np.min(raw_scores)), float(np.max(raw_scores)), 200)
        if np.isclose(grid_scores[0], grid_scores[-1]):
            grid_scores = np.array([grid_scores[0]], dtype=float)

        if isinstance(model, PrognosticLinearModel):
            if model.x_dim > 0:
                assert reference_covariates is not None
                X_grid = np.repeat(reference_covariates.reshape(1, -1), grid_scores.shape[0], axis=0)
                design = np.column_stack([np.ones(grid_scores.shape[0]), grid_scores, X_grid])
            else:
                design = np.column_stack([np.ones(grid_scores.shape[0]), grid_scores])
            fitted_curve = np.asarray(design @ model.coefficients[idx], dtype=float)
        else:
            fitted_curve = np.asarray(model.calibrators[idx].predict(grid_scores), dtype=float)

        per_output.append(
            CalibrationCurveDiagnostics(
                raw_labeled_scores=raw_scores,
                calibrated_labeled_scores=calibrated_scores,
                observed_outcomes=outcomes,
                bin_centers=bin_centers,
                bin_mean_raw_score=bin_mean_raw,
                bin_mean_calibrated_score=bin_mean_calibrated,
                bin_mean_outcome=bin_mean_outcome,
                bin_counts=bin_counts,
                grid_scores=grid_scores,
                fitted_curve=fitted_curve,
                blp=_fit_blp_diagnostics(outcomes, calibrated_scores, weights),
            )
        )

    return CalibrationDiagnostics(
        method=model.method,
        diagnostic_mode=diagnostic_mode,
        n_outputs=Y_2d.shape[1],
        n_labeled=Y_2d.shape[0],
        num_bins=min(num_bins, Y_2d.shape[0]),
        per_output=tuple(per_output),
        effective_num_folds=effective_num_folds,
        reference_covariates=reference_covariates,
    )


def plot_calibration(
    diagnostics: CalibrationDiagnostics,
    *,
    output_index: int = 0,
    ax: Any = None,
    show_identity: bool = True,
    show_bins: bool = True,
) -> Any:
    """Plots a calibration diagnostic from :func:`calibration_diagnostics` output.

    The fitted curve is the score-to-outcome map implied by the fitted
    calibrator. The filled raw-score points place each bin's mean outcome at
    that bin's mean raw score. The hollow calibrated-score points place the
    same bin mean outcome at the corresponding mean calibrated score, so their
    horizontal shift shows how recalibration moves the score scale.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "plot_calibration requires matplotlib. Install matplotlib or use the optional 'plot' extra."
        ) from exc

    if output_index < 0 or output_index >= diagnostics.n_outputs:
        raise IndexError(
            f"output_index must lie in [0, {diagnostics.n_outputs - 1}], got {output_index}."
        )

    record = diagnostics.per_output[output_index]
    if ax is None:
        _, ax = plt.subplots()

    x_min = float(min(np.min(record.grid_scores), np.min(record.raw_labeled_scores)))
    x_max = float(max(np.max(record.grid_scores), np.max(record.raw_labeled_scores)))
    y_min = float(min(np.min(record.observed_outcomes), np.min(record.fitted_curve)))
    y_max = float(max(np.max(record.observed_outcomes), np.max(record.fitted_curve)))

    if show_identity:
        lo = min(x_min, y_min)
        hi = max(x_max, y_max)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.7", linewidth=1.5, label="Identity")

    ax.plot(record.grid_scores, record.fitted_curve, color="C0", linewidth=2.5, label="Fitted calibration")

    if show_bins:
        ax.scatter(
            record.bin_mean_raw_score,
            record.bin_mean_outcome,
            color="C1",
            s=35,
            label="Bin mean outcome at raw score",
        )
        ax.scatter(
            record.bin_mean_calibrated_score,
            record.bin_mean_outcome,
            facecolors="none",
            edgecolors="C2",
            s=35,
            linewidths=1.5,
            label="Same bin outcome at calibrated score",
        )

    ax.set_xlabel("Score value")
    ax.set_ylabel("Observed outcome")
    title = f"{diagnostics.method} calibration"
    if diagnostics.diagnostic_mode == "out_of_fold":
        title += " (out-of-fold)"
    if diagnostics.n_outputs > 1:
        title += f" (output {output_index})"
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    return ax
