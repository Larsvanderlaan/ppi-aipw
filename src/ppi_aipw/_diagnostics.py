from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._api import MeanInferenceResult, PrognosticLinearModel
from ._calibration import CalibrationModel
from ._utils import construct_weight_vector, reshape_to_2d, validate_pair_inputs


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


@dataclass(frozen=True)
class CalibrationDiagnostics:
    method: str
    n_outputs: int
    n_labeled: int
    num_bins: int
    per_output: tuple[CalibrationCurveDiagnostics, ...]
    reference_covariates: np.ndarray | None = None


def _coerce_covariates(
    X: np.ndarray | None,
    *,
    n_obs: int,
) -> np.ndarray | None:
    if X is None:
        return None
    X_2d = reshape_to_2d(np.asarray(X, dtype=float))
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
    num_bins: int = 10,
) -> CalibrationDiagnostics:
    """Builds labeled-sample calibration diagnostics for a fitted result or model."""

    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    model = _resolve_model(obj)
    Y_2d, Yhat_2d = validate_pair_inputs(Y, Yhat)
    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=False)
    X_2d = _coerce_covariates(X, n_obs=Y_2d.shape[0])

    if isinstance(model, PrognosticLinearModel):
        if model.x_dim > 0:
            if X_2d is None:
                raise ValueError(
                    "X is required for calibration diagnostics when the fitted model uses prognostic covariates."
                )
            reference_covariates = np.average(X_2d, axis=0, weights=weights)
            calibrated_labeled = np.asarray(model.predict(Yhat_2d, X=X_2d), dtype=float)
        else:
            reference_covariates = None
            calibrated_labeled = np.asarray(model.predict(Yhat_2d), dtype=float)
    else:
        reference_covariates = None
        calibrated_labeled = np.asarray(model.predict(Yhat_2d), dtype=float)

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
            )
        )

    return CalibrationDiagnostics(
        method=model.method,
        n_outputs=Y_2d.shape[1],
        n_labeled=Y_2d.shape[0],
        num_bins=min(num_bins, Y_2d.shape[0]),
        per_output=tuple(per_output),
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
    """Plots a calibration curve from :func:`calibration_diagnostics` output."""

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
            label="Binned outcome vs raw score",
        )
        ax.scatter(
            record.bin_mean_calibrated_score,
            record.bin_mean_outcome,
            facecolors="none",
            edgecolors="C2",
            s=35,
            linewidths=1.5,
            label="Binned outcome vs calibrated score",
        )

    ax.set_xlabel("Prediction score")
    ax.set_ylabel("Observed outcome")
    title = f"{diagnostics.method} calibration"
    if diagnostics.n_outputs > 1:
        title += f" (output {output_index})"
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    return ax
