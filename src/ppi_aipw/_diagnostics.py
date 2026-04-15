from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ._api import MeanInferenceResult, PrognosticLinearModel, _fit_and_calibrate, _fit_prognostic_linear
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
    diagnostic_mode: str
    n_outputs: int
    n_labeled: int
    num_bins: int
    per_output: tuple[CalibrationCurveDiagnostics, ...]
    effective_num_folds: int | None = None
    reference_covariates: np.ndarray | None = None


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
