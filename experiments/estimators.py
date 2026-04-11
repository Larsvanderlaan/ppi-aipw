from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families import Gaussian
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


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
