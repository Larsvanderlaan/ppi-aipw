"""Calibration helpers for the MATH example in the paper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from sklearn.linear_model import LinearRegression, LogisticRegression

from experiments.llm_eval_models import load_jsonl_frame, row_cache_path, score_cache_path

INTRO_PREFERRED_EVALUATOR = "armorm_llama3_8b_v0_1"
DISPLAY_NAMES = {"armorm_llama3_8b_v0_1": "ArmoRM (PPE correctness)"}


@dataclass(frozen=True)
class CandidateData:
    candidate_id: str
    display_name: str
    family: str
    score_label: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    fitted: Any
    y_min: float
    y_max: float


def _z_value(alpha: float) -> float:
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _scalarize(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected a scalar-like value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def official_ppi_summary(
    y_labeled: np.ndarray,
    score_labeled: np.ndarray,
    score_unlabeled: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    estimate = _scalarize(ppi_mean_pointestimate(y_labeled, score_labeled, score_unlabeled, lam=1))
    lower, upper = ppi_mean_ci(y_labeled, score_labeled, score_unlabeled, alpha=alpha, lam=1)
    se = (_scalarize(upper) - _scalarize(lower)) / (2.0 * _z_value(alpha))
    return estimate, float(se)


def fit_one_dimensional_calibrator(
    score_labeled: np.ndarray,
    y_labeled: np.ndarray,
    *,
    method: str,
) -> CalibrationModel:
    score_labeled = np.asarray(score_labeled, dtype=float).reshape(-1)
    y_labeled = np.asarray(y_labeled, dtype=float).reshape(-1)
    y_min = float(np.min(y_labeled))
    y_max = float(np.max(y_labeled))
    mean_y = float(np.mean(y_labeled))

    if np.allclose(score_labeled, score_labeled[0]) or np.isclose(y_min, y_max):
        return CalibrationModel(
            method=method,
            fitted={"constant": mean_y},
            y_min=y_min,
            y_max=y_max,
        )

    if method == "platt":
        if not np.all(np.isin(np.unique(y_labeled), [0.0, 1.0])):
            raise ValueError("Platt calibration requires binary outcomes in {0, 1}.")
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )
        model.fit(score_labeled.reshape(-1, 1), y_labeled.astype(int))
        return CalibrationModel(method=method, fitted=model, y_min=y_min, y_max=y_max)

    if method == "linear":
        model = LinearRegression()
        model.fit(score_labeled.reshape(-1, 1), y_labeled)
        return CalibrationModel(method=method, fitted=model, y_min=y_min, y_max=y_max)

    raise ValueError(f"Unknown calibrator '{method}'.")


def predict_with_calibrator(model: CalibrationModel, score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float).reshape(-1)
    if isinstance(model.fitted, dict):
        pred = np.full(score.shape, float(model.fitted["constant"]), dtype=float)
    elif model.method == "platt":
        pred = model.fitted.predict_proba(score.reshape(-1, 1))[:, 1]
    elif model.method == "linear":
        pred = model.fitted.predict(score.reshape(-1, 1))
    else:  # pragma: no cover - guarded by fit_one_dimensional_calibrator
        raise ValueError(f"Unknown calibrator '{model.method}'.")
    return np.clip(np.asarray(pred, dtype=float), model.y_min, model.y_max)


def calibrator_for_candidate(candidate: CandidateData) -> str:
    unique_y = np.unique(np.asarray(candidate.frame["y"], dtype=float))
    if np.all(np.isin(unique_y, [0.0, 1.0])):
        return "platt"
    return "linear"


def load_ppe_candidate(cache_dir: Path, track: str, evaluator: str) -> CandidateData:
    row_frame = load_jsonl_frame(row_cache_path(cache_dir, track))
    score_frame = load_jsonl_frame(score_cache_path(cache_dir, track, evaluator))
    merged = row_frame.merge(score_frame[["unit_id", "margin"]], on="unit_id", how="inner")
    frame = merged[["unit_id", "benchmark", "target_model", "y", "margin"]].copy()
    frame["score"] = frame["margin"].astype(float)
    frame["y"] = frame["y"].astype(float)
    return CandidateData(
        candidate_id=evaluator,
        display_name=DISPLAY_NAMES[evaluator],
        family="ppe",
        score_label="Raw margin",
        frame=frame,
    )
