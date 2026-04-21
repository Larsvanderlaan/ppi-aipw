from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from ppi_py.baselines import classical_mean_ci
from ppi_py.datasets import load_dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.estimators import (
    aipw_em_result,
    aipp_from_prediction,
    auto_aipw_pointestimate_and_se,
    fit_binned_isotonic_calibration,
    fit_tuned_binned_isotonic_calibration,
    fit_linear_calibration,
    fit_monotone_spline_calibration,
    fit_platt_calibration,
    fit_venn_abers_shrinkage_calibration,
    influence_se_from_prediction,
    labeled_mean,
    predict_binned_isotonic,
    predict_linear,
    predict_monotone_spline,
    predict_platt,
    predict_venn_abers_shrinkage,
)


DATASET_GOOGLE_DRIVE_IDS: Dict[str, str] = {
    "alphafold": "1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho",
    "ballots": "1DJvTWvPM6zQD0V4yGH1O7DL3kfnTE06u",
    "census_education": "15iq7nLjwogb46v3stknMmx7kMuK9cnje",
    "census_income": "15dZeWw-RTw17-MieG4y1ILTZlreJOmBS",
    "census_healthcare": "1RjWsnq-gMngRFRj22DvezcdCVl2MxAIX",
    "forest": "1Vqi1wSmVnWh_2lLQuDwrhkGcipvoWBc0",
    "galaxies": "1pDLQesPhbH5fSZW1m4aWC-wnJWnp1rGV",
    "gene_expression": "17PwlvAAKeBYGLXPz9L2LVnNJ66XjuyZd",
    "plankton": "1KEk0ZFZ6KiB7_2tdPc5fyBDFNhhJUS_W",
}

ESTIMATOR_ORDER = [
    "classical",
    "imputation",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "linear_calibration",
    "platt_calibration",
    "isotonic_calibration_min10",
    "venn_abers_calibration",
]

ESTIMATOR_LABELS = {
    "classical": "Labeled-only",
    "imputation": "Imputation",
    "semisupervised": "Semisup. linear",
    "aipw": "AIPW",
    "ppi": "PPI",
    "ppi_plus_plus": "PPI++",
    "aipw_em": "AIPW-EM",
    "auto_calibration": "AutoCal",
    "monotone_spline": "MonoSpline",
    "linear_calibration": "LinearCal",
    "platt_calibration": "Platt",
    "isotonic_calibration_min10": "IsoCal",
    "venn_abers_calibration": "Venn-Abers",
}

ESTIMATOR_COLORS = {
    "classical": "#4D4D4D",
    "imputation": "#999999",
    "semisupervised": "#E69F00",
    "aipw": "#8C564B",
    "ppi": "#D55E00",
    "ppi_plus_plus": "#C44E52",
    "aipw_em": "#4E79A7",
    "auto_calibration": "#E69F00",
    "monotone_spline": "#0B6E4F",
    "linear_calibration": "#009E73",
    "platt_calibration": "#56B4E9",
    "isotonic_calibration_min10": "#CC79A7",
    "venn_abers_calibration": "#7A3E9D",
}

PAPER_ESTIMATOR_ORDER = [
    "classical",
    "imputation",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "linear_calibration",
    "platt_calibration",
    "isotonic_calibration_min10",
    "venn_abers_calibration",
]

MAIN_TEXT_ESTIMATOR_ORDER = [
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "linear_calibration",
]

APPENDIX_CALIBRATION_ORDER = [
    "ppi",
    "aipw",
    "linear_calibration",
    "monotone_spline",
    "isotonic_calibration_min10",
]

ALL_ESTIMATORS: Tuple[str, ...] = (
    "classical",
    "imputation",
    "semisupervised",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "linear_calibration",
    "platt_calibration",
    "isotonic_calibration_min10",
    "venn_abers_calibration",
)

PRIMARY_ESTIMATORS = {
    "linear_calibration",
    "aipw",
    "auto_calibration",
    "monotone_spline",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "isotonic_calibration_min10",
}


def plot_draw_order(order: Sequence[str]) -> List[str]:
    secondary = [name for name in order if name not in PRIMARY_ESTIMATORS]
    primary = [
        name
        for name in order
        if name in PRIMARY_ESTIMATORS and name not in {"aipw_em", "ppi_plus_plus", "ppi", "aipw"}
    ]
    tail = [name for name in ["aipw_em", "ppi_plus_plus", "ppi", "aipw"] if name in order]
    return secondary + primary + tail


def plot_line_width(estimator: str, default: float = 1.8) -> float:
    if estimator in PRIMARY_ESTIMATORS:
        return default + 0.35
    return max(1.0, default - 0.45)


def plot_line_alpha(estimator: str) -> float:
    return 1.0 if estimator in PRIMARY_ESTIMATORS else 0.5


def plot_line_zorder(estimator: str) -> int:
    if estimator == "aipw_em":
        return 14
    if estimator == "aipw":
        return 12
    if estimator == "ppi":
        return 11
    if estimator == "ppi_plus_plus":
        return 13
    if estimator == "auto_calibration":
        return 10
    if estimator == "monotone_spline":
        return 9
    if estimator in PRIMARY_ESTIMATORS:
        return 8
    return 3


def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def scalarize(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected a scalar-like value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class MeanExperimentConfig:
    name: str
    dataset_name: str
    source_notebook: str
    n_grid: Tuple[int, ...]
    alpha: float
    replications: int
    baselines: Tuple[str, ...]
    imputation_strategy: Optional[str] = None
    default_seed: int = 0
    has_features: bool = False


# This manifest intentionally excludes `ballots` because the official dataset
# bundle does not include unlabeled gold-standard outcomes, so it cannot support
# the full-label split/coverage workflow requested for this reproduction.
MEAN_EXPERIMENTS: Dict[str, MeanExperimentConfig] = {
    "galaxies_mean": MeanExperimentConfig(
        name="galaxies_mean",
        dataset_name="galaxies",
        source_notebook="examples/galaxies.ipynb",
        n_grid=tuple(np.linspace(50, 1000, 10).astype(int).tolist()),
        alpha=0.1,
        replications=100,
        baselines=("classical", "imputation"),
        imputation_strategy="threshold_0.5",
    ),
    "forest_mean": MeanExperimentConfig(
        name="forest_mean",
        dataset_name="forest",
        source_notebook="examples/forest.ipynb",
        n_grid=tuple(np.linspace(50, 500, 10).astype(int).tolist()),
        alpha=0.05,
        replications=100,
        baselines=("classical", "imputation"),
        imputation_strategy="threshold_0.5",
    ),
    "census_income_semisupervised_mean": MeanExperimentConfig(
        name="census_income_semisupervised_mean",
        dataset_name="census_income",
        source_notebook="examples/baselines/semisupervised.ipynb",
        n_grid=tuple(np.linspace(50, 1000, 10).astype(int).tolist()),
        alpha=0.1,
        replications=50,
        baselines=("classical", "semisupervised"),
        has_features=True,
    ),
}


def z_value(alpha: float) -> float:
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def ci_from_estimate_and_se(estimate: float, se: float, alpha: float) -> Tuple[float, float]:
    z = z_value(alpha)
    return float(estimate - z * se), float(estimate + z * se)


def se_from_ci(lower: float, upper: float, alpha: float) -> float:
    width = float(upper - lower)
    return width / (2.0 * z_value(alpha))


def clip_to_observed_range(values: np.ndarray, y_l: np.ndarray) -> np.ndarray:
    y_min = float(np.min(y_l))
    y_max = float(np.max(y_l))
    return np.clip(np.asarray(values, dtype=float), y_min, y_max)


def ensure_dataset_cached(cache_dir: Path, dataset_name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_dir / f"{dataset_name}.npz"
    if dataset_path.exists():
        return dataset_path

    dataset_id = DATASET_GOOGLE_DRIVE_IDS.get(dataset_name)
    if dataset_id is None:
        raise KeyError(f"Unknown PPI dataset: {dataset_name}")

    gdown_exec = shutil.which("gdown")
    if gdown_exec is None:
        # `sys.executable` may be a virtualenv shim; using `.parent` preserves
        # the active environment's bin directory instead of resolving to the
        # system Python install.
        candidate = Path(sys.executable).parent / "gdown"
        if candidate.exists():
            gdown_exec = str(candidate)
    if gdown_exec is None:
        raise RuntimeError(
            "Could not find `gdown`. Install it in the active environment before downloading PPI datasets."
        )

    subprocess.run(
        [gdown_exec, dataset_id, "-O", str(dataset_path)],
        check=True,
    )
    return dataset_path


def load_dataset_bundle(cache_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    ensure_dataset_cached(cache_dir, dataset_name)
    bundle = load_dataset(str(cache_dir), dataset_name, download=False)
    return {key: np.asarray(bundle[key]) for key in bundle.files}


def selected_experiments(dataset_names: Optional[Sequence[str]]) -> List[MeanExperimentConfig]:
    if not dataset_names:
        return list(MEAN_EXPERIMENTS.values())
    missing = [name for name in dataset_names if name not in MEAN_EXPERIMENTS]
    if missing:
        valid = ", ".join(sorted(MEAN_EXPERIMENTS))
        raise ValueError(f"Unknown dataset config(s): {missing}. Valid options: {valid}")
    return [MEAN_EXPERIMENTS[name] for name in dataset_names]


def prepare_mean_dataset(config: MeanExperimentConfig, cache_dir: Path) -> Dict[str, np.ndarray]:
    bundle = load_dataset_bundle(cache_dir, config.dataset_name)
    data = {
        "Y_total": np.asarray(bundle["Y"], dtype=float),
        "Yhat_total": np.asarray(bundle["Yhat"], dtype=float),
    }
    if config.has_features:
        data["X_total"] = np.asarray(bundle["X"], dtype=float)
    return data


def summarize_result(
    *,
    config: MeanExperimentConfig,
    estimator: str,
    n_labeled: int,
    n_unlabeled: int,
    replicate: int,
    alpha: float,
    true_theta: float,
    estimate: float,
    se: float,
    lower: float,
    upper: float,
) -> Dict[str, float]:
    return {
        "dataset": config.name,
        "dataset_name": config.dataset_name,
        "source_notebook": config.source_notebook,
        "estimand": "mean",
        "alpha": alpha,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "replicate": replicate,
        "estimator": estimator,
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_width": float(upper - lower),
        "bias": float(estimate - true_theta),
        "sq_error": float((estimate - true_theta) ** 2),
        "covered": float(lower <= true_theta <= upper),
        "true_theta": float(true_theta),
    }


def run_classical_estimator(y_l: np.ndarray, alpha: float) -> Dict[str, float]:
    estimate = float(labeled_mean(y_l))
    lower, upper = classical_mean_ci(y_l, alpha=alpha)
    lower_scalar = scalarize(lower)
    upper_scalar = scalarize(upper)
    se = se_from_ci(lower_scalar, upper_scalar, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower_scalar,
        "ci_upper": upper_scalar,
    }


def run_imputation_estimator(
    yhat_total: np.ndarray,
    alpha: float,
    strategy: str,
) -> Dict[str, float]:
    if strategy == "threshold_0.5":
        pseudo_outcomes = (np.asarray(yhat_total, dtype=float) > 0.5).astype(float)
    elif strategy == "direct":
        pseudo_outcomes = np.asarray(yhat_total, dtype=float)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    estimate = float(np.mean(pseudo_outcomes))
    lower, upper = classical_mean_ci(pseudo_outcomes, alpha=alpha)
    lower_scalar = scalarize(lower)
    upper_scalar = scalarize(upper)
    se = se_from_ci(lower_scalar, upper_scalar, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower_scalar,
        "ci_upper": upper_scalar,
    }


def run_ppi_estimator(y_l: np.ndarray, yhat_l: np.ndarray, yhat_u: np.ndarray, alpha: float) -> Dict[str, float]:
    estimate = scalarize(ppi_mean_pointestimate(y_l, yhat_l, yhat_u, lam=1))
    lower, upper = ppi_mean_ci(y_l, yhat_l, yhat_u, alpha=alpha, lam=1)
    lower_scalar = scalarize(lower)
    upper_scalar = scalarize(upper)
    se = se_from_ci(lower_scalar, upper_scalar, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower_scalar,
        "ci_upper": upper_scalar,
    }


def run_ppi_plus_plus_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    estimate = scalarize(ppi_mean_pointestimate(y_l, yhat_l, yhat_u, lam=None))
    lower, upper = ppi_mean_ci(y_l, yhat_l, yhat_u, alpha=alpha, lam=None)
    lower_scalar = scalarize(lower)
    upper_scalar = scalarize(upper)
    se = se_from_ci(lower_scalar, upper_scalar, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower_scalar,
        "ci_upper": upper_scalar,
    }


def run_aipw_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l = np.asarray(yhat_l, dtype=float)
    pred_u = np.asarray(yhat_u, dtype=float)
    estimate = float(aipp_from_prediction(pred_l, pred_u, np.asarray(y_l, dtype=float)))
    se = float(influence_se_from_prediction(estimate, pred_l, pred_u, np.asarray(y_l, dtype=float)))
    lower, upper = ci_from_estimate_and_se(estimate, se, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def run_aipw_em_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    return aipw_em_result(y_l, yhat_l, yhat_u, alpha=alpha)


def run_auto_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    result = auto_aipw_pointestimate_and_se(
        y_l,
        yhat_l,
        yhat_u,
        candidate_methods=("aipw", "linear", "monotone_spline", "isocal"),
        num_folds=20,
        random_state=0,
    )
    lower, upper = ci_from_estimate_and_se(result["estimate"], result["se"], alpha)
    return {
        "estimate": result["estimate"],
        "se": result["se"],
        "ci_lower": lower,
        "ci_upper": upper,
    }


def monotone_spline_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_monotone_spline_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
    )
    pred_l = clip_to_observed_range(predict_monotone_spline(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_monotone_spline(model, yhat_u), y_l)
    return pred_l, pred_u


def linear_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_linear_calibration(np.asarray(yhat_l, dtype=float), np.asarray(y_l, dtype=float))
    pred_l = clip_to_observed_range(predict_linear(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_linear(model, yhat_u), y_l)
    return pred_l, pred_u


def isotonic_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_tuned_binned_isotonic_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
    )
    pred_l = clip_to_observed_range(predict_binned_isotonic(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_binned_isotonic(model, yhat_u), y_l)
    return pred_l, pred_u


def isotonic_min10_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_binned_isotonic_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
        min_bin_size=10,
    )
    pred_l = clip_to_observed_range(predict_binned_isotonic(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_binned_isotonic(model, yhat_u), y_l)
    return pred_l, pred_u


def platt_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model = fit_platt_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
    )
    pred_l = clip_to_observed_range(predict_platt(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_platt(model, yhat_u), y_l)
    return pred_l, pred_u


def venn_abers_calibrated_predictions(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    reference_model = fit_tuned_binned_isotonic_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
    )
    model = fit_venn_abers_shrinkage_calibration(
        np.asarray(yhat_l, dtype=float),
        np.asarray(y_l, dtype=float),
        round_digits=2,
        reference_model=reference_model,
    )
    pred_l = clip_to_observed_range(predict_venn_abers_shrinkage(model, yhat_l), y_l)
    pred_u = clip_to_observed_range(predict_venn_abers_shrinkage(model, yhat_u), y_l)
    return pred_l, pred_u


def run_prediction_plugin_estimator(
    y_l: np.ndarray,
    pred_l: np.ndarray,
    pred_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    estimate = float(aipp_from_prediction(pred_l, pred_u, y_l))
    se = float(influence_se_from_prediction(estimate, pred_l, pred_u, y_l))
    lower, upper = ci_from_estimate_and_se(estimate, se, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def run_linear_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = linear_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_monotone_spline_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = monotone_spline_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_isotonic_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = isotonic_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_isotonic_min10_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = isotonic_min10_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_platt_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = platt_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_venn_abers_calibration_estimator(
    y_l: np.ndarray,
    yhat_l: np.ndarray,
    yhat_u: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    pred_l, pred_u = venn_abers_calibrated_predictions(y_l, yhat_l, yhat_u)
    return run_prediction_plugin_estimator(y_l, pred_l, pred_u, alpha)


def run_semisupervised_estimator(
    x_l: np.ndarray,
    y_l: np.ndarray,
    x_u: np.ndarray,
    alpha: float,
    k_folds: int = 5,
) -> Dict[str, float]:
    x_l = np.asarray(x_l, dtype=float)
    y_l = np.asarray(y_l, dtype=float)
    x_u = np.asarray(x_u, dtype=float)
    if x_l.ndim != 2 or x_u.ndim != 2:
        raise ValueError("Semisupervised baseline requires 2D feature arrays.")

    x_l = np.concatenate([np.ones((x_l.shape[0], 1)), x_l], axis=1)
    x_u = np.concatenate([np.ones((x_u.shape[0], 1)), x_u], axis=1)

    n = y_l.shape[0]
    n_folds = max(2, min(int(k_folds), n))
    fold_ids = np.arange(n) % n_folds
    yhat_l = np.zeros(n, dtype=float)
    yhat_u = np.zeros(x_u.shape[0], dtype=float)

    for fold in range(n_folds):
        train_idx = fold_ids != fold
        valid_idx = fold_ids == fold
        beta, _, _, _ = np.linalg.lstsq(x_l[train_idx], y_l[train_idx], rcond=None)
        yhat_l[valid_idx] = x_l[valid_idx] @ beta
        yhat_u += (x_u @ beta) / n_folds

    estimate = float(np.mean(yhat_u) + np.mean(y_l - yhat_l))
    residual = y_l - yhat_l
    variance = float(np.var(residual, ddof=0) / n + np.var(yhat_u, ddof=0) / x_u.shape[0])
    se = float(np.sqrt(max(variance, 0.0)))
    lower, upper = ci_from_estimate_and_se(estimate, se, alpha)
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def build_trial_results(
    *,
    config: MeanExperimentConfig,
    alpha: float,
    replicate: int,
    n_labeled: int,
    true_theta: float,
    y_total: np.ndarray,
    yhat_total: np.ndarray,
    permutation: np.ndarray,
    x_total: Optional[np.ndarray],
    selected_estimators: Optional[Sequence[str]] = None,
) -> List[Dict[str, float]]:
    labeled_idx = permutation[:n_labeled]
    unlabeled_idx = permutation[n_labeled:]
    y_l = y_total[labeled_idx]
    yhat_l = yhat_total[labeled_idx]
    yhat_u = yhat_total[unlabeled_idx]
    n_unlabeled = int(unlabeled_idx.shape[0])
    selected = set(selected_estimators) if selected_estimators is not None else None

    rows: List[Dict[str, float]] = []
    is_binary_outcome = bool(np.all(np.isin(np.unique(y_l), [0.0, 1.0])))

    for estimator in config.baselines:
        if selected is not None and estimator not in selected:
            continue
        if estimator == "classical":
            result = run_classical_estimator(y_l, alpha)
        elif estimator == "imputation":
            if config.imputation_strategy is None:
                raise ValueError(f"Dataset {config.name} does not define an imputation baseline.")
            result = run_imputation_estimator(yhat_total, alpha, config.imputation_strategy)
        elif estimator == "semisupervised":
            if x_total is None:
                raise ValueError(f"Dataset {config.name} does not include feature covariates.")
            result = run_semisupervised_estimator(x_total[labeled_idx], y_l, x_total[unlabeled_idx], alpha)
        else:
            raise ValueError(f"Unknown baseline estimator: {estimator}")

        rows.append(
            summarize_result(
                config=config,
                estimator=estimator,
                n_labeled=n_labeled,
                n_unlabeled=n_unlabeled,
                replicate=replicate,
                alpha=alpha,
                true_theta=true_theta,
                estimate=result["estimate"],
                se=result["se"],
                lower=result["ci_lower"],
                upper=result["ci_upper"],
            )
        )

    for estimator, runner in (
        ("aipw", run_aipw_estimator),
        ("ppi", run_ppi_estimator),
        ("ppi_plus_plus", run_ppi_plus_plus_estimator),
        ("aipw_em", run_aipw_em_estimator),
        ("auto_calibration", run_auto_calibration_estimator),
        ("monotone_spline", run_monotone_spline_estimator),
        ("linear_calibration", run_linear_calibration_estimator),
        ("platt_calibration", run_platt_calibration_estimator),
        ("isotonic_calibration_min10", run_isotonic_min10_calibration_estimator),
        ("venn_abers_calibration", run_venn_abers_calibration_estimator),
    ):
        if selected is not None and estimator not in selected:
            continue
        if estimator in {"platt_calibration", "venn_abers_calibration"} and not is_binary_outcome:
            continue
        result = runner(y_l, yhat_l, yhat_u, alpha)
        rows.append(
            summarize_result(
                config=config,
                estimator=estimator,
                n_labeled=n_labeled,
                n_unlabeled=n_unlabeled,
                replicate=replicate,
                alpha=alpha,
                true_theta=true_theta,
                estimate=result["estimate"],
                se=result["se"],
                lower=result["ci_lower"],
                upper=result["ci_upper"],
            )
        )

    return rows


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "dataset_name",
        "source_notebook",
        "estimand",
        "alpha",
        "n_labeled",
        "n_unlabeled",
        "estimator",
        "true_theta",
    ]
    summary = (
        raw_df.groupby(group_cols, as_index=False)
        .agg(
            mean_estimate=("estimate", "mean"),
            mean_bias=("bias", "mean"),
            emp_variance=("estimate", lambda values: float(np.var(values, ddof=0))),
            mse=("sq_error", "mean"),
            rmse=("sq_error", lambda values: float(np.sqrt(np.mean(values)))),
            mean_se=("se", "mean"),
            coverage=("covered", "mean"),
            mean_ci_width=("ci_width", "mean"),
        )
        .sort_values(["dataset", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )
    ppi_variance = (
        summary[summary["estimator"] == "ppi"][
            ["dataset", "n_labeled", "n_unlabeled", "emp_variance"]
        ]
        .rename(columns={"emp_variance": "emp_variance_ppi"})
        .drop_duplicates()
    )
    ppi_mse = (
        summary[summary["estimator"] == "ppi"][
            ["dataset", "n_labeled", "n_unlabeled", "mse"]
        ]
        .rename(columns={"mse": "mse_ppi"})
        .drop_duplicates()
    )
    summary = summary.merge(
        ppi_variance,
        on=["dataset", "n_labeled", "n_unlabeled"],
        how="left",
    )
    summary = summary.merge(
        ppi_mse,
        on=["dataset", "n_labeled", "n_unlabeled"],
        how="left",
    )
    summary["rel_eff_vs_ppi"] = summary["emp_variance_ppi"] / summary["emp_variance"]
    summary["mse_ratio_vs_ppi"] = summary["mse"] / summary["mse_ppi"]
    return summary


def write_dataset_tables(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = filter_paper_estimators(summary_df)
    for dataset_name, frame in summary_df.groupby("dataset", sort=False):
        table = frame[
            [
                "dataset",
                "source_notebook",
                "n_labeled",
                "n_unlabeled",
                "estimator",
                "mean_estimate",
                "mean_bias",
                "emp_variance",
                "mse",
                "rmse",
                "coverage",
                "rel_eff_vs_ppi",
            ]
        ].copy()
        table.to_csv(output_dir / f"table_{dataset_name}.csv", index=False)


def filter_paper_estimators(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[summary_df["estimator"].isin(PAPER_ESTIMATOR_ORDER)].copy()


def write_paper_summary_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = filter_paper_estimators(summary_df)
    selected_rows = []
    for dataset_name, frame in summary_df.groupby("dataset", sort=False):
        n_values = sorted(frame["n_labeled"].unique().tolist())
        for target_n in [n_values[0], n_values[len(n_values) // 2], n_values[-1]]:
            sub = frame[frame["n_labeled"] == target_n].copy()
            selected_rows.append(sub)
    table = pd.concat(selected_rows, ignore_index=True)[
        [
            "dataset",
            "n_labeled",
            "n_unlabeled",
            "estimator",
            "mean_bias",
            "emp_variance",
            "mse",
            "coverage",
            "rel_eff_vs_ppi",
        ]
    ].sort_values(["dataset", "n_labeled", "estimator"]).reset_index(drop=True)
    table.to_csv(output_dir / "table_paper_summary.csv", index=False)
    latex_lines = [
        r"\begin{tabular}{lllrccccc}",
        r"\toprule",
        r"Dataset & $n$ & $N$ & Estimator & Bias & Variance & MSE & Coverage & RelEff \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{int(row['n_labeled'])} & "
            f"{int(row['n_unlabeled'])} & "
            f"{latex_escape(ESTIMATOR_LABELS.get(row['estimator'], row['estimator']))} & "
            f"{row['mean_bias']:.4f} & "
            f"{row['emp_variance']:.4f} & "
            f"{row['mse']:.4f} & "
            f"{row['coverage']:.3f} & "
            f"{row['rel_eff_vs_ppi']:.3f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table_paper_summary.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def plot_dataset_metric(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    ylabel: str,
    suffix: str,
) -> None:
    for dataset_name, frame in summary_df.groupby("dataset", sort=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for estimator in plot_draw_order([name for name in ESTIMATOR_ORDER if name in frame["estimator"].unique()]):
            sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
            ax.plot(
                sub["n_labeled"],
                sub[metric],
                marker="o",
                linewidth=plot_line_width(estimator, default=2.0),
                color=ESTIMATOR_COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
                alpha=plot_line_alpha(estimator),
                zorder=plot_line_zorder(estimator),
            )
        if metric == "coverage":
            alpha = float(frame["alpha"].iloc[0])
            ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1)
        if metric in {"mse_ratio_vs_ppi", "rel_eff_vs_ppi"}:
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(frame["n_labeled"].min(), frame["n_labeled"].max())
        ax.set_title(dataset_name)
        ax.set_xlabel("Labeled sample size n")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"fig_{dataset_name}_{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_overall_metric_grid(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = filter_paper_estimators(summary_df)
    metrics = [
        ("mean_bias", "Bias", "bias"),
        ("emp_variance", "Variance", "variance"),
        ("mse_ratio_vs_ppi", "MSE / PPI MSE", "mse_vs_ppi"),
        ("coverage", "Coverage", "coverage"),
        ("rel_eff_vs_ppi", "Rel. eff. vs PPI", "relative_efficiency"),
    ]
    datasets = list(summary_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.5 * len(metrics), 3.8 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])
    for row_idx, dataset_name in enumerate(datasets):
        frame = summary_df[summary_df["dataset"] == dataset_name].copy()
        for col_idx, (metric, ylabel, _) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for estimator in plot_draw_order([name for name in PAPER_ESTIMATOR_ORDER if name in frame["estimator"].unique()]):
                sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
                ax.plot(
                    sub["n_labeled"],
                    sub[metric],
                    marker="o",
                    linewidth=plot_line_width(estimator),
                    color=ESTIMATOR_COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    alpha=plot_line_alpha(estimator),
                    zorder=plot_line_zorder(estimator),
                )
            if metric == "coverage":
                alpha = float(frame["alpha"].iloc[0])
                ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1)
            if metric in {"mse_ratio_vs_ppi", "rel_eff_vs_ppi"}:
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(frame["n_labeled"].min(), frame["n_labeled"].max())
            if row_idx == 0:
                ax.set_title(ylabel)
            if col_idx == 0:
                ax.set_ylabel(dataset_name)
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Labeled sample size n")
    legend_estimators = [name for name in PAPER_ESTIMATOR_ORDER if name in summary_df["estimator"].unique()]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ESTIMATOR_COLORS[name],
            marker="o",
            linewidth=plot_line_width(name),
            alpha=plot_line_alpha(name),
            label=ESTIMATOR_LABELS[name],
        )
        for name in legend_estimators
    ]
    fig.legend(
        legend_handles,
        [ESTIMATOR_LABELS[name] for name in legend_estimators],
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "fig_paper_metric_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_text_grid(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = summary_df[summary_df["estimator"].isin(MAIN_TEXT_ESTIMATOR_ORDER)].copy()
    metrics = [
        ("mse_ratio_vs_ppi", "MSE / PPI MSE"),
        ("rel_eff_vs_ppi", "Rel. eff. vs PPI"),
        ("coverage", "Coverage"),
    ]
    datasets = list(summary_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(4.6 * len(datasets), 6.0),
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        if len(metrics) == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(len(metrics), 1)
    for col_idx, dataset_name in enumerate(datasets):
        frame = summary_df[summary_df["dataset"] == dataset_name].copy()
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for estimator in plot_draw_order([name for name in MAIN_TEXT_ESTIMATOR_ORDER if name in frame["estimator"].unique()]):
                sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
                ax.plot(
                    sub["n_labeled"],
                    sub[metric],
                    marker="o",
                    linewidth=plot_line_width(estimator),
                    color=ESTIMATOR_COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    alpha=plot_line_alpha(estimator),
                    zorder=plot_line_zorder(estimator),
                )
            if metric == "coverage":
                alpha = float(frame["alpha"].iloc[0])
                ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1)
            if metric in {"mse_ratio_vs_ppi", "rel_eff_vs_ppi"}:
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(frame["n_labeled"].min(), frame["n_labeled"].max())
            if row_idx == 0:
                ax.set_title(dataset_name)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Labeled sample size n")
    legend_estimators = [name for name in MAIN_TEXT_ESTIMATOR_ORDER if name in summary_df["estimator"].unique()]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ESTIMATOR_COLORS[name],
            marker="o",
            linewidth=plot_line_width(name),
            alpha=plot_line_alpha(name),
            label=ESTIMATOR_LABELS[name],
        )
        for name in legend_estimators
    ]
    fig.legend(
        legend_handles,
        [ESTIMATOR_LABELS[name] for name in legend_estimators],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig_paper_main_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_appendix_calibration_grid(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = summary_df[summary_df["estimator"].isin(APPENDIX_CALIBRATION_ORDER)].copy()
    metrics = [
        ("mse_ratio_vs_ppi", "MSE / PPI MSE"),
        ("rel_eff_vs_ppi", "Rel. eff. vs PPI"),
        ("coverage", "Coverage"),
    ]
    datasets = list(summary_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(4.6 * len(datasets), 6.0),
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        if len(metrics) == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(len(metrics), 1)
    for col_idx, dataset_name in enumerate(datasets):
        frame = summary_df[summary_df["dataset"] == dataset_name].copy()
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for estimator in plot_draw_order([name for name in APPENDIX_CALIBRATION_ORDER if name in frame["estimator"].unique()]):
                sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
                ax.plot(
                    sub["n_labeled"],
                    sub[metric],
                    marker="o",
                    linewidth=plot_line_width(estimator),
                    color=ESTIMATOR_COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    alpha=plot_line_alpha(estimator),
                    zorder=plot_line_zorder(estimator),
                )
            if metric == "coverage":
                alpha = float(frame["alpha"].iloc[0])
                ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1)
            if metric in {"mse_ratio_vs_ppi", "rel_eff_vs_ppi"}:
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(frame["n_labeled"].min(), frame["n_labeled"].max())
            if row_idx == 0:
                ax.set_title(dataset_name)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Labeled sample size n")
    legend_estimators = [name for name in APPENDIX_CALIBRATION_ORDER if name in summary_df["estimator"].unique()]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ESTIMATOR_COLORS[name],
            marker="o",
            linewidth=plot_line_width(name),
            alpha=plot_line_alpha(name),
            label=ESTIMATOR_LABELS[name],
        )
        for name in legend_estimators
    ]
    fig.legend(
        legend_handles,
        [ESTIMATOR_LABELS[name] for name in legend_estimators],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig_paper_calibration_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="Experiment ids to run.")
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Override the notebook-default replication count.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override the notebook-default error level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the notebook-default random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppi_mean_reproduction/default_run"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/ppi_mean_reproduction/cache"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny subset of the experiment grid for quick validation.",
    )
    parser.add_argument(
        "--estimators",
        nargs="*",
        default=None,
        help="Optional estimator subset to run.",
    )
    parser.add_argument(
        "--main-text-only",
        action="store_true",
        help="Run only the estimators and asset generation needed for the main-text benchmark figure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.main_text_only and args.estimators is not None:
        parser.error("Use either --main-text-only or --estimators, not both.")

    experiments = selected_experiments(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.main_text_only:
        selected_estimators: Optional[Tuple[str, ...]] = tuple(MAIN_TEXT_ESTIMATOR_ORDER)
    elif args.estimators is None:
        selected_estimators = None
    else:
        unknown = sorted(set(args.estimators) - set(ALL_ESTIMATORS))
        if unknown:
            parser.error(f"Unknown estimators: {', '.join(unknown)}")
        selected_estimators = tuple(args.estimators)

    raw_rows: List[Dict[str, float]] = []
    for config in experiments:
        prepared = prepare_mean_dataset(config, args.cache_dir)
        y_total = prepared["Y_total"]
        yhat_total = prepared["Yhat_total"]
        x_total = prepared.get("X_total")
        true_theta = float(np.mean(y_total))
        seed = config.default_seed if args.seed is None else args.seed
        alpha = config.alpha if args.alpha is None else args.alpha
        replications = config.replications if args.replications is None else args.replications
        n_grid = config.n_grid

        if args.smoke:
            replications = min(replications, 2)
            n_grid = n_grid[:2]

        rng = np.random.default_rng(seed)
        for n_labeled in n_grid:
            if n_labeled >= y_total.shape[0]:
                raise ValueError(
                    f"Requested n={n_labeled} for {config.name}, but only {y_total.shape[0]} labels are available."
                )
            for replicate in range(replications):
                permutation = rng.permutation(y_total.shape[0])
                raw_rows.extend(
                    build_trial_results(
                        config=config,
                        alpha=alpha,
                        replicate=replicate,
                        n_labeled=int(n_labeled),
                        true_theta=true_theta,
                        y_total=y_total,
                        yhat_total=yhat_total,
                        permutation=permutation,
                        x_total=x_total,
                        selected_estimators=selected_estimators,
                    )
                )

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["dataset", "n_labeled", "replicate", "estimator"]
    ).reset_index(drop=True)
    summary_df = summarize_results(raw_df)

    raw_df.to_csv(args.output_dir / "raw_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    if args.main_text_only:
        plot_main_text_grid(summary_df, args.output_dir)
        return
    write_dataset_tables(summary_df, args.output_dir)
    write_paper_summary_table(summary_df, args.output_dir)
    plot_dataset_metric(summary_df, args.output_dir, "mean_bias", "Bias", "bias")
    plot_dataset_metric(summary_df, args.output_dir, "emp_variance", "Variance", "variance")
    plot_dataset_metric(summary_df, args.output_dir, "mse", "MSE", "mse")
    plot_dataset_metric(summary_df, args.output_dir, "mse_ratio_vs_ppi", "MSE / PPI MSE", "mse_vs_ppi")
    plot_dataset_metric(summary_df, args.output_dir, "rel_eff_vs_ppi", "Rel. eff. vs PPI", "relative_efficiency")
    plot_dataset_metric(summary_df, args.output_dir, "coverage", "Coverage", "coverage")
    plot_main_text_grid(summary_df, args.output_dir)
    plot_appendix_calibration_grid(summary_df, args.output_dir)
    plot_overall_metric_grid(summary_df, args.output_dir)


if __name__ == "__main__":
    main()
