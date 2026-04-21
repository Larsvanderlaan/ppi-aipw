from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.estimators import (
    TuningGrid,
    aipw_em_result,
    aipp_from_prediction,
    auto_aipw_pointestimate_and_se,
    fit_linear_calibration,
    fit_monotone_spline_calibration,
    fit_platt_calibration,
    fit_sklearn_isotonic_calibration,
    fit_venn_abers_calibration,
    influence_se_from_prediction,
    influence_se_labeled_only,
    labeled_mean,
    plugin_estimate,
    plugin_se_from_prediction,
    ppi_mean_from_prediction,
    ppi_mean_se_from_prediction,
    predict_linear,
    predict_monotone_spline,
    predict_platt,
    predict_sklearn_isotonic,
    predict_venn_abers,
)

ALL_ESTIMATORS: Tuple[str, ...] = (
    "labeled_only",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "aipw",
    "auto_calibration",
    "monotone_spline",
    "affine_calibration",
    "platt_calibration",
    "isotonic_calibration",
    "calibrated_plugin",
)

MAIN_TEXT_ESTIMATORS: Tuple[str, ...] = (
    "labeled_only",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "affine_calibration",
    "isotonic_calibration",
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


@dataclass
class Profile:
    name: str
    replications: int
    n_grid: Tuple[int, ...]
    unlabeled_ratio_grid: Tuple[int, ...]
    tuning_grid: TuningGrid
    cv_folds_small: int
    cv_folds_large: int
    n_estimators: int
    dgp_grid: Tuple[str, ...]


PROFILES: Dict[str, Profile] = {
    "quick": Profile(
        name="quick",
        replications=20,
        n_grid=(50, 100, 200, 400, 800, 1200, 2400),
        unlabeled_ratio_grid=(1, 4, 16),
        tuning_grid=TuningGrid(
            min_data_in_leaf=(15,),
            num_leaves=(200,),
            max_depth=(-1,),
        ),
        cv_folds_small=0,
        cv_folds_large=0,
        n_estimators=75,
        dgp_grid=("monotone",),
    ),
    "pilot": Profile(
        name="pilot",
        replications=50,
        n_grid=(50, 100, 200, 400, 800, 1200, 2400),
        unlabeled_ratio_grid=(1, 4, 16),
        tuning_grid=TuningGrid(
            min_data_in_leaf=(15,),
            num_leaves=(200,),
            max_depth=(-1,),
        ),
        cv_folds_small=0,
        cv_folds_large=0,
        n_estimators=75,
        dgp_grid=("monotone",),
    ),
    "paper": Profile(
        name="paper",
        replications=300,
        n_grid=(50, 100, 200, 400, 800, 1200, 2400),
        unlabeled_ratio_grid=(1, 16),
        tuning_grid=TuningGrid(
            min_data_in_leaf=(15,),
            num_leaves=(200,),
            max_depth=(-1,),
        ),
        cv_folds_small=0,
        cv_folds_large=0,
        n_estimators=125,
        dgp_grid=("monotone",),
    ),
}


@dataclass(frozen=True)
class ScoreSetting:
    quality: str
    family: str
    mu_slope: float = 5.0
    well_noise_sd: Optional[float] = None
    poor_a: Optional[float] = None
    poor_b: Optional[float] = None
    poor_shift: Optional[float] = None
    poor_scale: Optional[float] = None
    poor_noise_sd: Optional[float] = None
    poor_gamma: Optional[float] = None

    @property
    def setting_id(self) -> str:
        if self.quality == "well_calibrated":
            return f"well_noise_{self.well_noise_sd:.3f}"
        if self.family == "cubic_logit":
            return (
                f"poor_cubic_a{self.poor_a:.1f}_c{self.poor_gamma:.3f}_b{self.poor_b:.1f}"
                f"_shift{self.poor_shift:.2f}_scale{self.poor_scale:.2f}"
                f"_noise{self.poor_noise_sd:.3f}"
            )
        return (
            f"poor_logit_a{self.poor_a:.1f}_b{self.poor_b:.1f}"
            f"_shift{self.poor_shift:.2f}_scale{self.poor_scale:.2f}"
            f"_noise{self.poor_noise_sd:.3f}"
        )


DEFAULT_SCORE_SETTINGS = {
    "well_calibrated": ScoreSetting(
        quality="well_calibrated",
        family="identity",
        mu_slope=5.0,
        well_noise_sd=0.01,
    ),
    "poorly_calibrated": ScoreSetting(
        quality="poorly_calibrated",
        family="logit_tilt",
        mu_slope=5.0,
        poor_a=8.0,
        poor_b=-3.0,
        poor_shift=-0.15,
        poor_scale=1.20,
        poor_noise_sd=0.01,
        poor_gamma=None,
    ),
}


def candidate_score_settings() -> List[ScoreSetting]:
    return [
        ScoreSetting(
            quality="well_calibrated",
            family="identity",
            mu_slope=5.0,
            well_noise_sd=0.005,
        ),
        ScoreSetting(
            quality="poorly_calibrated",
            family="logit_tilt",
            mu_slope=5.0,
            poor_a=3.0,
            poor_b=-1.0,
            poor_shift=-0.10,
            poor_scale=0.85,
            poor_noise_sd=0.005,
        ),
    ]


def pilot_score_settings() -> List[ScoreSetting]:
    settings: List[ScoreSetting] = [
        ScoreSetting(
            quality="well_calibrated",
            family="identity",
            mu_slope=5.0,
            well_noise_sd=eps_sd,
        )
        for eps_sd in (0.005, 0.01)
    ]
    settings.extend(
        ScoreSetting(
            quality="poorly_calibrated",
            family="logit_tilt",
            mu_slope=5.0,
            poor_a=a,
            poor_b=b,
            poor_shift=shift,
            poor_scale=scale,
            poor_noise_sd=eps_sd,
        )
        for a in (2.0, 3.0, 4.0)
        for b in (-1.5, -1.0, -0.5)
        for shift in (-0.15, -0.10, -0.05)
        for scale in (0.75, 0.85, 0.95)
        for eps_sd in (0.0, 0.005)
    )
    return settings


def build_features(s: np.ndarray, rng: np.random.Generator, p: int = 10) -> np.ndarray:
    eps = rng.normal(size=(len(s), p))
    x = np.empty((len(s), p), dtype=float)
    x[:, 0] = s + 0.2 * eps[:, 0]
    x[:, 1] = s**2 + 0.2 * eps[:, 1]
    x[:, 2] = np.sin(s) + 0.2 * eps[:, 2]
    x[:, 3:] = eps[:, 3:]
    return x


PSI0_CACHE = {
    "monotone": 0.5,
    "nonmonotone": float(
        np.mean(
            np.clip(
                0.5 + 0.3 * np.sin(1.75 * np.random.default_rng(9102).normal(size=300_000)),
                0.05,
                0.95,
            )
        )
    ),
}


def mu_function(s: np.ndarray, dgp: str, mu_slope: float = 5.0) -> np.ndarray:
    if dgp == "monotone":
        return sigmoid(mu_slope * s)
    if dgp == "nonmonotone":
        return np.clip(0.5 + 0.3 * np.sin(1.75 * s), 0.05, 0.95)
    raise ValueError(f"Unknown dgp: {dgp}")


def score_from_mu(
    mu: np.ndarray,
    rng: np.random.Generator,
    setting: ScoreSetting,
) -> np.ndarray:
    if setting.quality == "well_calibrated":
        nearly_calibrated = np.clip(
            mu + float(setting.well_noise_sd) * rng.normal(size=len(mu)),
            0.01,
            0.99,
        )
        raw = sigmoid(1.5 * logit(nearly_calibrated))
    elif setting.family == "logit_tilt":
        distorted = sigmoid(float(setting.poor_a) * logit(mu) + float(setting.poor_b))
        raw = (
            float(setting.poor_shift)
            + float(setting.poor_scale) * distorted
            + float(setting.poor_noise_sd) * rng.normal(size=len(mu))
        )
    elif setting.family == "cubic_logit":
        z = logit(mu)
        distorted = sigmoid(
            float(setting.poor_a) * z
            + float(setting.poor_gamma) * np.power(z, 3.0)
            + float(setting.poor_b)
        )
        raw = (
            float(setting.poor_shift)
            + float(setting.poor_scale) * distorted
            + float(setting.poor_noise_sd) * rng.normal(size=len(mu))
        )
    elif setting.family == "power":
        distorted = 0.02 + 0.96 * np.power(mu, float(setting.poor_gamma))
        raw = distorted + float(setting.poor_noise_sd) * rng.normal(size=len(mu))
    else:
        raise ValueError(f"Unknown score setting: {setting}")
    return np.clip(raw, 0.01, 0.99)


def score_function(
    s: np.ndarray,
    dgp: str,
    rng: np.random.Generator,
    setting: ScoreSetting,
) -> np.ndarray:
    return score_from_mu(mu_function(s, dgp, setting.mu_slope), rng, setting)


def analytic_rho_grid() -> Tuple[float, ...]:
    return (0.5, 1.0 / 17.0)


def variance_for_score(
    y: np.ndarray,
    pred: np.ndarray,
    psi0: float,
    rho: float,
) -> float:
    d_l = pred - psi0 + (y - pred) / rho
    d_u = pred - psi0
    return float(rho * np.mean(d_l**2) + (1.0 - rho) * np.mean(d_u**2))


def population_linear_prediction(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([score, np.ones_like(score)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coef


def build_if_pilot_summary(
    seed: int,
    pop_size: int = 400_000,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for idx, setting in enumerate(pilot_score_settings()):
        rng = np.random.default_rng(seed + 50_000 + idx)
        s = rng.normal(size=pop_size)
        mu = sigmoid(setting.mu_slope * s)
        y = rng.binomial(1, mu).astype(float)
        raw_score = score_from_mu(mu, rng, setting)
        calibrated_score = mu
        linear_score = np.clip(population_linear_prediction(raw_score, y), 0.01, 0.99)
        psi0 = float(np.mean(mu))
        row: Dict[str, float] = {
            "score_quality": setting.quality,
            "score_setting_id": setting.setting_id,
            "score_family": setting.family,
            "well_noise_sd": setting.well_noise_sd,
            "poor_a": setting.poor_a,
            "poor_b": setting.poor_b,
            "poor_noise_sd": setting.poor_noise_sd,
            "poor_gamma": setting.poor_gamma,
            "mu_slope": setting.mu_slope,
            "psi0": psi0,
            "poor_shift": setting.poor_shift,
            "poor_scale": setting.poor_scale,
            "var_y_minus_m": float(np.var(y - raw_score)),
            "var_y_minus_mlin": float(np.var(y - linear_score)),
            "var_y_minus_mstar": float(np.var(y - calibrated_score)),
            "var_m_centered": float(np.var(raw_score - psi0)),
            "var_mlin_centered": float(np.var(linear_score - psi0)),
            "var_mstar_centered": float(np.var(calibrated_score - psi0)),
            "linear_residual_variance_reduction_pct": float(
                100.0 * (1.0 - np.var(y - linear_score) / np.var(y - raw_score))
            ),
            "residual_variance_reduction_pct": float(
                100.0 * (1.0 - np.var(y - calibrated_score) / np.var(y - raw_score))
            ),
            "nonlinear_extra_residual_reduction_pct": float(
                100.0 * (1.0 - np.var(y - calibrated_score) / np.var(y - linear_score))
            ),
            "mu_q01": float(np.quantile(mu, 0.01)),
            "mu_q99": float(np.quantile(mu, 0.99)),
        }
        gains = []
        for rho in analytic_rho_grid():
            key = f"{rho:.4f}".replace(".", "_")
            var_ppi = variance_for_score(y, raw_score, psi0, rho)
            var_cal = variance_for_score(y, calibrated_score, psi0, rho)
            gain_pct = 100.0 * (1.0 - var_cal / var_ppi)
            row[f"var_eif_ppi_rho_{key}"] = float(var_ppi)
            row[f"var_eif_cal_rho_{key}"] = float(var_cal)
            row[f"gain_pct_rho_{key}"] = float(gain_pct)
            gains.append(gain_pct)
        row["avg_gain_pct"] = float(np.mean(gains))
        row["min_gain_pct"] = float(np.min(gains))
        row["max_gain_pct"] = float(np.max(gains))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["score_quality", "score_setting_id"]).reset_index(drop=True)


def clean_json_record(record: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def select_if_recommendations(if_summary: pd.DataFrame) -> Dict[str, object]:
    well = if_summary[if_summary["score_quality"] == "well_calibrated"].copy()
    poor = if_summary[if_summary["score_quality"] == "poorly_calibrated"].copy()

    well["selection_score"] = (
        np.maximum(well["avg_gain_pct"] - 5.0, 0.0)
        + np.abs(well["avg_gain_pct"] - 2.5)
        + 5.0 * np.maximum(0.01 - well["mu_q01"], 0.0)
        + 5.0 * np.maximum(well["mu_q99"] - 0.99, 0.0)
    )
    poor["selection_score"] = (
        3.0 * np.maximum(10.0 - poor["avg_gain_pct"], 0.0)
        + 3.0 * np.maximum(poor["avg_gain_pct"] - 20.0, 0.0)
        + 0.25 * np.abs(poor["avg_gain_pct"] - 15.0)
        - 0.02 * poor["residual_variance_reduction_pct"]
        + 5.0 * np.maximum(0.01 - poor["mu_q01"], 0.0)
        + 5.0 * np.maximum(poor["mu_q99"] - 0.99, 0.0)
    )

    rec_well = well.sort_values("selection_score").iloc[0]
    rec_poor = poor.sort_values("selection_score").iloc[0]
    return {
        "well_calibrated": clean_json_record(rec_well.to_dict()),
        "poorly_calibrated": clean_json_record(rec_poor.to_dict()),
    }


def build_mc_pilot_summary(
    seed: int,
    poor_candidates: List[ScoreSetting],
    well_candidates: List[ScoreSetting],
    replications: int = 200,
    regimes: Tuple[Tuple[int, int], ...] = ((50, 1), (100, 1), (200, 1), (400, 1), (50, 16), (100, 16), (200, 16), (400, 16)),
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    candidate_id = 0
    all_candidates = well_candidates + poor_candidates
    candidate_group_cols = [
        "score_quality",
        "score_setting_id",
        "score_family",
        "mu_slope",
        "well_noise_sd",
        "poor_a",
        "poor_b",
        "poor_shift",
        "poor_scale",
        "poor_noise_sd",
        "poor_gamma",
    ]
    for setting in all_candidates:
        regime_results: List[Dict[str, float]] = []
        for n, ratio in regimes:
            N = n * ratio
            for rep in range(replications):
                rng_seed = seed + 200_000 + 10_000 * candidate_id + 100 * n + rep
                rng = np.random.default_rng(rng_seed)
                sample = generate_sample(n=n, N=N, dgp="monotone", setting=setting, rng=rng)
                results, diagnostics = run_one(sample, PROFILES["quick"], rng_seed=rng_seed)
                for result in results:
                    regime_results.append(
                        {
                            "score_quality": setting.quality,
                            "score_setting_id": setting.setting_id,
                            "score_family": setting.family,
                            "mu_slope": setting.mu_slope,
                            "well_noise_sd": setting.well_noise_sd,
                            "poor_a": setting.poor_a,
                            "poor_b": setting.poor_b,
                            "poor_shift": setting.poor_shift,
                            "poor_scale": setting.poor_scale,
                            "poor_noise_sd": setting.poor_noise_sd,
                            "poor_gamma": setting.poor_gamma,
                            "n_labeled": n,
                            "unlabeled_ratio": ratio,
                            "replicate": rep,
                            **result,
                            **diagnostics,
                        }
                    )
        frame = pd.DataFrame(regime_results)
        grouped = (
            frame.groupby(
                [
                    "score_quality",
                    "score_setting_id",
                    "score_family",
                    "mu_slope",
                    "well_noise_sd",
                    "poor_a",
                    "poor_b",
                    "poor_shift",
                    "poor_scale",
                    "poor_noise_sd",
                    "poor_gamma",
                    "n_labeled",
                    "unlabeled_ratio",
                    "estimator",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
                coverage=("covered", "mean"),
                mean_ci_length=("ci_length", "mean"),
            )
        )
        pivot = grouped.pivot(
            index=candidate_group_cols + ["n_labeled", "unlabeled_ratio"],
            columns="estimator",
            values=["rmse", "coverage", "mean_ci_length"],
        )
        pivot.columns = [f"{metric}_{est}" for metric, est in pivot.columns]
        pivot = pivot.reset_index()
        pivot["best_calibrated_rmse"] = pivot[
            [
                "rmse_affine_calibration",
                "rmse_platt_calibration",
                "rmse_isotonic_calibration",
                "rmse_calibrated_plugin",
            ]
        ].min(axis=1)
        pivot["best_nonlinear_rmse"] = pivot[
            ["rmse_isotonic_calibration", "rmse_calibrated_plugin"]
        ].min(axis=1)
        pivot["gain_vs_ppi_pct"] = 100.0 * (1.0 - pivot["best_calibrated_rmse"] / pivot["rmse_ppi"])
        pivot["gain_vs_label_pct"] = 100.0 * (
            1.0 - pivot["best_calibrated_rmse"] / pivot["rmse_labeled_only"]
        )
        pivot["linear_gain_vs_ppi_pct"] = 100.0 * (
            1.0 - pivot["rmse_affine_calibration"] / pivot["rmse_ppi"]
        )
        pivot["nonlinear_gain_vs_linear_pct"] = 100.0 * (
            1.0 - pivot["best_nonlinear_rmse"] / pivot["rmse_affine_calibration"]
        )
        pivot["venn_gain_vs_iso_pct"] = 100.0 * (
            1.0 - pivot["rmse_calibrated_plugin"] / pivot["rmse_isotonic_calibration"]
        )
        pivot["venn_ci_gain_vs_iso_pct"] = 100.0 * (
            1.0 - pivot["mean_ci_length_calibrated_plugin"] / pivot["mean_ci_length_isotonic_calibration"]
        )
        pivot["min_score_coverage"] = pivot[
            [
                "coverage_ppi",
                "coverage_affine_calibration",
                "coverage_platt_calibration",
                "coverage_isotonic_calibration",
                "coverage_calibrated_plugin",
            ]
        ].min(axis=1)
        overall = (
            pivot.groupby(candidate_group_cols, as_index=False, dropna=False)
            .agg(
                mc_avg_gain_vs_ppi_pct=("gain_vs_ppi_pct", "mean"),
                mc_min_gain_vs_ppi_pct=("gain_vs_ppi_pct", "min"),
                mc_max_gain_vs_ppi_pct=("gain_vs_ppi_pct", "max"),
                mc_avg_gain_vs_label_pct=("gain_vs_label_pct", "mean"),
                mc_avg_linear_gain_vs_ppi_pct=("linear_gain_vs_ppi_pct", "mean"),
                mc_avg_nonlinear_gain_vs_linear_pct=("nonlinear_gain_vs_linear_pct", "mean"),
                mc_min_score_coverage=("min_score_coverage", "min"),
            )
        )
        small = pivot[pivot["n_labeled"].isin([50, 100, 200])].copy()
        small_summary = (
            small.groupby(candidate_group_cols, as_index=False, dropna=False)
            .agg(
                mc_small_avg_gain_vs_ppi_pct=("gain_vs_ppi_pct", "mean"),
                mc_small_avg_linear_gain_vs_ppi_pct=("linear_gain_vs_ppi_pct", "mean"),
                mc_small_avg_nonlinear_gain_vs_linear_pct=("nonlinear_gain_vs_linear_pct", "mean"),
                mc_small_n_venn_gain_vs_iso_pct=("venn_gain_vs_iso_pct", "mean"),
                mc_small_n_venn_ci_gain_vs_iso_pct=("venn_ci_gain_vs_iso_pct", "mean"),
            )
        )
        diag_frames = [
            frame.groupby(candidate_group_cols, as_index=False, dropna=False).agg(
                mc_avg_raw_residual_var=("raw_residual_var", "mean"),
                mc_avg_affine_residual_var=("affine_residual_var", "mean"),
                mc_avg_platt_residual_var=("platt_residual_var", "mean"),
                mc_avg_isotonic_residual_var=("isotonic_residual_var", "mean"),
                mc_avg_nonlinear_residual_var=("nonlinear_residual_var", "mean"),
            )
        ]
        merged = overall.merge(small_summary, on=candidate_group_cols, how="left")
        for diag in diag_frames:
            merged = merged.merge(diag, on=candidate_group_cols, how="left")
        rows.append(merged)
        candidate_id += 1
    return pd.concat(rows, ignore_index=True).sort_values(
        ["score_quality", "score_setting_id"]
    ).reset_index(drop=True)


def shortlist_candidates_from_if(if_summary: pd.DataFrame) -> Tuple[List[ScoreSetting], List[ScoreSetting]]:
    well = (
        if_summary[if_summary["score_quality"] == "well_calibrated"]
        .sort_values(["avg_gain_pct", "well_noise_sd"])
        .head(2)
    )
    poor = (
        if_summary[
            (if_summary["score_quality"] == "poorly_calibrated")
            & (if_summary["avg_gain_pct"] >= 10.0)
            & (if_summary["avg_gain_pct"] <= 40.0)
            & (if_summary["residual_variance_reduction_pct"] >= 10.0)
            & (if_summary["linear_residual_variance_reduction_pct"] >= 3.0)
            & (if_summary["nonlinear_extra_residual_reduction_pct"] >= 2.0)
        ]
        .sort_values(
            by=[
                "linear_residual_variance_reduction_pct",
                "nonlinear_extra_residual_reduction_pct",
                "avg_gain_pct",
            ],
            ascending=[False, False, False],
        )
        .head(12)
    )
    return (
        [score_setting_from_record(row) for _, row in well.iterrows()],
        [score_setting_from_record(row) for _, row in poor.iterrows()],
    )


def select_final_recommendations(
    if_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
) -> Dict[str, object]:
    well = mc_summary[mc_summary["score_quality"] == "well_calibrated"].merge(
        if_summary,
        on=[
            "score_quality",
            "score_setting_id",
            "score_family",
            "mu_slope",
            "well_noise_sd",
            "poor_a",
            "poor_b",
            "poor_shift",
            "poor_scale",
            "poor_noise_sd",
            "poor_gamma",
        ],
        how="left",
        suffixes=("", "_if"),
    )
    poor = mc_summary[mc_summary["score_quality"] == "poorly_calibrated"].merge(
        if_summary,
        on=[
            "score_quality",
            "score_setting_id",
            "score_family",
            "mu_slope",
            "well_noise_sd",
            "poor_a",
            "poor_b",
            "poor_shift",
            "poor_scale",
            "poor_noise_sd",
            "poor_gamma",
        ],
        how="left",
        suffixes=("", "_if"),
    )

    well["selection_score"] = (
        np.abs(well["mc_avg_gain_vs_ppi_pct"] - 1.0)
        + 25.0 * np.maximum(0.9 - well["mc_min_score_coverage"], 0.0)
        + 0.25 * np.abs(well["mc_avg_linear_gain_vs_ppi_pct"])
        + 0.25 * np.abs(well["mc_avg_nonlinear_gain_vs_linear_pct"])
    )
    poor["selection_score"] = (
        6.0 * np.maximum(5.0 - poor["mc_small_avg_linear_gain_vs_ppi_pct"], 0.0)
        + 5.0 * np.maximum(2.0 - poor["mc_small_avg_nonlinear_gain_vs_linear_pct"], 0.0)
        + 5.0 * np.maximum(0.0 - poor["mc_small_n_venn_gain_vs_iso_pct"], 0.0)
        + 2.0 * np.maximum(0.0 - poor["mc_small_n_venn_ci_gain_vs_iso_pct"], 0.0)
        + 40.0 * np.maximum(0.90 - poor["mc_min_score_coverage"], 0.0)
        + 2.0 * np.maximum(0.0 - poor["mc_avg_gain_vs_ppi_pct"], 0.0)
        + 0.15 * np.abs(poor["avg_gain_pct"] - poor["mc_avg_gain_vs_ppi_pct"])
        - 0.2 * poor["mc_small_avg_linear_gain_vs_ppi_pct"]
        - 0.15 * poor["mc_small_avg_nonlinear_gain_vs_linear_pct"]
        - 0.1 * poor["mc_small_n_venn_gain_vs_iso_pct"]
    )

    rec_poor = poor.sort_values("selection_score").iloc[0]
    well_same_slope = well[well["mu_slope"] == rec_poor["mu_slope"]].copy()
    if well_same_slope.empty:
        well_same_slope = well
    rec_well = well_same_slope.sort_values("selection_score").iloc[0]
    return {
        "well_calibrated": clean_json_record(rec_well.to_dict()),
        "poorly_calibrated": clean_json_record(rec_poor.to_dict()),
    }


def score_setting_from_record(record: Dict[str, object]) -> ScoreSetting:
    return ScoreSetting(
        quality=str(record["score_quality"]),
        family=str(record["score_family"]),
        mu_slope=float(record.get("mu_slope", 4.0)),
        well_noise_sd=record.get("well_noise_sd"),
        poor_a=record.get("poor_a"),
        poor_b=record.get("poor_b"),
        poor_shift=record.get("poor_shift"),
        poor_scale=record.get("poor_scale"),
        poor_noise_sd=record.get("poor_noise_sd"),
        poor_gamma=record.get("poor_gamma"),
    )


def selected_score_settings(recommendations: Dict[str, object]) -> List[ScoreSetting]:
    return [
        score_setting_from_record(recommendations["well_calibrated"]),
        score_setting_from_record(recommendations["poorly_calibrated"]),
    ]


def generate_sample(
    n: int,
    N: int,
    dgp: str,
    setting: ScoreSetting,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    s_l = rng.normal(size=n)
    s_u = rng.normal(size=N)
    mu_l = mu_function(s_l, dgp, setting.mu_slope)
    y_l = rng.binomial(1, mu_l).astype(float)
    score_l = score_function(s_l, dgp, rng, setting)
    score_u = score_function(s_u, dgp, rng, setting)
    return {
        "score_l": score_l,
        "score_u": score_u,
        "y_l": y_l,
        "psi0": PSI0_CACHE[dgp],
    }


def ci_bounds(estimate: float, se: float) -> Tuple[float, float]:
    return float(estimate - 1.96 * se), float(estimate + 1.96 * se)


def official_ppi_summary(
    y_l: np.ndarray,
    score_l: np.ndarray,
    score_u: np.ndarray,
    lam: Optional[float],
) -> Tuple[float, float, float, float]:
    estimate = float(np.asarray(ppi_mean_pointestimate(y_l, score_l, score_u, lam=lam)).reshape(-1)[0])
    lower, upper = ppi_mean_ci(y_l, score_l, score_u, alpha=0.05, lam=lam)
    lower_scalar = float(np.asarray(lower).reshape(-1)[0])
    upper_scalar = float(np.asarray(upper).reshape(-1)[0])
    se = float((upper_scalar - lower_scalar) / (2.0 * 1.959963984540054))
    return estimate, se, lower_scalar, upper_scalar


def summarize_result(
    estimator: str,
    estimate: float,
    se: float,
    psi0: float,
    lower: float,
    upper: float,
) -> Dict[str, float]:
    return {
        "estimator": estimator,
        "estimate": estimate,
        "se": se,
        "bias": estimate - psi0,
        "sq_error": (estimate - psi0) ** 2,
        "covered": float(lower <= psi0 <= upper),
        "ci_length": upper - lower,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def run_one(
    sample: Dict[str, np.ndarray],
    profile: Profile,
    rng_seed: int,
    estimators: Tuple[str, ...] = ALL_ESTIMATORS,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    score_l = sample["score_l"]
    score_u = sample["score_u"]
    y_l = sample["y_l"]
    psi0 = sample["psi0"]
    selected = set(estimators)
    results: List[Dict[str, float]] = []
    diagnostics: Dict[str, float] = {
        "raw_calibration_mse": float(np.mean((y_l - score_l) ** 2)),
        "raw_residual_var": float(np.var(y_l - score_l)),
    }

    psi_ppi: Optional[float] = None
    pred_lin_l: Optional[np.ndarray] = None
    pred_platt_l: Optional[np.ndarray] = None
    pred_iso_l: Optional[np.ndarray] = None
    pred_cal_l: Optional[np.ndarray] = None

    if "labeled_only" in selected:
        psi_label = labeled_mean(y_l)
        se_label = influence_se_labeled_only(y_l, psi_label)
        lo, hi = ci_bounds(psi_label, se_label)
        results.append(summarize_result("labeled_only", psi_label, se_label, psi0, lo, hi))

    if "ppi" in selected or "calibrated_plugin" in selected:
        psi_ppi, se_ppi, lo, hi = official_ppi_summary(y_l, score_l, score_u, lam=1)
        if "ppi" in selected:
            results.append(summarize_result("ppi", psi_ppi, se_ppi, psi0, lo, hi))

    if "ppi_plus_plus" in selected:
        psi_ppi_plus_plus, se_ppi_plus_plus, lo, hi = official_ppi_summary(y_l, score_l, score_u, lam=None)
        results.append(summarize_result("ppi_plus_plus", psi_ppi_plus_plus, se_ppi_plus_plus, psi0, lo, hi))

    if "aipw_em" in selected:
        aipw_em = aipw_em_result(y_l, score_l, score_u, alpha=0.05)
        results.append(
            summarize_result(
                "aipw_em",
                aipw_em["estimate"],
                aipw_em["se"],
                psi0,
                aipw_em["ci_lower"],
                aipw_em["ci_upper"],
            )
        )

    if "aipw" in selected:
        psi_aipw = aipp_from_prediction(score_l, score_u, y_l)
        se_aipw = influence_se_from_prediction(psi_aipw, score_l, score_u, y_l)
        lo, hi = ci_bounds(psi_aipw, se_aipw)
        results.append(summarize_result("aipw", psi_aipw, se_aipw, psi0, lo, hi))

    if "auto_calibration" in selected:
        auto_result = auto_aipw_pointestimate_and_se(
            y_l,
            score_l,
            score_u,
            candidate_methods=("aipw", "linear", "monotone_spline", "isocal"),
            num_folds=20,
            random_state=rng_seed,
        )
        lo, hi = ci_bounds(auto_result["estimate"], auto_result["se"])
        results.append(
            summarize_result(
                "auto_calibration",
                auto_result["estimate"],
                auto_result["se"],
                psi0,
                lo,
                hi,
            )
        )

    if "monotone_spline" in selected:
        monotone_spline_model = fit_monotone_spline_calibration(score_l, y_l)
        pred_spline_l = predict_monotone_spline(monotone_spline_model, score_l)
        pred_spline_u = predict_monotone_spline(monotone_spline_model, score_u)
        psi_spline = plugin_estimate(pred_spline_l, pred_spline_u)
        se_spline = influence_se_from_prediction(psi_spline, pred_spline_l, pred_spline_u, y_l)
        lo, hi = ci_bounds(psi_spline, se_spline)
        results.append(summarize_result("monotone_spline", psi_spline, se_spline, psi0, lo, hi))

    if "affine_calibration" in selected:
        linear_model = fit_linear_calibration(score_l, y_l)
        pred_lin_l = predict_linear(linear_model, score_l)
        pred_lin_u = predict_linear(linear_model, score_u)
        psi_lin = plugin_estimate(pred_lin_l, pred_lin_u)
        se_lin = influence_se_from_prediction(psi_lin, pred_lin_l, pred_lin_u, y_l)
        lo, hi = ci_bounds(psi_lin, se_lin)
        results.append(summarize_result("affine_calibration", psi_lin, se_lin, psi0, lo, hi))
        diagnostics.update(
            {
                "affine_calibration_mse": float(np.mean((y_l - pred_lin_l) ** 2)),
                "affine_residual_var": float(np.var(y_l - pred_lin_l)),
            }
        )

    if "platt_calibration" in selected:
        platt_model = fit_platt_calibration(score_l, y_l)
        pred_platt_l = predict_platt(platt_model, score_l)
        pred_platt_u = predict_platt(platt_model, score_u)
        psi_platt = plugin_estimate(pred_platt_l, pred_platt_u)
        se_platt = influence_se_from_prediction(psi_platt, pred_platt_l, pred_platt_u, y_l)
        lo, hi = ci_bounds(psi_platt, se_platt)
        results.append(summarize_result("platt_calibration", psi_platt, se_platt, psi0, lo, hi))
        diagnostics.update(
            {
                "platt_calibration_mse": float(np.mean((y_l - pred_platt_l) ** 2)),
                "platt_residual_var": float(np.var(y_l - pred_platt_l)),
            }
        )

    if "isotonic_calibration" in selected:
        iso_model = fit_sklearn_isotonic_calibration(score_l, y_l)
        pred_iso_l = predict_sklearn_isotonic(iso_model, score_l)
        pred_iso_u = predict_sklearn_isotonic(iso_model, score_u)
        psi_iso = plugin_estimate(pred_iso_l, pred_iso_u)
        se_iso = influence_se_from_prediction(psi_iso, pred_iso_l, pred_iso_u, y_l)
        lo, hi = ci_bounds(psi_iso, se_iso)
        results.append(summarize_result("isotonic_calibration", psi_iso, se_iso, psi0, lo, hi))
        diagnostics.update(
            {
                "isotonic_calibration_mse": float(np.mean((y_l - pred_iso_l) ** 2)),
                "isotonic_residual_var": float(np.var(y_l - pred_iso_l)),
            }
        )

    if "calibrated_plugin" in selected:
        if psi_ppi is None:
            psi_ppi, _, _, _ = official_ppi_summary(y_l, score_l, score_u, lam=1)
        ppi_reference = float(psi_ppi)
        reference_l = np.full_like(score_l, ppi_reference, dtype=float)
        reference_u = np.full_like(score_u, ppi_reference, dtype=float)
        calib_model = fit_venn_abers_calibration(score_l=score_l, y_l=y_l)
        pred_cal_l = predict_venn_abers(calib_model, score_l, reference_l)
        pred_cal_u = predict_venn_abers(calib_model, score_u, reference_u)
        psi_cal = plugin_estimate(pred_cal_l, pred_cal_u)
        se_cal = influence_se_from_prediction(psi_cal, pred_cal_l, pred_cal_u, y_l)
        lo, hi = ci_bounds(psi_cal, se_cal)
        results.append(summarize_result("calibrated_plugin", psi_cal, se_cal, psi0, lo, hi))
        diagnostics.update(
            {
                "ppi_reference": ppi_reference,
                "nonlinear_calibration_mse": float(np.mean((y_l - pred_cal_l) ** 2)),
                "nonlinear_residual_var": float(np.var(y_l - pred_cal_l)),
            }
        )
    return results, diagnostics


GROUP_COLS = [
    "dgp",
    "score_quality",
    "score_setting_id",
    "score_family",
    "mu_slope",
    "well_noise_sd",
    "poor_a",
    "poor_b",
    "poor_shift",
    "poor_scale",
    "poor_noise_sd",
    "poor_gamma",
    "n_labeled",
    "n_unlabeled",
    "unlabeled_ratio",
]


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw.groupby(
            GROUP_COLS + ["estimator"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_estimate=("estimate", "mean"),
            mean_bias=("bias", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            emp_sd=("estimate", "std"),
            mean_se=("se", "mean"),
            coverage=("covered", "mean"),
            mean_ci_length=("ci_length", "mean"),
        )
    )
    return grouped


def add_efficiency_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    merged = summary.copy()
    label = merged[merged["estimator"] == "labeled_only"][
        GROUP_COLS + ["rmse", "mean_ci_length"]
    ].rename(columns={"rmse": "rmse_label", "mean_ci_length": "ci_label"})
    ppi = merged[merged["estimator"] == "ppi"][
        GROUP_COLS + ["rmse"]
    ].rename(columns={"rmse": "rmse_ppi"})
    merged = merged.merge(label, on=GROUP_COLS, how="left")
    merged = merged.merge(ppi, on=GROUP_COLS, how="left")
    merged["rmse_ratio_vs_label"] = merged["rmse"] / merged["rmse_label"]
    merged["rmse_ratio_vs_ppi"] = merged["rmse"] / merged["rmse_ppi"]
    merged["ci_ratio_vs_label"] = merged["mean_ci_length"] / merged["ci_label"]
    return merged


def build_pilot_summary(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    metric_summary = (
        summary.groupby(
            [
                "score_quality",
                "score_setting_id",
                "score_family",
                "mu_slope",
                "well_noise_sd",
                "poor_a",
                "poor_b",
                "poor_shift",
                "poor_scale",
                "poor_noise_sd",
                "poor_gamma",
                "estimator",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_ci_length=("mean_ci_length", "mean"),
        )
    )
    pivot = metric_summary.pivot(
        index=[
            "score_quality",
            "score_setting_id",
            "score_family",
            "mu_slope",
            "well_noise_sd",
            "poor_a",
            "poor_b",
            "poor_shift",
            "poor_scale",
            "poor_noise_sd",
            "poor_gamma",
        ],
        columns="estimator",
        values=["mean_rmse", "mean_coverage", "mean_ci_length"],
    )
    pivot.columns = [f"{metric}_{est}" for metric, est in pivot.columns]
    pivot = pivot.reset_index()

    diag_summary = (
        diagnostics[diagnostics["replicate"] >= 0]
        .groupby(
            [
                "score_quality",
                "score_setting_id",
                "score_family",
                "mu_slope",
                "well_noise_sd",
                "poor_a",
                "poor_b",
                "poor_shift",
                "poor_scale",
                "poor_noise_sd",
                "poor_gamma",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            raw_calibration_mse=("raw_calibration_mse", "mean"),
            affine_calibration_mse=("affine_calibration_mse", "mean"),
            platt_calibration_mse=("platt_calibration_mse", "mean"),
            isotonic_calibration_mse=("isotonic_calibration_mse", "mean"),
            nonlinear_calibration_mse=("nonlinear_calibration_mse", "mean"),
            raw_residual_var=("raw_residual_var", "mean"),
            affine_residual_var=("affine_residual_var", "mean"),
            platt_residual_var=("platt_residual_var", "mean"),
            isotonic_residual_var=("isotonic_residual_var", "mean"),
            nonlinear_residual_var=("nonlinear_residual_var", "mean"),
        )
    )
    pilot = pivot.merge(
        diag_summary,
        on=[
            "score_quality",
            "score_setting_id",
            "score_family",
            "mu_slope",
            "well_noise_sd",
            "poor_a",
            "poor_b",
            "poor_shift",
            "poor_scale",
            "poor_noise_sd",
            "poor_gamma",
        ],
        how="left",
    )
    pilot["best_calibrated_rmse"] = pilot[
        [
            "mean_rmse_affine_calibration",
            "mean_rmse_platt_calibration",
            "mean_rmse_isotonic_calibration",
            "mean_rmse_calibrated_plugin",
        ]
    ].min(axis=1)
    pilot["gain_best_calibrated_vs_ppi"] = pilot["mean_rmse_ppi"] - pilot["best_calibrated_rmse"]
    pilot["gain_best_calibrated_vs_label"] = pilot["mean_rmse_labeled_only"] - pilot["best_calibrated_rmse"]
    pilot["rmse_spread_score_methods"] = pilot[
        [
            "mean_rmse_ppi",
            "mean_rmse_affine_calibration",
            "mean_rmse_platt_calibration",
            "mean_rmse_isotonic_calibration",
            "mean_rmse_calibrated_plugin",
        ]
    ].max(axis=1) - pilot[
        [
            "mean_rmse_ppi",
            "mean_rmse_affine_calibration",
            "mean_rmse_platt_calibration",
            "mean_rmse_isotonic_calibration",
            "mean_rmse_calibrated_plugin",
        ]
    ].min(axis=1)
    pilot["min_coverage_score_methods"] = pilot[
        [
            "mean_coverage_ppi",
            "mean_coverage_affine_calibration",
            "mean_coverage_platt_calibration",
            "mean_coverage_isotonic_calibration",
            "mean_coverage_calibrated_plugin",
        ]
    ].min(axis=1)
    return pilot.sort_values(["score_quality", "score_setting_id"]).reset_index(drop=True)


def select_pilot_recommendations(pilot_summary: pd.DataFrame) -> Dict[str, object]:
    def clean_record(record: Dict[str, object]) -> Dict[str, object]:
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned

    well = pilot_summary[pilot_summary["score_quality"] == "well_calibrated"].copy()
    poor = pilot_summary[pilot_summary["score_quality"] == "poorly_calibrated"].copy()

    well["selection_score"] = (
        well["rmse_spread_score_methods"]
        + 2.0 * np.maximum(0.9 - well["min_coverage_score_methods"], 0.0)
        + 0.05 * well["raw_calibration_mse"]
    )
    poor["selection_score"] = -(
        poor["gain_best_calibrated_vs_ppi"] + poor["gain_best_calibrated_vs_label"]
    ) + 2.0 * np.maximum(0.85 - poor["min_coverage_score_methods"], 0.0)

    rec_well = well.sort_values("selection_score").iloc[0]
    rec_poor = poor.sort_values("selection_score").iloc[0]
    return {
        "well_calibrated": clean_record(
            rec_well[
                [
                    "score_setting_id",
                    "score_family",
                    "mu_slope",
                    "well_noise_sd",
                    "poor_a",
                    "poor_b",
                    "poor_shift",
                    "poor_scale",
                    "poor_noise_sd",
                    "poor_gamma",
                    "selection_score",
                ]
            ].to_dict()
        ),
        "poorly_calibrated": clean_record(
            rec_poor[
                [
                    "score_setting_id",
                    "score_family",
                    "mu_slope",
                    "well_noise_sd",
                    "poor_a",
                    "poor_b",
                    "poor_shift",
                    "poor_scale",
                    "poor_noise_sd",
                    "poor_gamma",
                    "selection_score",
                ]
            ].to_dict()
        ),
    }


def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary[
        summary["estimator"].isin(
            [
                "labeled_only",
                "ppi",
                "ppi_plus_plus",
                "aipw_em",
                "aipw",
                "affine_calibration",
                "isotonic_calibration",
            ]
        )
    ].copy()
    selected = [
        ("monotone", "well_calibrated", 100, 100),
        ("monotone", "poorly_calibrated", 100, 100),
        ("monotone", "well_calibrated", 400, 6400),
        ("monotone", "poorly_calibrated", 400, 6400),
    ]
    chunks = []
    for dgp, quality, n, N in selected:
        frame = summary[
            (summary["dgp"] == dgp)
            & (summary["score_quality"] == quality)
            & (summary["n_labeled"] == n)
            & (summary["n_unlabeled"] == N)
        ].copy()
        label = f"{dgp}_{quality}_n{n}_N{N}"
        frame[label] = (
            "mean_bias="
            + frame["mean_bias"].round(3).astype(str)
            + "; emp_sd="
            + frame["emp_sd"].round(3).astype(str)
            + "; mean_ci_length="
            + frame["mean_ci_length"].round(3).astype(str)
            + "; rmse="
            + frame["rmse"].round(3).astype(str)
            + "; coverage="
            + frame["coverage"].round(3).astype(str)
        )
        chunks.append(frame[["estimator", label]])
    table = chunks[0]
    for chunk in chunks[1:]:
        table = table.merge(chunk, on="estimator", how="outer")
    return table.sort_values("estimator").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES.keys()), default="quick")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulations"))
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--recommendations-file", type=Path, default=None)
    parser.add_argument(
        "--estimators",
        nargs="*",
        default=None,
        help="Optional estimator subset to run.",
    )
    parser.add_argument(
        "--main-text-only",
        action="store_true",
        help="Run only the estimators used in the main-text simulation figure.",
    )
    args = parser.parse_args()

    if args.main_text_only and args.estimators is not None:
        parser.error("Use either --main-text-only or --estimators, not both.")

    profile = PROFILES[args.profile]
    if args.replications is not None:
        profile = replace(profile, replications=args.replications)

    if args.main_text_only:
        estimators = MAIN_TEXT_ESTIMATORS
    elif args.estimators is None:
        estimators = ALL_ESTIMATORS
    else:
        unknown = sorted(set(args.estimators) - set(ALL_ESTIMATORS))
        if unknown:
            parser.error(f"Unknown estimators: {', '.join(unknown)}")
        estimators = tuple(args.estimators)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot_pop_size = None
    pilot_replications = None
    if_summary = None
    mc_pilot_summary = None
    if args.recommendations_file is not None:
        with open(args.recommendations_file, "r", encoding="utf-8") as fh:
            recommendations = json.load(fh)
    else:
        pilot_pop_size = 250_000 if profile.name == "quick" else 1_000_000
        if_summary = build_if_pilot_summary(seed=args.seed, pop_size=pilot_pop_size)
        well_candidates, poor_candidates = shortlist_candidates_from_if(if_summary)
        if profile.name == "quick":
            poor_candidates = poor_candidates[:6]
        pilot_replications = 40 if profile.name == "quick" else 180
        mc_pilot_summary = build_mc_pilot_summary(
            seed=args.seed,
            well_candidates=well_candidates,
            poor_candidates=poor_candidates,
            replications=pilot_replications,
        )
        recommendations = select_final_recommendations(if_summary, mc_pilot_summary)
        if_summary.to_csv(args.output_dir / "pilot_if_summary.csv", index=False)
        mc_pilot_summary.to_csv(args.output_dir / "pilot_mc_summary.csv", index=False)
        with open(args.output_dir / "pilot_recommendations.json", "w", encoding="utf-8") as fh:
            json.dump(recommendations, fh, indent=2)

    raw_rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    global_start = time.time()
    score_settings = selected_score_settings(recommendations)

    regime_id = 0
    for dgp in profile.dgp_grid:
        for setting in score_settings:
            for n in profile.n_grid:
                for ratio in profile.unlabeled_ratio_grid:
                    N = int(n * ratio)
                    regime_start = time.time()
                    for rep in range(profile.replications):
                        rng_seed = args.seed + 10_000 * regime_id + rep
                        rng = np.random.default_rng(rng_seed)
                        sample = generate_sample(n=n, N=N, dgp=dgp, setting=setting, rng=rng)
                        results, diagnostics = run_one(
                            sample,
                            profile,
                            rng_seed=rng_seed,
                            estimators=estimators,
                        )
                        for result in results:
                            raw_rows.append(
                                {
                                    "dgp": dgp,
                                    "score_quality": setting.quality,
                                    "score_setting_id": setting.setting_id,
                                    "score_family": setting.family,
                                    "mu_slope": setting.mu_slope,
                                    "well_noise_sd": setting.well_noise_sd,
                                    "poor_a": setting.poor_a,
                                    "poor_b": setting.poor_b,
                                    "poor_shift": setting.poor_shift,
                                    "poor_scale": setting.poor_scale,
                                    "poor_noise_sd": setting.poor_noise_sd,
                                    "poor_gamma": setting.poor_gamma,
                                    "n_labeled": n,
                                    "n_unlabeled": N,
                                    "unlabeled_ratio": ratio,
                                    "replicate": rep,
                                    **result,
                                }
                            )
                        diag_rows.append(
                            {
                                "dgp": dgp,
                                "score_quality": setting.quality,
                                "score_setting_id": setting.setting_id,
                                "score_family": setting.family,
                                "mu_slope": setting.mu_slope,
                                "well_noise_sd": setting.well_noise_sd,
                                "poor_a": setting.poor_a,
                                "poor_b": setting.poor_b,
                                "poor_shift": setting.poor_shift,
                                "poor_scale": setting.poor_scale,
                                "poor_noise_sd": setting.poor_noise_sd,
                                "poor_gamma": setting.poor_gamma,
                                "n_labeled": n,
                                "n_unlabeled": N,
                                "unlabeled_ratio": ratio,
                                "replicate": rep,
                                **diagnostics,
                            }
                        )
                    diag_rows.append(
                        {
                            "dgp": dgp,
                            "score_quality": setting.quality,
                            "score_setting_id": setting.setting_id,
                            "score_family": setting.family,
                            "mu_slope": setting.mu_slope,
                            "well_noise_sd": setting.well_noise_sd,
                            "poor_a": setting.poor_a,
                            "poor_b": setting.poor_b,
                            "poor_shift": setting.poor_shift,
                            "poor_scale": setting.poor_scale,
                            "poor_noise_sd": setting.poor_noise_sd,
                            "poor_gamma": setting.poor_gamma,
                            "n_labeled": n,
                            "n_unlabeled": N,
                            "unlabeled_ratio": ratio,
                            "replicate": -1,
                            "runtime_seconds": time.time() - regime_start,
                        }
                    )
                    regime_id += 1

    raw_df = pd.DataFrame(raw_rows)
    diag_df = pd.DataFrame(diag_rows)
    summary = add_efficiency_metrics(summarise(raw_df))

    raw_df.to_csv(args.output_dir / "raw_results.csv", index=False)
    diag_df.to_csv(args.output_dir / "diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if profile.name == "pilot":
        pilot_summary = build_pilot_summary(summary, diag_df)
        pilot_summary.to_csv(args.output_dir / "pilot_summary.csv", index=False)
    else:
        table_main = build_main_table(summary)
        table_main.to_csv(args.output_dir / "table_main.csv", index=False)

    summary_json = {
        "profile": profile.name,
        "replications": profile.replications,
        "estimators": list(estimators),
        "pilot_replications": pilot_replications,
        "pilot_population_size": pilot_pop_size,
        "recommendations_file": str(args.recommendations_file) if args.recommendations_file is not None else None,
        "total_runtime_seconds": time.time() - global_start,
        "venn_abers_reference": "ppi_estimator",
        "selected_well_score_setting": recommendations["well_calibrated"]["score_setting_id"],
        "selected_poor_score_setting": recommendations["poorly_calibrated"]["score_setting_id"],
        "selected_well_avg_gain_pct": recommendations["well_calibrated"].get("avg_gain_pct"),
        "selected_poor_avg_gain_pct": recommendations["poorly_calibrated"].get("avg_gain_pct"),
        "selected_well_mc_gain_vs_ppi_pct": recommendations["well_calibrated"].get("mc_avg_gain_vs_ppi_pct"),
        "selected_poor_mc_gain_vs_ppi_pct": recommendations["poorly_calibrated"].get("mc_avg_gain_vs_ppi_pct"),
    }
    with open(args.output_dir / "diagnostic_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)


if __name__ == "__main__":
    main()
