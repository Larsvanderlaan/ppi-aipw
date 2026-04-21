from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from ppi_py.datasets import load_dataset
from sklearn.linear_model import LinearRegression, LogisticRegression

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.llm_eval_models import load_jsonl_frame, row_cache_path, score_cache_path


INTRO_PREFERRED_EVALUATOR = "armorm_llama3_8b_v0_1"
INTRO_PATHOLOGICAL_EVALUATOR = "skywork_reward_llama3_1_8b"
PPE_EVALUATORS = (
    "armorm_llama3_8b_v0_1",
    "athene_rm_8b",
    "skywork_reward_llama3_1_8b",
)
CLASSIC_DATASETS = ("census_income", "forest", "galaxies")

DISPLAY_NAMES = {
    "armorm_llama3_8b_v0_1": "ArmoRM (PPE correctness)",
    "athene_rm_8b": "Athene (PPE correctness)",
    "skywork_reward_llama3_1_8b": "Skywork (PPE correctness)",
    "census_income": "Census income",
    "forest": "Forest cover",
    "galaxies": "Galaxies",
}


@dataclass(frozen=True)
class ScreeningConfig:
    track: str = "ppe_correctness"
    evaluators: tuple[str, ...] = PPE_EVALUATORS
    classic_datasets: tuple[str, ...] = CLASSIC_DATASETS
    arm_quantiles: tuple[float, float] = (0.1, 0.9)
    alpha: float = 0.10
    n_grid: tuple[int, ...] = (25, 50)
    n_unlabeled_per_arm: int = 400
    reference_fit_n_per_arm: int = 40
    replications: int = 200
    seed: int = 20260420
    llm_cache_dir: Path = field(default_factory=lambda: Path("outputs/cache/llm_eval"))
    dataset_cache_dir: Path = field(default_factory=lambda: Path("outputs/cache/ppi_datasets"))


@dataclass(frozen=True)
class CandidateData:
    candidate_id: str
    display_name: str
    family: str
    score_label: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class ArmData:
    candidate_id: str
    display_name: str
    family: str
    score_label: str
    arm_quantiles: tuple[float, float]
    score_low: float
    score_high: float
    frame: pd.DataFrame
    arm_a: pd.DataFrame
    arm_b: pd.DataFrame


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


def labeled_only_ci_length(y_a: np.ndarray, y_b: np.ndarray, *, alpha: float) -> tuple[float, float]:
    estimate = float(np.mean(y_a) - np.mean(y_b))
    se = float(np.sqrt(np.var(y_a, ddof=1) / len(y_a) + np.var(y_b, ddof=1) / len(y_b)))
    return estimate, 2.0 * _z_value(alpha) * se


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


def load_classic_candidate(dataset_cache_dir: Path, dataset_name: str) -> CandidateData:
    bundle = load_dataset(str(dataset_cache_dir), dataset_name, download=False)
    y = np.asarray(bundle["Y"], dtype=float)
    score = np.asarray(bundle["Yhat"], dtype=float)
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(len(y), dtype=int),
            "y": y,
            "score": score,
        }
    )
    return CandidateData(
        candidate_id=dataset_name,
        display_name=DISPLAY_NAMES[dataset_name],
        family="classic",
        score_label="Raw score",
        frame=frame,
    )


def build_arm_data(candidate: CandidateData, arm_quantiles: tuple[float, float]) -> ArmData:
    score = np.asarray(candidate.frame["score"], dtype=float)
    low_q, high_q = arm_quantiles
    score_low = float(np.quantile(score, low_q))
    score_high = float(np.quantile(score, high_q))
    frame = candidate.frame.copy()
    frame["arm"] = "middle"
    frame.loc[frame["score"] <= score_low, "arm"] = "B"
    frame.loc[frame["score"] >= score_high, "arm"] = "A"
    arm_a = frame[frame["arm"] == "A"].copy().reset_index(drop=True)
    arm_b = frame[frame["arm"] == "B"].copy().reset_index(drop=True)
    return ArmData(
        candidate_id=candidate.candidate_id,
        display_name=candidate.display_name,
        family=candidate.family,
        score_label=candidate.score_label,
        arm_quantiles=arm_quantiles,
        score_low=score_low,
        score_high=score_high,
        frame=frame.reset_index(drop=True),
        arm_a=arm_a,
        arm_b=arm_b,
    )


def _reference_split_indices(
    arm_data: ArmData,
    *,
    fit_n_per_arm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_a = np.arange(len(arm_data.arm_a))
    idx_b = np.arange(len(arm_data.arm_b))
    fit_a = rng.choice(idx_a, size=min(fit_n_per_arm, len(idx_a) // 2), replace=False)
    fit_b = rng.choice(idx_b, size=min(fit_n_per_arm, len(idx_b) // 2), replace=False)
    hold_a = np.setdiff1d(idx_a, fit_a)
    hold_b = np.setdiff1d(idx_b, fit_b)
    return fit_a, fit_b, hold_a, hold_b


def build_reference_metrics(
    arm_data: ArmData,
    *,
    calibrator_name: str,
    fit_n_per_arm: int,
    seed: int,
) -> dict[str, float]:
    fit_a, fit_b, hold_a, hold_b = _reference_split_indices(
        arm_data,
        fit_n_per_arm=fit_n_per_arm,
        seed=seed,
    )
    fit_frame = pd.concat([arm_data.arm_a.iloc[fit_a], arm_data.arm_b.iloc[fit_b]], ignore_index=True)
    hold_frame = pd.concat([arm_data.arm_a.iloc[hold_a], arm_data.arm_b.iloc[hold_b]], ignore_index=True)

    calibrator = fit_one_dimensional_calibrator(
        fit_frame["score"].to_numpy(dtype=float),
        fit_frame["y"].to_numpy(dtype=float),
        method=calibrator_name,
    )
    calibrated_holdout = predict_with_calibrator(calibrator, hold_frame["score"].to_numpy(dtype=float))
    y_holdout = hold_frame["y"].to_numpy(dtype=float)
    raw_holdout = hold_frame["score"].to_numpy(dtype=float)
    raw_mse = float(np.mean((y_holdout - raw_holdout) ** 2))
    calibrated_mse = float(np.mean((y_holdout - calibrated_holdout) ** 2))
    return {
        "raw_mse": raw_mse,
        "calibrated_mse": calibrated_mse,
        "mse_reduction": float(1.0 - calibrated_mse / raw_mse),
    }


def run_candidate_monte_carlo(
    arm_data: ArmData,
    *,
    calibrator_name: str,
    n_grid: Iterable[int],
    n_unlabeled_per_arm: int,
    replications: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arm_a_score = arm_data.arm_a["score"].to_numpy(dtype=float)
    arm_b_score = arm_data.arm_b["score"].to_numpy(dtype=float)
    arm_a_y = arm_data.arm_a["y"].to_numpy(dtype=float)
    arm_b_y = arm_data.arm_b["y"].to_numpy(dtype=float)
    true_delta = float(np.mean(arm_a_y) - np.mean(arm_b_y))
    rows: list[dict[str, float]] = []

    for n_per_arm in n_grid:
        for replication in range(replications):
            idx_a = np.arange(len(arm_a_y))
            idx_b = np.arange(len(arm_b_y))
            labeled_a = rng.choice(idx_a, size=n_per_arm, replace=False)
            labeled_b = rng.choice(idx_b, size=n_per_arm, replace=False)
            remaining_a = np.setdiff1d(idx_a, labeled_a)
            remaining_b = np.setdiff1d(idx_b, labeled_b)
            unlabeled_size = min(n_unlabeled_per_arm, len(remaining_a), len(remaining_b))
            unlabeled_a = rng.choice(remaining_a, size=unlabeled_size, replace=False)
            unlabeled_b = rng.choice(remaining_b, size=unlabeled_size, replace=False)

            estimate_labeled, ci_labeled = labeled_only_ci_length(
                arm_a_y[labeled_a],
                arm_b_y[labeled_b],
                alpha=alpha,
            )
            labeled_se = ci_labeled / (2.0 * _z_value(alpha))
            rows.append(
                {
                    "n_per_arm": n_per_arm,
                    "replication": replication,
                    "estimator": "labeled_only",
                    "estimate": estimate_labeled,
                    "mean_ci_length": ci_labeled,
                    "covered": float(
                        estimate_labeled - _z_value(alpha) * labeled_se
                        <= true_delta
                        <= estimate_labeled + _z_value(alpha) * labeled_se
                    ),
                }
            )

            estimate_a_raw, se_a_raw = official_ppi_summary(
                arm_a_y[labeled_a],
                arm_a_score[labeled_a],
                arm_a_score[unlabeled_a],
                alpha=alpha,
            )
            estimate_b_raw, se_b_raw = official_ppi_summary(
                arm_b_y[labeled_b],
                arm_b_score[labeled_b],
                arm_b_score[unlabeled_b],
                alpha=alpha,
            )
            raw_delta = estimate_a_raw - estimate_b_raw
            raw_se = float(np.sqrt(se_a_raw**2 + se_b_raw**2))
            rows.append(
                {
                    "n_per_arm": n_per_arm,
                    "replication": replication,
                    "estimator": "raw_ppi",
                    "estimate": raw_delta,
                    "mean_ci_length": 2.0 * _z_value(alpha) * raw_se,
                    "covered": float(
                        raw_delta - _z_value(alpha) * raw_se
                        <= true_delta
                        <= raw_delta + _z_value(alpha) * raw_se
                    ),
                }
            )

            calibrator = fit_one_dimensional_calibrator(
                np.concatenate([arm_a_score[labeled_a], arm_b_score[labeled_b]]),
                np.concatenate([arm_a_y[labeled_a], arm_b_y[labeled_b]]),
                method=calibrator_name,
            )
            calibrated_a_l = predict_with_calibrator(calibrator, arm_a_score[labeled_a])
            calibrated_a_u = predict_with_calibrator(calibrator, arm_a_score[unlabeled_a])
            calibrated_b_l = predict_with_calibrator(calibrator, arm_b_score[labeled_b])
            calibrated_b_u = predict_with_calibrator(calibrator, arm_b_score[unlabeled_b])
            estimate_a_cal, se_a_cal = official_ppi_summary(
                arm_a_y[labeled_a],
                calibrated_a_l,
                calibrated_a_u,
                alpha=alpha,
            )
            estimate_b_cal, se_b_cal = official_ppi_summary(
                arm_b_y[labeled_b],
                calibrated_b_l,
                calibrated_b_u,
                alpha=alpha,
            )
            cal_delta = estimate_a_cal - estimate_b_cal
            cal_se = float(np.sqrt(se_a_cal**2 + se_b_cal**2))
            rows.append(
                {
                    "n_per_arm": n_per_arm,
                    "replication": replication,
                    "estimator": "calibrated_ppi",
                    "estimate": cal_delta,
                    "mean_ci_length": 2.0 * _z_value(alpha) * cal_se,
                    "covered": float(
                        cal_delta - _z_value(alpha) * cal_se
                        <= true_delta
                        <= cal_delta + _z_value(alpha) * cal_se
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["n_per_arm", "estimator"], as_index=False)
        .agg(
            mean_ci_length=("mean_ci_length", "mean"),
            coverage=("covered", "mean"),
        )
        .sort_values(["n_per_arm", "estimator"])
        .reset_index(drop=True)
    )


def _ratio_safe(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return float("inf")
    return float(numerator / denominator)


def believable_intro_score(row: pd.Series, *, alpha: float) -> float:
    ci_gain = max(float(row["ci_gain_n25"]), float(row["ci_gain_n50"]))
    mse_bonus = np.clip(float(row["mse_reduction"]) / 0.50, 0.0, 1.0)
    ci_bonus = np.clip(ci_gain / 0.15, 0.0, 1.0)
    compression_bonus = np.clip(
        (float(row["label_gap"]) - abs(float(row["raw_gap"])))
        / max(float(row["label_gap"]), abs(float(row["raw_gap"])), 1e-8),
        0.0,
        1.0,
    )
    nominal_coverage = 1.0 - alpha
    coverage_bonus = np.clip(1.0 - abs(float(row["calibrated_coverage_n50"]) - nominal_coverage) / 0.05, 0.0, 1.0)
    pathology_penalty = 0.20 * max(0.0, float(row["raw_ci_ratio_n25"]) - 2.0)
    pathology_penalty += 0.20 * max(0.0, float(row["raw_ci_ratio_n50"]) - 2.0)
    return float(0.35 * ci_bonus + 0.25 * mse_bonus + 0.25 * compression_bonus + 0.15 * coverage_bonus - pathology_penalty)


def screening_note(row: pd.Series) -> str:
    if bool(row["selected_intro"]):
        return "Chosen intro anchor"
    if row["candidate_id"] == INTRO_PREFERRED_EVALUATOR:
        return "Benign real failure, but too mild for the main toy"
    if row["candidate_id"] == INTRO_PATHOLOGICAL_EVALUATOR:
        return "Very large gain, but driven by pathological raw scaling"
    if row["family"] == "classic":
        return "Real benchmark, but smaller incremental gain"
    return "Strong appendix robustness case"


def evaluate_candidate(
    candidate: CandidateData,
    config: ScreeningConfig,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    arm_data = build_arm_data(candidate, config.arm_quantiles)
    calibrator_name = calibrator_for_candidate(candidate)
    reference = build_reference_metrics(
        arm_data,
        calibrator_name=calibrator_name,
        fit_n_per_arm=config.reference_fit_n_per_arm,
        seed=config.seed + seed_offset,
    )
    monte_carlo = run_candidate_monte_carlo(
        arm_data,
        calibrator_name=calibrator_name,
        n_grid=config.n_grid,
        n_unlabeled_per_arm=config.n_unlabeled_per_arm,
        replications=config.replications,
        alpha=config.alpha,
        seed=config.seed + 10_000 + seed_offset,
    )
    pivot_width = monte_carlo.pivot(index="n_per_arm", columns="estimator", values="mean_ci_length")
    pivot_coverage = monte_carlo.pivot(index="n_per_arm", columns="estimator", values="coverage")

    result = {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "family": candidate.family,
        "score_label": candidate.score_label,
        "calibrator": calibrator_name,
        "arm_a_size": int(len(arm_data.arm_a)),
        "arm_b_size": int(len(arm_data.arm_b)),
        "arm_low_quantile": float(config.arm_quantiles[0]),
        "arm_high_quantile": float(config.arm_quantiles[1]),
        "score_mean_a": float(arm_data.arm_a["score"].mean()),
        "score_mean_b": float(arm_data.arm_b["score"].mean()),
        "y_mean_a": float(arm_data.arm_a["y"].mean()),
        "y_mean_b": float(arm_data.arm_b["y"].mean()),
        "label_gap": float(arm_data.arm_a["y"].mean() - arm_data.arm_b["y"].mean()),
        "raw_gap": float(arm_data.arm_a["score"].mean() - arm_data.arm_b["score"].mean()),
        "score_low_cutoff": float(arm_data.score_low),
        "score_high_cutoff": float(arm_data.score_high),
        **reference,
    }

    for n in config.n_grid:
        result[f"ci_gain_n{n}"] = float(
            1.0 - pivot_width.loc[n, "calibrated_ppi"] / pivot_width.loc[n, "raw_ppi"]
        )
        result[f"raw_ci_ratio_n{n}"] = _ratio_safe(
            float(pivot_width.loc[n, "raw_ppi"]),
            float(pivot_width.loc[n, "labeled_only"]),
        )
        result[f"calibrated_coverage_n{n}"] = float(pivot_coverage.loc[n, "calibrated_ppi"])
        result[f"raw_coverage_n{n}"] = float(pivot_coverage.loc[n, "raw_ppi"])
        result[f"labeled_coverage_n{n}"] = float(pivot_coverage.loc[n, "labeled_only"])

    result["ordering_correct"] = bool(result["score_mean_a"] > result["score_mean_b"] and result["y_mean_a"] > result["y_mean_b"])
    result["passes_error_threshold"] = bool(result["mse_reduction"] >= 0.25)
    result["passes_width_threshold"] = bool(
        max(result[f"ci_gain_n{n}"] for n in config.n_grid) >= 0.10
    )
    target_coverage = 1.0 - config.alpha
    average_calibrated_coverage = float(
        np.mean([result[f"calibrated_coverage_n{n}"] for n in config.n_grid])
    )
    result["average_calibrated_coverage"] = average_calibrated_coverage
    result["passes_coverage_threshold"] = bool(
        abs(average_calibrated_coverage - target_coverage) <= 0.03
    )
    result["pathological_raw_ppi"] = bool(
        max(result[f"raw_ci_ratio_n{n}"] for n in config.n_grid) > 2.0
    )
    result["passes_intro_thresholds"] = bool(
        result["ordering_correct"]
        and result["passes_error_threshold"]
        and result["passes_width_threshold"]
        and result["passes_coverage_threshold"]
        and not result["pathological_raw_ppi"]
    )
    return result


def screen_candidates(config: ScreeningConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_offset = 0

    for evaluator in config.evaluators:
        candidate = load_ppe_candidate(config.llm_cache_dir, config.track, evaluator)
        rows.append(evaluate_candidate(candidate, config, seed_offset=seed_offset))
        seed_offset += 1_000

    for dataset_name in config.classic_datasets:
        candidate = load_classic_candidate(config.dataset_cache_dir, dataset_name)
        rows.append(evaluate_candidate(candidate, config, seed_offset=seed_offset))
        seed_offset += 1_000

    frame = pd.DataFrame(rows)
    frame["believable_intro_score"] = frame.apply(
        lambda row: believable_intro_score(row, alpha=config.alpha),
        axis=1,
    )
    frame["selected_intro"] = False
    passing = frame[frame["passes_intro_thresholds"]].copy()
    if INTRO_PREFERRED_EVALUATOR in set(passing["candidate_id"]):
        selected_id = INTRO_PREFERRED_EVALUATOR
    elif not passing.empty:
        selected_id = passing.sort_values("believable_intro_score", ascending=False).iloc[0]["candidate_id"]
    else:
        selected_id = frame.sort_values("believable_intro_score", ascending=False).iloc[0]["candidate_id"]
    frame.loc[frame["candidate_id"] == selected_id, "selected_intro"] = True
    frame["screening_note"] = frame.apply(screening_note, axis=1)
    return frame.sort_values(
        ["selected_intro", "believable_intro_score", "family", "display_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def select_intro_candidate(screening: pd.DataFrame) -> pd.Series:
    selected = screening[screening["selected_intro"]]
    if selected.empty:
        raise ValueError("Screening summary does not identify a selected intro candidate.")
    return selected.iloc[0]


def write_screening_outputs(
    screening: pd.DataFrame,
    config: ScreeningConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output_dir / "toy_anchor_screening.csv", index=False)
    payload = {
        "config": {
            **asdict(config),
            "llm_cache_dir": str(config.llm_cache_dir),
            "dataset_cache_dir": str(config.dataset_cache_dir),
        },
        "records": screening.to_dict(orient="records"),
    }
    (output_dir / "toy_anchor_screening.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    latex_rows = [
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Candidate & $\mu_A$ & $\mu_B$ & score$_A$ & score$_B$ & MSE drop & Note \\",
        r"\midrule",
    ]
    for _, row in screening.iterrows():
        prefix = r"\textbf{" if bool(row["selected_intro"]) else ""
        suffix = "}" if bool(row["selected_intro"]) else ""
        latex_rows.append(
            (
                f"{prefix}{row['display_name']}{suffix} & "
                f"{row['y_mean_a']:.3f} & "
                f"{row['y_mean_b']:.3f} & "
                f"{row['score_mean_a']:.3f} & "
                f"{row['score_mean_b']:.3f} & "
                f"{100.0 * row['mse_reduction']:.0f}\\% & "
                f"{row['screening_note']} \\\\"
            )
        )
    latex_rows.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table_toy_anchor_screening.tex").write_text(
        "\n".join(latex_rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen real candidates for the grounded calibration toy.")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulations/toy"))
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("outputs/cache/llm_eval"))
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path("outputs/cache/ppi_datasets"))
    args = parser.parse_args()

    config = ScreeningConfig(
        seed=args.seed,
        replications=args.replications,
        llm_cache_dir=args.llm_cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    screening = screen_candidates(config)
    write_screening_outputs(screening, config, args.output_dir)


if __name__ == "__main__":
    main()
