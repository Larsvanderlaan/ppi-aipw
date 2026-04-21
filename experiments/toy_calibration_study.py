from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.grounded_toy_screening import (
    INTRO_PREFERRED_EVALUATOR,
    _z_value,
    calibrator_for_candidate,
    fit_one_dimensional_calibrator,
    load_ppe_candidate,
    official_ppi_summary,
    predict_with_calibrator,
)


COLORS = {
    "labeled_only": "#4D4D4D",
    "raw_ppi": "#D55E00",
    "calibrated_ppi": "#009E73",
    "bins": "#7A7A7A",
}

LABELS = {
    "labeled_only": "Labeled only",
    "raw_ppi": "Raw PPI",
    "calibrated_ppi": "Calibrated PPI",
}

ESTIMATOR_ORDER = ("labeled_only", "raw_ppi", "calibrated_ppi")


@dataclass(frozen=True)
class ToyStudyConfig:
    track: str = "ppe_correctness"
    evaluator: str = INTRO_PREFERRED_EVALUATOR
    benchmark: str = "math_best_of_k"
    calibrator: str = "platt"
    n_grid: tuple[int, ...] = (25, 50, 100, 200)
    n_unlabeled: int = 2000
    reference_fit_n: int = 100
    alpha: float = 0.10
    binned_curve_bins: int = 12
    llm_cache_dir: Path = field(default_factory=lambda: Path("outputs/cache/llm_eval"))
    dataset_cache_dir: Path = field(default_factory=lambda: Path("outputs/cache/ppi_datasets"))


def _subset_label(benchmark: str) -> str:
    if benchmark == "math_best_of_k":
        return "MATH correctness"
    return benchmark.replace("_best_of_k", "").replace("_", " ").upper()


def _load_toy_subset(config: ToyStudyConfig) -> tuple[Any, pd.DataFrame]:
    candidate = load_ppe_candidate(config.llm_cache_dir, config.track, config.evaluator)
    frame = candidate.frame[candidate.frame["benchmark"] == config.benchmark].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"No rows found for evaluator '{config.evaluator}' on benchmark '{config.benchmark}'."
        )
    return candidate, frame


def _calibrator_name_for_toy(config: ToyStudyConfig, candidate: Any) -> str:
    if config.calibrator == "linear":
        return "linear"
    if config.calibrator == "platt":
        return calibrator_for_candidate(candidate)
    raise ValueError(f"Unknown calibrator '{config.calibrator}'.")


def _bin_empirical_curve(frame: pd.DataFrame, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("score").reset_index(drop=True)
    scores = ordered["score"].to_numpy(dtype=float)
    outcomes = ordered["y"].to_numpy(dtype=float)
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    centers: list[float] = []
    means: list[float] = []
    counts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isclose(left, right):
            mask = np.isclose(scores, left)
        else:
            mask = (scores >= left) & (scores <= right)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(scores[mask])))
        means.append(float(np.mean(outcomes[mask])))
        counts.append(int(np.sum(mask)))
    return np.asarray(centers), np.asarray(means), np.asarray(counts)


def _labeled_only_mean_summary(y_labeled: np.ndarray, *, alpha: float) -> tuple[float, float]:
    estimate = float(np.mean(y_labeled))
    se = float(np.sqrt(np.var(y_labeled, ddof=1) / len(y_labeled)))
    return estimate, 2.0 * _z_value(alpha) * se


def build_reference_summary(
    config: ToyStudyConfig,
    *,
    screening: pd.DataFrame | None = None,
) -> dict[str, Any]:
    del screening
    candidate, frame = _load_toy_subset(config)
    calibrator_name = _calibrator_name_for_toy(config, candidate)
    fit_n = min(config.reference_fit_n, len(frame) // 2)
    if fit_n < 20:
        raise ValueError("The toy subset is too small to build a stable calibration reference.")

    fit_indices = frame.sample(n=fit_n, random_state=20260420).index
    fit_frame = frame.loc[fit_indices].reset_index(drop=True)
    holdout_frame = frame.drop(index=fit_indices).reset_index(drop=True)
    calibrator = fit_one_dimensional_calibrator(
        fit_frame["score"].to_numpy(dtype=float),
        fit_frame["y"].to_numpy(dtype=float),
        method=calibrator_name,
    )

    x_grid = np.linspace(float(frame["score"].min()), float(frame["score"].max()), 300)
    fitted_curve = predict_with_calibrator(calibrator, x_grid)
    bin_centers, bin_means, bin_counts = _bin_empirical_curve(frame, config.binned_curve_bins)

    raw_holdout = holdout_frame["score"].to_numpy(dtype=float)
    y_holdout = holdout_frame["y"].to_numpy(dtype=float)
    calibrated_holdout = predict_with_calibrator(calibrator, raw_holdout)
    raw_mse = float(np.mean((y_holdout - raw_holdout) ** 2))
    calibrated_mse = float(np.mean((y_holdout - calibrated_holdout) ** 2))

    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "benchmark": config.benchmark,
        "benchmark_label": _subset_label(config.benchmark),
        "score_label": "Raw evaluator margin",
        "calibrator": calibrator_name,
        "subset_size": int(len(frame)),
        "reference_fit_n": int(fit_n),
        "mean_y": float(frame["y"].mean()),
        "score_mean": float(frame["score"].mean()),
        "score_sd": float(frame["score"].std(ddof=1)),
        "raw_mse": raw_mse,
        "calibrated_mse": calibrated_mse,
        "mse_reduction": float(1.0 - calibrated_mse / raw_mse),
        "bin_centers": bin_centers.tolist(),
        "bin_means": bin_means.tolist(),
        "bin_counts": bin_counts.tolist(),
        "x_grid": x_grid.tolist(),
        "fitted_curve": fitted_curve.tolist(),
    }


def run_monte_carlo(
    config: ToyStudyConfig,
    *,
    replications: int,
    seed: int,
    screening: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del screening
    candidate, frame = _load_toy_subset(config)
    calibrator_name = _calibrator_name_for_toy(config, candidate)

    rng = np.random.default_rng(seed)
    y = frame["y"].to_numpy(dtype=float)
    score = frame["score"].to_numpy(dtype=float)
    true_mean = float(np.mean(y))
    rows: list[dict[str, float]] = []

    for n_labeled in config.n_grid:
        unlabeled_size = min(config.n_unlabeled, len(y) - n_labeled)
        if unlabeled_size <= 0:
            continue
        for replication in range(replications):
            indices = np.arange(len(y))
            labeled_idx = rng.choice(indices, size=n_labeled, replace=False)
            remaining_idx = np.setdiff1d(indices, labeled_idx)
            unlabeled_idx = rng.choice(remaining_idx, size=unlabeled_size, replace=False)

            estimate_labeled, ci_labeled = _labeled_only_mean_summary(y[labeled_idx], alpha=config.alpha)
            labeled_se = ci_labeled / (2.0 * _z_value(config.alpha))
            rows.append(
                {
                    "n_labeled": n_labeled,
                    "replication": replication,
                    "estimator": "labeled_only",
                    "estimate": estimate_labeled,
                    "mean_ci_length": ci_labeled,
                    "covered": float(
                        estimate_labeled - _z_value(config.alpha) * labeled_se
                        <= true_mean
                        <= estimate_labeled + _z_value(config.alpha) * labeled_se
                    ),
                }
            )

            estimate_raw, se_raw = official_ppi_summary(
                y[labeled_idx],
                score[labeled_idx],
                score[unlabeled_idx],
                alpha=config.alpha,
            )
            rows.append(
                {
                    "n_labeled": n_labeled,
                    "replication": replication,
                    "estimator": "raw_ppi",
                    "estimate": estimate_raw,
                    "mean_ci_length": 2.0 * _z_value(config.alpha) * se_raw,
                    "covered": float(
                        estimate_raw - _z_value(config.alpha) * se_raw
                        <= true_mean
                        <= estimate_raw + _z_value(config.alpha) * se_raw
                    ),
                }
            )

            calibrator = fit_one_dimensional_calibrator(
                score[labeled_idx],
                y[labeled_idx],
                method=calibrator_name,
            )
            calibrated_labeled = predict_with_calibrator(calibrator, score[labeled_idx])
            calibrated_unlabeled = predict_with_calibrator(calibrator, score[unlabeled_idx])
            estimate_calibrated, se_calibrated = official_ppi_summary(
                y[labeled_idx],
                calibrated_labeled,
                calibrated_unlabeled,
                alpha=config.alpha,
            )
            rows.append(
                {
                    "n_labeled": n_labeled,
                    "replication": replication,
                    "estimator": "calibrated_ppi",
                    "estimate": estimate_calibrated,
                    "mean_ci_length": 2.0 * _z_value(config.alpha) * se_calibrated,
                    "covered": float(
                        estimate_calibrated - _z_value(config.alpha) * se_calibrated
                        <= true_mean
                        <= estimate_calibrated + _z_value(config.alpha) * se_calibrated
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["n_labeled", "estimator"], as_index=False)
        .agg(
            mean_ci_length=("mean_ci_length", "mean"),
            coverage=("covered", "mean"),
        )
        .sort_values(["n_labeled", "estimator"])
        .reset_index(drop=True)
    )


def plot_toy_study(
    summary: pd.DataFrame,
    reference: dict[str, Any],
    config: ToyStudyConfig,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.4,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    bin_centers = np.asarray(reference["bin_centers"], dtype=float)
    bin_means = np.asarray(reference["bin_means"], dtype=float)
    bin_counts = np.asarray(reference["bin_counts"], dtype=float)

    ax = axes[0]
    ax.scatter(
        bin_centers,
        bin_means,
        s=18.0 + 0.05 * bin_counts,
        color=COLORS["bins"],
        alpha=0.75,
    )
    ax.plot(bin_centers, bin_means, color="#9A9A9A", linewidth=1.3, alpha=0.85)
    ax.set_title("Raw Score Has Signal, But Wrong Scale")
    ax.set_xlabel(reference["score_label"])
    ax.set_ylabel("Empirical human correctness")
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.98,
        0.05,
        (
            f"{reference['benchmark_label']} slice\n"
            f"Mean correctness = {reference['mean_y']:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )

    ax = axes[1]
    x_grid = np.asarray(reference["x_grid"], dtype=float)
    fitted_curve = np.asarray(reference["fitted_curve"], dtype=float)
    ax.plot(
        x_grid,
        fitted_curve,
        color=COLORS["calibrated_ppi"],
        linewidth=2.3,
        label=f"{reference['calibrator'].title()} calibration",
    )
    ax.scatter(
        bin_centers,
        bin_means,
        color=COLORS["bins"],
        alpha=0.7,
        s=18.0 + 0.05 * bin_counts,
        label="Empirical bins",
    )
    ax.text(
        0.98,
        0.05,
        f"Held-out squared error\nfalls by {100.0 * reference['mse_reduction']:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=COLORS["calibrated_ppi"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    ax.set_title("Calibration Restores Probability Scale")
    ax.set_xlabel(reference["score_label"])
    ax.set_ylabel("Predicted human correctness")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[2]
    for estimator in ESTIMATOR_ORDER:
        frame = summary[summary["estimator"] == estimator].sort_values("n_labeled")
        ax.plot(
            frame["n_labeled"],
            frame["mean_ci_length"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=COLORS[estimator],
            label=LABELS[estimator],
        )
    pivot = summary.pivot(index="n_labeled", columns="estimator", values="mean_ci_length")
    gain_25 = 1.0 - float(pivot.loc[25, "calibrated_ppi"]) / float(pivot.loc[25, "labeled_only"])
    gain_50 = 1.0 - float(pivot.loc[50, "calibrated_ppi"]) / float(pivot.loc[50, "labeled_only"])
    ax.text(
        0.98,
        0.05,
        (
            f"CI width drops by {100.0 * gain_25:.0f}% at n=25\n"
            f"and by {100.0 * gain_50:.0f}% at n=50\n"
            f"relative to labeled only."
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLORS["calibrated_ppi"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    ax.set_title("Calibration Buys Label Efficiency")
    ax.set_xlabel("Labeled examples")
    ax.set_ylabel(r"Mean 90\% CI length for mean correctness")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xticks(list(config.n_grid))

    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_summary_json(
    config: ToyStudyConfig,
    summary: pd.DataFrame,
    reference: dict[str, Any],
    output_path: Path,
) -> None:
    payload = {
        "config": {
            **asdict(config),
            "llm_cache_dir": str(config.llm_cache_dir),
            "dataset_cache_dir": str(config.dataset_cache_dir),
        },
        "reference": reference,
        "monte_carlo": summary.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_screening_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded one-mean calibration-vs-PPI toy figure.")
    parser.add_argument("--track", type=str, default="ppe_correctness")
    parser.add_argument("--evaluator", type=str, default=INTRO_PREFERRED_EVALUATOR)
    parser.add_argument("--benchmark", type=str, default="math_best_of_k")
    parser.add_argument("--calibrator", type=str, default="platt")
    parser.add_argument("--n-grid", type=str, default="25,50,100,200")
    parser.add_argument("--n-unlabeled", type=int, default=2000)
    parser.add_argument("--reference-fit-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulations/toy"))
    parser.add_argument("--screening-csv", type=Path, default=None)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("outputs/cache/llm_eval"))
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path("outputs/cache/ppi_datasets"))
    args = parser.parse_args()

    n_grid = tuple(int(piece) for piece in args.n_grid.split(","))
    config = ToyStudyConfig(
        track=args.track,
        evaluator=args.evaluator,
        benchmark=args.benchmark,
        calibrator=args.calibrator,
        n_grid=n_grid,
        n_unlabeled=args.n_unlabeled,
        reference_fit_n=args.reference_fit_n,
        llm_cache_dir=args.llm_cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
    )

    _ = _load_screening_csv(args.screening_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_monte_carlo(config, replications=args.replications, seed=args.seed)
    reference = build_reference_summary(config)
    plot_toy_study(summary, reference, config, args.output_dir / "fig_toy_calibration_ppi.pdf")
    write_summary_json(config, summary, reference, args.output_dir / "toy_calibration_summary.json")


if __name__ == "__main__":
    main()
