from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))


ESTIMATOR_LABELS = {
    "labeled_only": "Naive",
    "ppi": "PPI",
    "ppi_plus_plus": "PPI++",
    "aipw_em": "AIPW-EM",
    "aipw": "AIPW",
    "auto_calibration": "AutoCal",
    "monotone_spline": "MonoSpline",
    "affine_calibration": "LinearCal",
    "platt_calibration": "Platt",
    "isotonic_calibration": "IsoCal",
    "calibrated_plugin": "Venn-Abers",
}

COLORS = {
    "labeled_only": "#4D4D4D",
    "ppi": "#D55E00",
    "ppi_plus_plus": "#C44E52",
    "aipw_em": "#4E79A7",
    "aipw": "#8C564B",
    "auto_calibration": "#E69F00",
    "monotone_spline": "#0B6E4F",
    "affine_calibration": "#009E73",
    "platt_calibration": "#56B4E9",
    "isotonic_calibration": "#7A3E9D",
    "calibrated_plugin": "#0072B2",
}

ESTIMATOR_ORDER = [
    "labeled_only",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "affine_calibration",
    "isotonic_calibration",
    "platt_calibration",
    "calibrated_plugin",
]

SIMULATION_PAPER_LEGEND_ORDER = [
    "labeled_only",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "monotone_spline",
    "affine_calibration",
    "isotonic_calibration",
]

PRIMARY_ESTIMATORS = {
    "affine_calibration",
    "aipw",
    "auto_calibration",
    "monotone_spline",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "isotonic_calibration",
}


def draw_order(order: list[str]) -> list[str]:
    secondary = [name for name in order if name not in PRIMARY_ESTIMATORS]
    primary = [
        name
        for name in order
        if name in PRIMARY_ESTIMATORS and name not in {"aipw_em", "ppi_plus_plus", "ppi", "aipw"}
    ]
    tail = [name for name in ["aipw_em", "ppi_plus_plus", "ppi", "aipw"] if name in order]
    return secondary + primary + tail


def line_width(estimator: str, default: float = 1.8) -> float:
    if estimator in PRIMARY_ESTIMATORS:
        return default + 0.35
    return max(1.0, default - 0.45)


def line_zorder(estimator: str) -> int:
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


def line_alpha(estimator: str) -> float:
    return 1.0 if estimator in PRIMARY_ESTIMATORS else 0.5


def build_legend(order: list[str], available: set[str]) -> tuple[list[Line2D], list[str]]:
    legend_estimators = [name for name in order if name in available]
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker="o",
            linewidth=line_width(name),
            alpha=line_alpha(name),
            label=ESTIMATOR_LABELS[name],
        )
        for name in legend_estimators
    ]
    labels = [ESTIMATOR_LABELS[name] for name in legend_estimators]
    return handles, labels

QUALITY_LABELS = {
    "well_calibrated": "Mildly miscalibrated score",
    "poorly_calibrated": "Poorly calibrated score",
}


def plot_rmse(summary: pd.DataFrame, output_path: Path) -> None:
    keep = summary[summary["dgp"] == "monotone"].copy()
    row_keys = [("monotone", "well_calibrated"), ("monotone", "poorly_calibrated")]
    col_keys = sorted(keep["n_labeled"].unique())
    fig, axes = plt.subplots(len(row_keys), len(col_keys), figsize=(16, 12), sharex=True, sharey=True)
    for row_idx, (dgp, quality) in enumerate(row_keys):
        for col_idx, n in enumerate(col_keys):
            ax = axes[row_idx, col_idx]
            sub = keep[
                (keep["dgp"] == dgp)
                & (keep["score_quality"] == quality)
                & (keep["n_labeled"] == n)
            ]
            for estimator in draw_order(ESTIMATOR_ORDER):
                frame = sub[sub["estimator"] == estimator].sort_values("unlabeled_ratio")
                ax.plot(
                    frame["unlabeled_ratio"],
                    frame["rmse"],
                    marker="o",
                    color=COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    linewidth=line_width(estimator),
                    zorder=line_zorder(estimator),
                    alpha=line_alpha(estimator),
                )
            if row_idx == 0:
                ax.set_title(f"n={n}")
            if col_idx == 0:
                ax.set_ylabel(f"{dgp}\n{QUALITY_LABELS[quality]}\nRMSE")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(sub["unlabeled_ratio"].unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    handles, labels = build_legend(ESTIMATOR_ORDER, set(keep["estimator"].unique()))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.7,
        columnspacing=0.9,
    )
    fig.supxlabel("Unlabeled-to-labeled ratio N/n")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_bias_coverage(summary: pd.DataFrame, output_path: Path) -> None:
    keep = summary[summary["dgp"] == "monotone"].copy()
    target_ratio = keep["unlabeled_ratio"].max()
    keep = keep[keep["unlabeled_ratio"] == target_ratio]
    row_keys = [("monotone", "well_calibrated"), ("monotone", "poorly_calibrated")]
    fig, axes = plt.subplots(len(row_keys), 2, figsize=(14.5, 12), sharex=True)
    for row_idx, (dgp, quality) in enumerate(row_keys):
        sub = keep[(keep["dgp"] == dgp) & (keep["score_quality"] == quality)]
        for estimator in draw_order(ESTIMATOR_ORDER):
            frame = sub[sub["estimator"] == estimator].sort_values("n_labeled")
            axes[row_idx, 0].plot(
                frame["n_labeled"],
                frame["mean_bias"],
                marker="o",
                color=COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
                linewidth=line_width(estimator),
                zorder=line_zorder(estimator),
                alpha=line_alpha(estimator),
            )
            axes[row_idx, 1].plot(
                frame["n_labeled"],
                frame["coverage"],
                marker="o",
                color=COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
                linewidth=line_width(estimator),
                zorder=line_zorder(estimator),
                alpha=line_alpha(estimator),
            )
        axes[row_idx, 0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[row_idx, 1].axhline(0.95, color="black", linewidth=0.8, linestyle="--")
        axes[row_idx, 0].set_ylabel(f"{dgp}\n{QUALITY_LABELS[quality]}")
    axes[0, 0].set_title(f"Bias at N/n={target_ratio}")
    axes[0, 1].set_title(f"Coverage at N/n={target_ratio}")
    axes[-1, 0].set_xlabel("Labeled sample size n")
    axes[-1, 1].set_xlabel("Labeled sample size n")
    handles, labels = build_legend(ESTIMATOR_ORDER, set(keep["estimator"].unique()))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.7,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_gain(summary: pd.DataFrame, output_path: Path) -> None:
    sub = summary[summary["estimator"] == "calibrated_plugin"].copy()
    row_keys = [("monotone", "well_calibrated"), ("monotone", "poorly_calibrated")]
    fig, axes = plt.subplots(len(row_keys), 2, figsize=(10, 12), sharex=True, sharey=True)
    for row_idx, (dgp, quality) in enumerate(row_keys):
        frame = sub[(sub["dgp"] == dgp) & (sub["score_quality"] == quality)].copy()
        frame["gain_vs_label"] = 100.0 * (1.0 - frame["rmse_ratio_vs_label"])
        frame["gain_vs_ppi"] = 100.0 * (1.0 - frame["rmse_ratio_vs_ppi"])
        for col_idx, metric in enumerate(["gain_vs_label", "gain_vs_ppi"]):
            pivot = frame.pivot(index="n_labeled", columns="unlabeled_ratio", values=metric).sort_index()
            ax = axes[row_idx, col_idx]
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-30, vmax=30)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.tolist())
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist())
            if row_idx == 0:
                ax.set_title("Gain vs labeled only" if col_idx == 0 else "Gain vs PPI")
            if col_idx == 0:
                ax.set_ylabel(f"{dgp}\n{QUALITY_LABELS[quality]}\nn")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Percent RMSE improvement")
    fig.supxlabel("Unlabeled-to-labeled ratio N/n")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_main(summary: pd.DataFrame, output_path: Path) -> None:
    keep = summary[summary["dgp"] == "monotone"].copy()
    ratio_keys = sorted(set(keep["unlabeled_ratio"].unique()).intersection({1, keep["unlabeled_ratio"].max()}))
    keep["abs_mean_bias"] = keep["mean_bias"].abs()
    ppi_sd = (
        keep[keep["estimator"] == "ppi"][
            ["dgp", "score_quality", "n_labeled", "unlabeled_ratio", "emp_sd"]
        ]
        .rename(columns={"emp_sd": "ppi_emp_sd"})
        .drop_duplicates()
    )
    keep = keep.merge(
        ppi_sd,
        on=["dgp", "score_quality", "n_labeled", "unlabeled_ratio"],
        how="left",
    )
    keep["rel_eff_vs_ppi"] = (keep["ppi_emp_sd"] / keep["emp_sd"]) ** 2
    row_keys = [("monotone", "poorly_calibrated", ratio) for ratio in ratio_keys]
    estimators = SIMULATION_PAPER_LEGEND_ORDER
    metrics = [
        ("abs_mean_bias", "Absolute Bias"),
        ("emp_sd", "Empirical SD"),
        ("coverage", "Coverage"),
        ("rel_eff_vs_ppi", "Rel. Eff. vs PPI"),
    ]
    fig, axes = plt.subplots(len(row_keys), len(metrics), figsize=(16.2, 5.4), sharex=True)
    if len(row_keys) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, (dgp, quality, ratio) in enumerate(row_keys):
        sub = keep[
            (keep["dgp"] == dgp)
            & (keep["score_quality"] == quality)
            & (keep["unlabeled_ratio"] == ratio)
        ]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for estimator in draw_order(estimators):
                frame = sub[sub["estimator"] == estimator].sort_values("n_labeled")
                ax.plot(
                    frame["n_labeled"],
                    frame[metric],
                    marker="o",
                    color=COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    linewidth=line_width(estimator),
                    zorder=line_zorder(estimator),
                    alpha=line_alpha(estimator),
                )
            if row_idx == 0:
                ax.set_title(title)
            if metric == "coverage":
                ax.axhline(0.95, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylim(0.84, 1.01)
            if metric == "rel_eff_vs_ppi":
                ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            if col_idx == 0:
                ax.set_ylabel(f"N/n={ratio}")
            if row_idx == len(row_keys) - 1:
                ax.set_xlabel("Labeled sample size n")
            ax.margins(x=0.02)
    handles, labels = build_legend(SIMULATION_PAPER_LEGEND_ORDER, set(keep["estimator"].unique()))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        handlelength=1.55,
        handletextpad=0.35,
        columnspacing=0.72,
        fontsize=10.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=0.6, h_pad=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/simulations"))
    parser.add_argument(
        "--main-text-only",
        action="store_true",
        help="Generate only the simulation figure used in the paper main text.",
    )
    args = parser.parse_args()
    summary = pd.read_csv(args.input_dir / "summary.csv")
    plot_simulation_main(summary, args.input_dir / "fig_simulation_main.pdf")
    if args.main_text_only:
        return
    plot_rmse(summary, args.input_dir / "fig_rmse.pdf")
    plot_bias_coverage(summary, args.input_dir / "fig_bias_coverage.pdf")
    plot_efficiency_gain(summary, args.input_dir / "fig_efficiency_gain.pdf")


if __name__ == "__main__":
    main()
