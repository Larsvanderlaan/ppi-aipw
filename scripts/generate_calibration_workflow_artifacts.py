from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ppi_aipw import calibration_diagnostics, fit_calibrator, plot_calibration  # noqa: E402


CACHE_DIR = ROOT / "outputs" / "cache" / "ppi_datasets"
DOCS_ASSETS = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs" / "calibration_workflows"


@dataclass(frozen=True)
class CaseStudySpec:
    dataset_name: str
    n_labeled: int
    seed: int
    calibrated_method: str
    calibrated_label: str
    smooth_label: str
    docs_stem: str
    title: str
    note: str


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def clip_unit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def ensure_dirs() -> None:
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_case_split(dataset_name: str, *, n_labeled: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(CACHE_DIR / f"{dataset_name}.npz")
    y_all = np.asarray(data["Y"], dtype=float)
    yhat_all = np.asarray(data["Yhat"], dtype=float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_all.shape[0])
    labeled_idx = perm[:n_labeled]
    unlabeled_idx = perm[n_labeled:]
    return y_all[labeled_idx], yhat_all[labeled_idx], yhat_all[unlabeled_idx]


def ppi_implied_scores(yhat_labeled: np.ndarray, *, n_labeled: int, n_unlabeled: int) -> np.ndarray:
    rho = n_labeled / float(n_labeled + n_unlabeled)
    return np.asarray(yhat_labeled, dtype=float) / (1.0 - rho)


def diagnostics_for_scores(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
    isocal_backend: str = "xgboost",
) -> object:
    model = fit_calibrator(
        y,
        scores,
        method=method,
        isocal_backend=isocal_backend,
    )
    return calibration_diagnostics(model, y, scores, num_bins=10)


def comparison_diagnostics_for_scores(y: np.ndarray, scores: np.ndarray) -> object:
    return diagnostics_for_scores(y, scores, method="monotone_spline")


def save_case_study_figure(spec: CaseStudySpec) -> tuple[Path, Path]:
    y_labeled, yhat_labeled, yhat_unlabeled = load_case_split(
        spec.dataset_name,
        n_labeled=spec.n_labeled,
        seed=spec.seed,
    )
    n_unlabeled = yhat_unlabeled.shape[0]
    ppi_scores = ppi_implied_scores(
        yhat_labeled,
        n_labeled=spec.n_labeled,
        n_unlabeled=n_unlabeled,
    )
    calibrated_scores = np.asarray(
        fit_calibrator(
            y_labeled,
            yhat_labeled,
            method=spec.calibrated_method,
        ).predict(yhat_labeled),
        dtype=float,
    )
    smooth_scores = np.asarray(
        fit_calibrator(
            y_labeled,
            yhat_labeled,
            method="monotone_spline",
        ).predict(yhat_labeled),
        dtype=float,
    )
    curves = [
        ("AIPW raw score", comparison_diagnostics_for_scores(y_labeled, yhat_labeled), "#1f77b4"),
        ("PPI-implied score", comparison_diagnostics_for_scores(y_labeled, ppi_scores), "#ff7f0e"),
        (spec.calibrated_label, comparison_diagnostics_for_scores(y_labeled, calibrated_scores), "#2ca02c"),
        (spec.smooth_label, comparison_diagnostics_for_scores(y_labeled, smooth_scores), "#9467bd"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    all_grid = np.concatenate([curve.per_output[0].grid_scores for _, curve, _ in curves])
    all_fit = np.concatenate([curve.per_output[0].fitted_curve for _, curve, _ in curves])
    lo = float(min(np.min(all_grid), np.min(all_fit)))
    hi = float(max(np.max(all_grid), np.max(all_fit)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.75", linewidth=1.6, label="Identity")
    for label, diagnostics, color in curves:
        record = diagnostics.per_output[0]
        ax.plot(record.grid_scores, record.fitted_curve, linewidth=2.6, color=color, label=label)
    ax.set_xlabel("Score value")
    ax.set_ylabel("Observed outcome")
    ax.grid(alpha=0.12)
    ax.legend(frameon=True, fontsize=10.5, loc="upper left")

    # The webpage already supplies the benchmark title and explanation, so let
    # the exported image spend its area on the plot itself.
    fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.14)

    docs_png = DOCS_ASSETS / f"{spec.docs_stem}.png"
    out_pdf = OUTPUT_DIR / f"{spec.docs_stem}.pdf"
    fig.savefig(docs_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return docs_png, out_pdf


def make_binary_scenario(kind: str, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=320)
    latent = 0.9 * x - 0.35 * np.sin(1.2 * x)
    p_true = clip_unit(sigmoid(latent))
    y = (rng.random(size=x.shape[0]) < p_true).astype(float)

    if kind == "well_calibrated":
        raw = clip_unit(p_true + rng.normal(scale=0.02, size=x.shape[0]))
    elif kind == "miscalibrated":
        raw = clip_unit(sigmoid(1.9 * np.log(p_true / (1.0 - p_true)) - 0.35))
        raw = clip_unit(raw + rng.normal(scale=0.015, size=x.shape[0]))
    else:
        raise ValueError(f"Unknown scenario '{kind}'.")

    return y, raw


def save_synthetic_method_gallery() -> list[Path]:
    methods = [
        ("Raw score", "aipw", {}, "raw"),
        ("Linear", "linear", {}, "linear"),
        ("MonoSpline", "monotone_spline", {}, "monotone_spline"),
        ("Iso smooth", "isotonic", {"isocal_backend": "xgboost"}, "isotonic_smooth"),
        ("Iso step", "isotonic", {"isocal_backend": "sklearn"}, "isotonic_step"),
    ]
    scenario_specs = [
        ("well_calibrated", "Well-calibrated score"),
        ("miscalibrated", "Classically miscalibrated score"),
    ]

    output_paths: list[Path] = []
    combined_png = OUTPUT_DIR / "synthetic_calibration_workflow_gallery.png"
    combined_pdf = OUTPUT_DIR / "synthetic_calibration_workflow_gallery.pdf"

    fig, axes = plt.subplots(
        len(scenario_specs),
        len(methods),
        figsize=(20, 8.6),
        constrained_layout=True,
    )

    with PdfPages(combined_pdf) as pdf:
        for row_idx, (scenario_name, scenario_title) in enumerate(scenario_specs):
            y, raw_scores = make_binary_scenario(scenario_name, seed=100 + row_idx)
            row_fig, row_axes = plt.subplots(1, len(methods), figsize=(20, 4.1), constrained_layout=True)
            for col_idx, (title, method, kwargs, slug) in enumerate(methods):
                diagnostics = diagnostics_for_scores(y, raw_scores, method=method, **kwargs)

                plot_calibration(diagnostics, ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(title, fontsize=11.5, fontweight="bold")
                axes[row_idx, col_idx].grid(alpha=0.12)

                plot_calibration(diagnostics, ax=row_axes[col_idx])
                row_axes[col_idx].set_title(title, fontsize=11.5, fontweight="bold")
                row_axes[col_idx].grid(alpha=0.12)

                single_fig, single_ax = plt.subplots(figsize=(4.4, 4.1), constrained_layout=True)
                plot_calibration(diagnostics, ax=single_ax)
                single_ax.set_title(f"{scenario_title}: {title}", fontsize=11.5, fontweight="bold")
                single_ax.grid(alpha=0.12)
                single_path = OUTPUT_DIR / f"{scenario_name}_{slug}.png"
                single_fig.savefig(single_path, dpi=220, bbox_inches="tight")
                output_paths.append(single_path)
                plt.close(single_fig)

            row_fig.suptitle(f"{scenario_title} workflow", fontsize=15, fontweight="bold")
            row_png = OUTPUT_DIR / f"{scenario_name}_workflow.png"
            row_fig.savefig(row_png, dpi=220, bbox_inches="tight")
            pdf.savefig(row_fig, bbox_inches="tight")
            output_paths.append(row_png)
            plt.close(row_fig)

        fig.suptitle("Calibration workflow gallery", fontsize=17, fontweight="bold")
        fig.savefig(combined_png, dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        output_paths.extend([combined_png, combined_pdf])
        plt.close(fig)

    return output_paths


def save_case_study_gallery(case_specs: list[CaseStudySpec]) -> list[Path]:
    combined_pdf = OUTPUT_DIR / "case_study_calibration_gallery.pdf"
    pdf_paths: list[Path] = []
    with PdfPages(combined_pdf) as pdf:
        for spec in case_specs:
            png_path, pdf_path = save_case_study_figure(spec)
            fig = plt.figure(figsize=(13.5, 4.2))
            img = plt.imread(png_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pdf_paths.extend([png_path, pdf_path])
    pdf_paths.append(combined_pdf)
    return pdf_paths


def main() -> None:
    ensure_dirs()
    case_specs = [
        CaseStudySpec(
            dataset_name="census_income",
            n_labeled=1000,
            seed=0,
            calibrated_method="linear",
            calibrated_label="Linear calibration",
            smooth_label="MonoSpline",
            docs_stem="case_study_census_income_calibration",
            title="census_income: representative split with n = 1000",
            note="Smooth out-of-fold diagnostic curves on a fixed split. AIPW uses the raw score, PPI uses the implied rescaled score, and linear plus smooth monotone calibration are two corrected alternatives.",
        ),
        CaseStudySpec(
            dataset_name="forest",
            n_labeled=500,
            seed=0,
            calibrated_method="linear",
            calibrated_label="Linear calibration",
            smooth_label="MonoSpline",
            docs_stem="case_study_forest_calibration_n500",
            title="forest: representative split with n = 500",
            note="Smooth out-of-fold diagnostic curves on a fixed split. At n = 500, the raw AIPW score is already close to the outcome scale while the PPI-style rescaling visibly overshoots.",
        ),
    ]
    save_case_study_gallery(case_specs)
    save_synthetic_method_gallery()


if __name__ == "__main__":
    main()
