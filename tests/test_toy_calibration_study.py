from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.llm_eval_models import (
    build_ppe_correctness_unit_id,
    write_score_cache,
    write_track_rows_cache,
)
from experiments.grounded_toy_screening import (
    CandidateData,
    ScreeningConfig,
    build_arm_data,
    load_ppe_candidate,
    screen_candidates,
)
from experiments.toy_calibration_study import (
    ToyStudyConfig,
    build_reference_summary,
    plot_toy_study,
    run_monte_carlo,
    write_summary_json,
)


def write_fake_toy_llm_cache(cache_dir: Path) -> None:
    rows = []
    for benchmark in ("mmlu_pro_best_of_k", "math_best_of_k"):
        for target_model in ("alpha-generator", "beta-generator"):
            for idx in range(120):
                rows.append(
                    {
                        "unit_id": build_ppe_correctness_unit_id(benchmark, target_model, f"{benchmark}-q{idx}", 0, 1),
                        "track": "ppe_correctness",
                        "benchmark": benchmark,
                        "prompt": f"{benchmark} prompt {idx}",
                        "response_a": f"A-{idx}",
                        "response_b": f"B-{idx}",
                        "model_a": target_model,
                        "model_b": target_model,
                        "target_model": target_model,
                        "subset": "math" if idx % 2 else "coding",
                        "y": 1.0 if idx % 3 else 0.0,
                    }
                )
    frame = pd.DataFrame(rows)
    margins = pd.Series(
        [
            (0.08 if float(row["y"]) == 1.0 else -0.08) + 0.01 * ((idx % 5) - 2)
            for idx, row in enumerate(frame.to_dict(orient="records"))
        ],
        dtype=float,
    )
    score_frame = pd.DataFrame(
        {
            "unit_id": frame["unit_id"].astype(str),
            "evaluator": "armorm_llama3_8b_v0_1",
            "score_a": margins / 2.0,
            "score_b": -margins / 2.0,
            "margin": margins,
            "score_source": "test_cache",
        }
    )
    write_track_rows_cache(cache_dir, "ppe_correctness", frame)
    write_score_cache(cache_dir, "ppe_correctness", "armorm_llama3_8b_v0_1", score_frame)


def test_screening_prefers_declared_intro_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    ppe_candidates = {
        "armorm_llama3_8b_v0_1": CandidateData(
            candidate_id="armorm_llama3_8b_v0_1",
            display_name="ArmoRM",
            family="ppe",
            score_label="Raw margin",
            frame=pd.DataFrame(),
        ),
        "athene_rm_8b": CandidateData(
            candidate_id="athene_rm_8b",
            display_name="Athene",
            family="ppe",
            score_label="Raw margin",
            frame=pd.DataFrame(),
        ),
    }
    classic_candidates = {
        "forest": CandidateData(
            candidate_id="forest",
            display_name="Forest",
            family="classic",
            score_label="Raw score",
            frame=pd.DataFrame(),
        )
    }
    candidate_rows = {
        "armorm_llama3_8b_v0_1": {"passes_intro_thresholds": True, "believable_intro_score": 0.65},
        "athene_rm_8b": {"passes_intro_thresholds": True, "believable_intro_score": 0.80},
        "forest": {"passes_intro_thresholds": True, "believable_intro_score": 0.40},
    }

    def fake_load_ppe_candidate(_cache_dir: Path, _track: str, evaluator: str) -> CandidateData:
        return ppe_candidates[evaluator]

    def fake_load_classic_candidate(_cache_dir: Path, dataset_name: str) -> CandidateData:
        return classic_candidates[dataset_name]

    def fake_evaluate_candidate(candidate: CandidateData, _config: ScreeningConfig, *, seed_offset: int) -> dict[str, object]:
        del seed_offset
        base = candidate_rows[candidate.candidate_id]
        return {
            "candidate_id": candidate.candidate_id,
            "display_name": candidate.display_name,
            "family": candidate.family,
            "score_label": candidate.score_label,
            "passes_intro_thresholds": base["passes_intro_thresholds"],
        }

    def fake_believable_intro_score(row: pd.Series, *, alpha: float) -> float:
        del alpha
        return float(candidate_rows[str(row["candidate_id"])]["believable_intro_score"])

    monkeypatch.setitem(screen_candidates.__globals__, "load_ppe_candidate", fake_load_ppe_candidate)
    monkeypatch.setitem(screen_candidates.__globals__, "load_classic_candidate", fake_load_classic_candidate)
    monkeypatch.setitem(screen_candidates.__globals__, "evaluate_candidate", fake_evaluate_candidate)
    monkeypatch.setitem(screen_candidates.__globals__, "believable_intro_score", fake_believable_intro_score)

    screening = screen_candidates(
        ScreeningConfig(
            evaluators=("armorm_llama3_8b_v0_1", "athene_rm_8b"),
            classic_datasets=("forest",),
        )
    )

    assert screening.loc[screening["selected_intro"], "candidate_id"].tolist() == ["armorm_llama3_8b_v0_1"]


def test_armorm_top_and_bottom_deciles_have_larger_label_gap_than_raw_gap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "llm_cache"
    write_fake_toy_llm_cache(cache_dir)

    candidate = load_ppe_candidate(
        cache_dir,
        "ppe_correctness",
        "armorm_llama3_8b_v0_1",
    )
    arm_data = build_arm_data(candidate, (0.1, 0.9))
    label_gap = float(arm_data.arm_a["y"].mean() - arm_data.arm_b["y"].mean())
    raw_gap = float(arm_data.arm_a["score"].mean() - arm_data.arm_b["score"].mean())

    assert label_gap > 4.5 * abs(raw_gap)


def test_grounded_toy_calibrated_ppi_improves_ci_length(tmp_path: Path) -> None:
    cache_dir = tmp_path / "llm_cache"
    write_fake_toy_llm_cache(cache_dir)

    config = ToyStudyConfig(n_grid=(25, 50), n_unlabeled=2000, llm_cache_dir=cache_dir)
    summary = run_monte_carlo(config, replications=80, seed=20260420)
    reference = build_reference_summary(config)

    pivot = summary.pivot(index="n_labeled", columns="estimator", values="mean_ci_length")
    coverage = summary.pivot(index="n_labeled", columns="estimator", values="coverage")

    assert (pivot["calibrated_ppi"] < pivot["labeled_only"]).all()
    assert (pivot["calibrated_ppi"] < pivot["raw_ppi"]).all()
    assert (pivot["raw_ppi"] <= 1.05 * pivot["labeled_only"]).all()
    assert reference["calibrated_mse"] < reference["raw_mse"]
    assert coverage.loc[25, "calibrated_ppi"] == pytest.approx(0.9, abs=0.12)
    assert coverage.loc[50, "calibrated_ppi"] == pytest.approx(0.9, abs=0.12)

    figure_path = tmp_path / "fig_toy_calibration_ppi.pdf"
    summary_path = tmp_path / "toy_calibration_summary.json"
    plot_toy_study(summary, reference, config, figure_path)
    write_summary_json(config, summary, reference, summary_path)

    assert figure_path.exists()
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["reference"]["benchmark"] == "math_best_of_k"
    assert payload["reference"]["calibrated_mse"] < payload["reference"]["raw_mse"]
    assert pd.DataFrame(payload["monte_carlo"]).shape[0] == 6
