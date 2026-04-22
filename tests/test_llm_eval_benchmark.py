from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.llm_eval_benchmark import (
    ALL_ESTIMATORS,
    build_ppe_correctness_panel,
    build_ppe_correctness_rows_from_result_records,
    build_split_schedule,
    build_ppe_panel,
    main,
    normalize_rewardbench_rows,
    orient_ppe_rows_for_target,
    ppe_target_outcome,
    run_target_experiment,
    select_top_ppe_models,
    summarize_ppe_rankings,
    top1_regret,
)
from experiments.llm_eval_models import (
    PPE_HUMAN_BENCHMARK,
    REWARDBENCH_BENCHMARK,
    build_ppe_correctness_score_frame_from_records,
    build_ppe_correctness_unit_id,
    configure_offline_hf_runtime,
    load_score_cache,
    load_track_rows_cache,
    resolve_local_hf_snapshot_path,
    row_cache_path,
    score_cache_path,
    write_frame_jsonl_gz,
    write_score_cache,
    write_track_rows_cache,
)


def build_fake_ppe_rows() -> pd.DataFrame:
    rows = []
    for idx in range(80):
        winner = "model_a" if idx % 5 else "tie"
        rows.append(
            {
                "unit_id": f"ppe-alpha-{idx}",
                "track": "ppe_human",
                "benchmark": PPE_HUMAN_BENCHMARK,
                "prompt": f"alpha prompt {idx}",
                "response_a": f"alpha response {idx}",
                "response_b": f"gamma response {idx}",
                "model_a": "alpha",
                "model_b": "gamma",
                "winner": winner,
                "language": "English" if idx % 3 else "Spanish",
                "language_group": "english" if idx % 3 else "non_english",
                "hard_prompt": bool(idx % 2),
                "math_prompt": bool(idx % 4 == 0),
                "is_code": bool(idx % 5 == 0),
            }
        )
    for idx in range(80):
        winner = "model_a" if idx % 4 else "tie"
        rows.append(
            {
                "unit_id": f"ppe-beta-{idx}",
                "track": "ppe_human",
                "benchmark": PPE_HUMAN_BENCHMARK,
                "prompt": f"beta prompt {idx}",
                "response_a": f"beta response {idx}",
                "response_b": f"delta response {idx}",
                "model_a": "beta",
                "model_b": "delta",
                "winner": winner,
                "language": "English" if idx % 2 else "German",
                "language_group": "english" if idx % 2 else "non_english",
                "hard_prompt": bool(idx % 2),
                "math_prompt": bool(idx % 3 == 0),
                "is_code": bool(idx % 6 == 0),
            }
        )
    return pd.DataFrame(rows)


def build_fake_ppe_scores(evaluator: str, rows: pd.DataFrame) -> pd.DataFrame:
    margins = []
    for idx, winner in enumerate(rows["winner"].astype(str)):
        base = 0.75 - 0.01 * (idx % 7)
        if winner == "model_a":
            margin = base
        elif winner == "model_b":
            margin = -base
        else:
            margin = 0.05 * ((idx % 3) - 1)
        margins.append(margin)
    margins_arr = np.asarray(margins, dtype=float)
    return pd.DataFrame(
        {
            "unit_id": rows["unit_id"].astype(str),
            "evaluator": evaluator,
            "score_a": margins_arr / 2.0,
            "score_b": -margins_arr / 2.0,
            "margin": margins_arr,
            "score_source": "test_cache",
        }
    )


def build_fake_rewardbench_rows() -> pd.DataFrame:
    rows = []
    subsets = ["alpacaeval-easy", "mt-bench-hard", "math-prm", "hep-python"]
    for idx in range(96):
        response_a_is_chosen = idx % 2 == 0
        rows.append(
            {
                "unit_id": f"rb-{idx}",
                "track": "rewardbench_v1",
                "benchmark": REWARDBENCH_BENCHMARK,
                "prompt": f"rewardbench prompt {idx}",
                "response_a": f"{'chosen' if response_a_is_chosen else 'rejected'} response {idx}",
                "response_b": f"{'rejected' if response_a_is_chosen else 'chosen'} response {idx}",
                "model_a": "chosen-model" if response_a_is_chosen else "rejected-model",
                "model_b": "rejected-model" if response_a_is_chosen else "chosen-model",
                "subset": subsets[idx % len(subsets)],
                "y": 1.0 if response_a_is_chosen else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_fake_ppe_correctness_rows() -> pd.DataFrame:
    rows = []
    benchmarks = ["mmlu_pro_best_of_k", "math_best_of_k"]
    for benchmark in benchmarks:
        for target_model in ("alpha-generator", "beta-generator"):
            for idx in range(40):
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
    return pd.DataFrame(rows)


def build_fake_ppe_correctness_scores(evaluator: str, rows: pd.DataFrame) -> pd.DataFrame:
    margins = np.asarray(
        [
            (0.9 if float(row["y"]) == 1.0 else -0.9) + 0.05 * ((idx % 5) - 2)
            for idx, row in enumerate(rows.to_dict(orient="records"))
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "unit_id": rows["unit_id"].astype(str),
            "evaluator": evaluator,
            "score_a": margins / 2.0,
            "score_b": -margins / 2.0,
            "margin": margins,
            "score_source": "test_cache",
        }
    )


def build_fake_rewardbench_scores(evaluator: str, rows: pd.DataFrame) -> pd.DataFrame:
    margins = np.asarray(
        [
            (0.8 if float(row["y"]) == 1.0 else -0.8) + (0.1 if idx % 6 else -0.15)
            for idx, row in enumerate(rows.to_dict(orient="records"))
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "unit_id": rows["unit_id"].astype(str),
            "evaluator": evaluator,
            "score_a": margins / 2.0,
            "score_b": -margins / 2.0,
            "margin": margins,
            "score_source": "test_cache",
        }
    )


def populate_fake_caches(cache_dir: Path) -> None:
    ppe_rows = build_fake_ppe_rows()
    ppe_correctness_rows = build_fake_ppe_correctness_rows()
    rewardbench_rows = build_fake_rewardbench_rows()
    write_track_rows_cache(cache_dir, "ppe_human", ppe_rows)
    write_track_rows_cache(cache_dir, "ppe_correctness", ppe_correctness_rows)
    write_track_rows_cache(cache_dir, "rewardbench_v1", rewardbench_rows)
    for evaluator in (
        "skywork_reward_llama3_1_8b",
        "armorm_llama3_8b_v0_1",
        "athene_rm_8b",
        "starling_rm_7b_alpha",
    ):
        write_score_cache(cache_dir, "ppe_human", evaluator, build_fake_ppe_scores(evaluator, ppe_rows))
        write_score_cache(
            cache_dir,
            "ppe_correctness",
            evaluator,
            build_fake_ppe_correctness_scores(evaluator, ppe_correctness_rows),
        )
    write_score_cache(
        cache_dir,
        "rewardbench_v1",
        "skywork_qwen3_4b",
        build_fake_rewardbench_scores("skywork_qwen3_4b", rewardbench_rows),
    )


def test_top1_regret_returns_true_gap_for_selected_model() -> None:
    estimated = np.asarray([0.4, 0.9, 0.3], dtype=float)
    truth = np.asarray([0.8, 0.6, 0.2], dtype=float)
    assert top1_regret(estimated, truth) == pytest.approx(0.2)


def test_summarize_ppe_rankings_includes_top1_regret() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "track": "ppe_human",
                "evaluator": "eval_a",
                "n_labeled": 25,
                "replicate": 0,
                "estimator": "aipw",
                "target_model": "alpha",
                "true_theta": 0.8,
                "estimate": 0.4,
            },
            {
                "track": "ppe_human",
                "evaluator": "eval_a",
                "n_labeled": 25,
                "replicate": 0,
                "estimator": "aipw",
                "target_model": "beta",
                "true_theta": 0.6,
                "estimate": 0.9,
            },
            {
                "track": "ppe_human",
                "evaluator": "eval_a",
                "n_labeled": 25,
                "replicate": 0,
                "estimator": "aipw",
                "target_model": "gamma",
                "true_theta": 0.2,
                "estimate": 0.3,
            },
        ]
    )

    raw_ranking, macro = summarize_ppe_rankings(raw_df)

    assert raw_ranking["top1_regret"].tolist() == pytest.approx([0.2])
    assert macro["top1_regret"].tolist() == pytest.approx([0.2])
    assert macro["top1_accuracy"].tolist() == pytest.approx([0.0])


def test_ppe_orientation_flips_score_sign_with_model_order() -> None:
    rows = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "track": "ppe_human",
                "prompt": "prompt",
                "response_a": "A",
                "response_b": "B",
                "model_a": "alpha",
                "model_b": "beta",
                "winner": "model_a",
                "language": "English",
                "language_group": "english",
                "hard_prompt": True,
                "math_prompt": False,
                "is_code": False,
            }
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "evaluator": "pairrm",
                "score_a": 0.4,
                "score_b": -0.2,
                "margin": 0.6,
                "score_source": "test_cache",
            }
        ]
    )
    panel = build_ppe_panel(rows, scores, target_models=["alpha", "beta"])
    alpha_row = panel[panel["target_model"] == "alpha"].iloc[0]
    beta_row = panel[panel["target_model"] == "beta"].iloc[0]
    assert alpha_row["raw_score"] == pytest.approx(0.6)
    assert beta_row["raw_score"] == pytest.approx(-0.6)
    assert alpha_row["y"] == pytest.approx(1.0)
    assert beta_row["y"] == pytest.approx(0.0)


def test_ppe_ties_encode_half_win() -> None:
    rows = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "track": "ppe_human",
                "prompt": "prompt",
                "response_a": "A",
                "response_b": "B",
                "model_a": "alpha",
                "model_b": "beta",
                "winner": "tie",
                "language": "English",
                "language_group": "english",
                "hard_prompt": False,
                "math_prompt": False,
                "is_code": False,
            }
        ]
    )
    oriented = orient_ppe_rows_for_target(rows, "alpha")
    assert oriented["y"].iloc[0] == pytest.approx(0.5)
    assert ppe_target_outcome("tie", target_is_model_a=True) == pytest.approx(0.5)


def test_select_top_ppe_models_is_deterministic_under_ties() -> None:
    rows = pd.DataFrame(
        {
            "model_a": ["beta", "alpha", "delta", "gamma"],
            "model_b": ["alpha", "beta", "gamma", "delta"],
        }
    )
    assert select_top_ppe_models(rows, top_k=4) == ["alpha", "beta", "delta", "gamma"]


def test_score_cache_roundtrip_preserves_margins_and_metadata(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    rows = build_fake_rewardbench_rows().head(5).copy()
    scores = build_fake_rewardbench_scores("pairrm", rows)
    write_track_rows_cache(cache_dir, "rewardbench_v1", rows)
    write_score_cache(cache_dir, "rewardbench_v1", "pairrm", scores)

    loaded_rows = load_track_rows_cache(cache_dir, "rewardbench_v1").sort_values("unit_id").reset_index(drop=True)
    loaded_scores = load_score_cache(cache_dir, "rewardbench_v1", "pairrm").sort_values("unit_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        loaded_rows[["unit_id", "subset", "y"]],
        rows.sort_values("unit_id").reset_index(drop=True)[["unit_id", "subset", "y"]],
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        loaded_scores[["unit_id", "margin", "score_source"]],
        scores.sort_values("unit_id").reset_index(drop=True)[["unit_id", "margin", "score_source"]],
        check_dtype=False,
    )


def test_configure_offline_hf_runtime_sets_offline_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    configure_offline_hf_runtime()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_resolve_local_hf_snapshot_path_uses_local_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
        assert repo_id == "Skywork/Skywork-Reward-V2-Qwen3-4B"
        assert local_files_only is True
        return str(snapshot_dir)

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    assert resolve_local_hf_snapshot_path("Skywork/Skywork-Reward-V2-Qwen3-4B") == str(snapshot_dir)


def test_resolve_local_hf_snapshot_path_raises_when_model_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
        raise FileNotFoundError(repo_id)

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    with pytest.raises(RuntimeError, match="not available in the local Hugging Face cache"):
        resolve_local_hf_snapshot_path("Skywork/Skywork-Reward-V2-Qwen3-4B")


def test_rewardbench_normalization_is_invariant_to_input_order() -> None:
    raw = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "prompt": ["p1", "p2", "p3", "p4"],
            "chosen": ["ca", "cb", "cc", "cd"],
            "rejected": ["ra", "rb", "rc", "rd"],
            "chosen_model": ["cm"] * 4,
            "rejected_model": ["rm"] * 4,
            "subset": ["s1", "s2", "s3", "s4"],
        }
    )
    normalized_a = normalize_rewardbench_rows(raw)
    normalized_b = normalize_rewardbench_rows(raw.sample(frac=1.0, random_state=0))
    pd.testing.assert_frame_equal(
        normalized_a.sort_values("unit_id").reset_index(drop=True),
        normalized_b.sort_values("unit_id").reset_index(drop=True),
        check_dtype=False,
    )


def test_rewardbench_duplicate_ids_are_stably_disambiguated() -> None:
    raw = pd.DataFrame(
        {
            "id": ["dup", "dup", "unique"],
            "prompt": ["p1", "p2", "p3"],
            "chosen": ["ca1", "ca2", "ca3"],
            "rejected": ["ra1", "ra2", "ra3"],
            "chosen_model": ["cm1", "cm2", "cm3"],
            "rejected_model": ["rm1", "rm2", "rm3"],
            "subset": ["s1", "s2", "s3"],
        }
    )
    normalized_a = normalize_rewardbench_rows(raw)
    normalized_b = normalize_rewardbench_rows(raw.sample(frac=1.0, random_state=1))
    assert normalized_a["unit_id"].is_unique
    pd.testing.assert_frame_equal(
        normalized_a.sort_values("unit_id").reset_index(drop=True),
        normalized_b.sort_values("unit_id").reset_index(drop=True),
        check_dtype=False,
    )


def test_shared_split_schedule_is_evaluator_invariant() -> None:
    schedule_a = build_split_schedule(
        num_rows=80,
        budgets=[25, 50],
        replications=3,
        seed=4,
        schedule_key="ppe_human|ppe_human_v1|alpha",
    )
    schedule_b = build_split_schedule(
        num_rows=80,
        budgets=[25, 50],
        replications=3,
        seed=4,
        schedule_key="ppe_human|ppe_human_v1|alpha",
    )
    assert schedule_a.keys() == schedule_b.keys()
    for key in schedule_a:
        labeled_a, unlabeled_a = schedule_a[key]
        labeled_b, unlabeled_b = schedule_b[key]
        assert np.array_equal(labeled_a, labeled_b)
        assert np.array_equal(unlabeled_a, unlabeled_b)


def test_ppe_correctness_record_helpers_handle_scalar_and_list_scores() -> None:
    records = [
        {
            "question_id": "q1",
            "question": "prompt",
            "model_name": "generator-a",
            "parsed_outputs": ["A", "B", "C"],
            "scores": [True, False, True],
            "sampled_conflict_pairs": [[0, 1], [1, 2]],
            "score_1": 1.5,
            "score_2": [0.25],
            "score_3": -0.75,
            "category": "math",
        }
    ]
    rows = build_ppe_correctness_rows_from_result_records("mmlu_pro_best_of_k", records)
    scores = build_ppe_correctness_score_frame_from_records(
        benchmark="mmlu_pro_best_of_k",
        evaluator="starling_rm_7b_alpha",
        records=records,
    )
    assert list(rows["unit_id"]) == [
        build_ppe_correctness_unit_id("mmlu_pro_best_of_k", "generator-a", "q1", 0, 1),
        build_ppe_correctness_unit_id("mmlu_pro_best_of_k", "generator-a", "q1", 1, 2),
    ]
    assert rows["y"].tolist() == [1.0, 0.0]
    assert scores["margin"].tolist() == pytest.approx([1.25, 1.0])


def test_ppe_correctness_exact_duplicate_pairs_are_deduplicated() -> None:
    records = [
        {
            "question_id": "q1",
            "question": "prompt",
            "model_name": "generator-a",
            "parsed_outputs": ["A", "B", "C"],
            "scores": [True, False, True],
            "sampled_conflict_pairs": [[0, 1], [0, 1], [1, 2]],
            "score_1": 1.5,
            "score_2": 0.25,
            "score_3": -0.75,
            "category": "math",
        }
    ]
    rows = build_ppe_correctness_rows_from_result_records("mmlu_pro_best_of_k", records)
    scores = build_ppe_correctness_score_frame_from_records(
        benchmark="mmlu_pro_best_of_k",
        evaluator="starling_rm_7b_alpha",
        records=records,
    )
    assert rows["unit_id"].tolist() == [
        build_ppe_correctness_unit_id("mmlu_pro_best_of_k", "generator-a", "q1", 0, 1),
        build_ppe_correctness_unit_id("mmlu_pro_best_of_k", "generator-a", "q1", 1, 2),
    ]
    assert scores["unit_id"].tolist() == rows["unit_id"].tolist()


def test_ppe_correctness_reversed_duplicate_pairs_raise() -> None:
    records = [
        {
            "question_id": "q1",
            "question": "prompt",
            "model_name": "generator-a",
            "parsed_outputs": ["A", "B"],
            "scores": [True, False],
            "sampled_conflict_pairs": [[0, 1], [1, 0]],
            "score_1": 1.5,
            "score_2": 0.25,
            "category": "math",
        }
    ]
    with pytest.raises(ValueError, match="reversed conflict pairs"):
        build_ppe_correctness_rows_from_result_records("mmlu_pro_best_of_k", records)

    with pytest.raises(ValueError, match="reversed conflict pairs"):
        build_ppe_correctness_score_frame_from_records(
            benchmark="mmlu_pro_best_of_k",
            evaluator="starling_rm_7b_alpha",
            records=records,
        )


def test_ppe_correctness_panel_uses_pairwise_margin_orientation() -> None:
    rows = build_fake_ppe_correctness_rows().head(3).copy()
    scores = build_fake_ppe_correctness_scores("athene_rm_8b", rows)
    panel = build_ppe_correctness_panel(rows, scores)
    expected = scores.set_index("unit_id")["margin"]
    observed = panel.set_index("unit_id")["raw_score"]
    assert observed.to_dict() == pytest.approx(expected.to_dict())


def test_run_target_experiment_returns_finite_estimates_on_binary_track() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=90).astype(float)
    raw_score = y + rng.normal(scale=0.3, size=90)
    rows = run_target_experiment(
        track="rewardbench_v1",
        benchmark=REWARDBENCH_BENCHMARK,
        evaluator="pairrm",
        target_model="__overall__",
        y_total=y,
        score_total=raw_score,
        budgets=[25],
        replications=2,
        alpha=0.1,
        seed=0,
        selected_estimators=ALL_ESTIMATORS,
    )
    result_df = pd.DataFrame(rows)
    assert not result_df.empty
    assert result_df["estimate"].notna().all()
    assert np.isfinite(result_df["estimate"]).all()
    assert (result_df["ci_upper"] >= result_df["ci_lower"]).all()


def test_constant_scores_match_labeled_only_for_stable_calibrators() -> None:
    y = np.tile(np.array([0.0, 1.0], dtype=float), 40)
    raw_score = np.full(y.shape[0], 0.2, dtype=float)
    rows = run_target_experiment(
        track="rewardbench_v1",
        benchmark=REWARDBENCH_BENCHMARK,
        evaluator="pairrm",
        target_model="__overall__",
        y_total=y,
        score_total=raw_score,
        budgets=[25],
        replications=1,
        alpha=0.1,
        seed=1,
        selected_estimators=("classical", "linear_calibration", "isotonic_calibration_min10"),
    )
    frame = pd.DataFrame(rows)
    baseline = frame[frame["estimator"] == "classical"]["estimate"].iloc[0]
    assert frame[frame["estimator"] == "linear_calibration"]["estimate"].iloc[0] == pytest.approx(baseline)
    assert frame[frame["estimator"] == "isotonic_calibration_min10"]["estimate"].iloc[0] == pytest.approx(baseline)


def test_binary_only_calibrators_are_skipped_on_ppe_and_enabled_on_rewardbench() -> None:
    ppe_y = np.tile(np.array([0.0, 0.5, 1.0], dtype=float), 30)
    ppe_score = np.linspace(-1.0, 1.0, ppe_y.shape[0])
    ppe_rows = pd.DataFrame(
        run_target_experiment(
            track="ppe_human",
            benchmark=PPE_HUMAN_BENCHMARK,
            evaluator="pairrm",
            target_model="alpha",
            y_total=ppe_y,
            score_total=ppe_score,
            budgets=[25],
            replications=1,
            alpha=0.1,
            seed=2,
            selected_estimators=("classical", "platt_calibration"),
        )
    )
    assert set(ppe_rows["estimator"]) == {"classical"}

    rb_y = np.tile(np.array([0.0, 1.0], dtype=float), 45)
    rb_score = np.linspace(-1.0, 1.0, rb_y.shape[0])
    rb_rows = pd.DataFrame(
        run_target_experiment(
            track="rewardbench_v1",
            benchmark=REWARDBENCH_BENCHMARK,
            evaluator="pairrm",
            target_model="__overall__",
            y_total=rb_y,
            score_total=rb_score,
            budgets=[25],
            replications=1,
            alpha=0.1,
            seed=3,
            selected_estimators=("classical", "platt_calibration"),
        )
    )
    assert {"classical", "platt_calibration"}.issubset(set(rb_rows["estimator"]))


def test_llm_eval_smoke_ppe_runs_from_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out_ppe"
    populate_fake_caches(cache_dir)

    main(
        [
            "--smoke",
            "--tracks",
            "ppe_human",
            "--evaluators",
            "skywork_reward_llama3_1_8b",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
        ]
    )

    assert (output_dir / "llm_eval_raw_results.csv").exists()
    assert (output_dir / "llm_eval_summary_main.csv").exists()
    assert (output_dir / "fig_llm_eval_main.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_evaluator.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_benchmark.pdf").exists()
    assert (output_dir / "fig_llm_ppe_ranking.pdf").exists()
    assert (output_dir / "table_llm_summary.tex").exists()
    assert (output_dir / "table_llm_ppe_ranking.tex").exists()


def test_llm_eval_smoke_rewardbench_runs_from_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out_rewardbench"
    populate_fake_caches(cache_dir)

    main(
        [
            "--smoke",
            "--tracks",
            "rewardbench_v1",
            "--evaluators",
            "skywork_qwen3_4b",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "11",
        ]
    )

    assert (output_dir / "llm_eval_raw_results.csv").exists()
    assert (output_dir / "llm_eval_summary_main.csv").exists()
    assert (output_dir / "fig_llm_eval_main.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_evaluator.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_benchmark.pdf").exists()
    assert (output_dir / "fig_llm_ppe_ranking.pdf").exists()
    assert (output_dir / "table_llm_summary.tex").exists()
    assert (output_dir / "table_llm_ppe_ranking.tex").exists()


def test_llm_eval_smoke_ppe_correctness_runs_from_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out_ppe_correctness"
    populate_fake_caches(cache_dir)

    main(
        [
            "--smoke",
            "--tracks",
            "ppe_correctness",
            "--evaluators",
            "athene_rm_8b",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "13",
        ]
    )

    assert (output_dir / "llm_eval_raw_results.csv").exists()
    assert (output_dir / "llm_eval_summary_by_benchmark.csv").exists()
    assert (output_dir / "fig_llm_eval_main.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_evaluator.pdf").exists()
    assert (output_dir / "fig_llm_eval_by_benchmark.pdf").exists()
    assert (output_dir / "table_llm_ppe_ranking.tex").exists()


def test_ppe_correctness_duplicate_caches_rebuild_in_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    populate_fake_caches(cache_dir)
    good_rows = build_fake_ppe_correctness_rows()
    good_scores = build_fake_ppe_correctness_scores("athene_rm_8b", good_rows)

    bad_rows = good_rows.copy()
    bad_rows = pd.concat([bad_rows, bad_rows.iloc[[0]]], ignore_index=True)
    write_frame_jsonl_gz(row_cache_path(cache_dir, "ppe_correctness"), bad_rows)

    bad_scores = pd.concat([good_scores, good_scores.iloc[[0]]], ignore_index=True)
    write_frame_jsonl_gz(score_cache_path(cache_dir, "ppe_correctness", "athene_rm_8b"), bad_scores)

    def fake_row_rebuild(local_cache_dir: Path) -> pd.DataFrame:
        (local_cache_dir / "ppe_correctness_dedup_audit.json").write_text(
            json.dumps({"mmlu_pro_best_of_k": {"records_scanned": 1}}),
            encoding="utf-8",
        )
        return good_rows.copy()

    def fake_score_rebuild(*, rows: pd.DataFrame, cache_dir: Path, track: str, evaluator: str) -> pd.DataFrame:
        assert track == "ppe_correctness"
        repaired = good_scores[good_scores["unit_id"].isin(rows["unit_id"].astype(str))].reset_index(drop=True)
        write_score_cache(cache_dir, track, evaluator, repaired)
        return repaired

    monkeypatch.setattr("experiments.llm_eval_benchmark.build_ppe_correctness_rows_from_archive", fake_row_rebuild)
    monkeypatch.setattr("experiments.llm_eval_models.bootstrap_ppe_family_score_cache", fake_score_rebuild)

    main(
        [
            "--smoke",
            "--tracks",
            "ppe_correctness",
            "--evaluators",
            "athene_rm_8b",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
        ]
    )

    repaired_rows = load_track_rows_cache(cache_dir, "ppe_correctness")
    repaired_scores = load_score_cache(cache_dir, "ppe_correctness", "athene_rm_8b")
    assert repaired_rows["unit_id"].is_unique
    assert repaired_scores["unit_id"].is_unique
    assert set(repaired_scores["unit_id"]) == set(repaired_rows["unit_id"])
    assert (cache_dir / "ppe_correctness_dedup_audit.json").exists()


def test_llm_eval_resume_skips_completed_stages_and_rebuilds_root_outputs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out_resume"
    populate_fake_caches(cache_dir)

    args = [
        "--smoke",
        "--tracks",
        "ppe_human",
        "--evaluators",
        "skywork_reward_llama3_1_8b",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        "19",
    ]
    main(args)

    stage_dirs = sorted((output_dir / "stages").glob("*"))
    assert len(stage_dirs) >= 2
    (output_dir / "llm_eval_summary_main.csv").unlink()

    main(args + ["--resume"])

    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["complete"] is True
    assert status["skipped_stage_count"] == status["expected_stage_count"]
    assert (output_dir / "llm_eval_summary_main.csv").exists()


def test_partial_resume_rebuilds_outputs_before_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out_partial"
    populate_fake_caches(cache_dir)

    args = [
        "--smoke",
        "--tracks",
        "ppe_human",
        "--evaluators",
        "skywork_reward_llama3_1_8b",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        "23",
    ]
    main(args)

    stage_dirs = sorted((output_dir / "stages").glob("*"))
    assert len(stage_dirs) >= 2
    shutil_target = stage_dirs[-1]
    for path in (
        output_dir / "llm_eval_raw_results.csv",
        output_dir / "llm_eval_summary_main.csv",
        output_dir / "fig_llm_eval_main.pdf",
    ):
        if path.exists():
            path.unlink()
    for file_path in shutil_target.glob("*"):
        file_path.unlink()
    shutil_target.rmdir()

    def fail_run_target_experiment(*args, **kwargs):
        raise RuntimeError("forced stage failure")

    monkeypatch.setattr("experiments.llm_eval_benchmark.run_target_experiment", fail_run_target_experiment)

    with pytest.raises(RuntimeError, match="forced stage failure"):
        main(args + ["--resume"])

    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["failed_stage"]["error"] == "forced stage failure"
    assert (output_dir / "llm_eval_raw_results.csv").exists()
    assert (output_dir / "llm_eval_summary_main.csv").exists()
