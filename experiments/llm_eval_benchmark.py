from __future__ import annotations

import argparse
from collections import Counter
from hashlib import md5
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.llm_eval_models import (
    EVALUATOR_SPECS,
    PPE_CORRECTNESS_BENCHMARKS,
    PPE_HUMAN_BENCHMARK,
    REWARDBENCH_BENCHMARK,
    build_ppe_correctness_score_frame_from_records,
    build_ppe_correctness_unit_id,
    empty_ppe_correctness_audit,
    get_evaluator_spec,
    iter_deduped_ppe_correctness_pairs,
    load_track_rows_cache,
    load_ppe_result_json,
    merge_ppe_correctness_audit,
    resolve_score_frame,
    row_cache_path,
    write_track_rows_cache,
)
from experiments.ppi_mean_reproduction import (
    ESTIMATOR_COLORS,
    ESTIMATOR_LABELS,
    plot_draw_order,
    plot_line_alpha,
    plot_line_width,
    plot_line_zorder,
    run_aipw_em_estimator,
    run_aipw_estimator,
    run_auto_calibration_estimator,
    run_classical_estimator,
    run_isotonic_min10_calibration_estimator,
    run_linear_calibration_estimator,
    run_monotone_spline_estimator,
    run_platt_calibration_estimator,
    run_ppi_estimator,
    run_ppi_plus_plus_estimator,
)


TRACK_ORDER = ["ppe_human", "ppe_correctness", "rewardbench_v1"]
TRACK_LABELS = {
    "ppe_human": "PPE Human Preference",
    "ppe_correctness": "PPE Correctness",
    "rewardbench_v1": "RewardBench v1",
}
BENCHMARK_LABELS = {
    PPE_HUMAN_BENCHMARK: "PPE Human Preference",
    "mmlu_pro_best_of_k": "PPE MMLU-Pro",
    "math_best_of_k": "PPE MATH",
    "gpqa_best_of_k": "PPE GPQA",
    "ifeval_best_of_k": "PPE IFEval",
    "mbpp_plus_best_of_k": "PPE MBPP+",
    REWARDBENCH_BENCHMARK: "RewardBench v1",
}

DEFAULT_TRACKS = tuple(TRACK_ORDER)
DEFAULT_EVALUATORS = (
    "skywork_reward_llama3_1_8b",
    "armorm_llama3_8b_v0_1",
    "athene_rm_8b",
    "starling_rm_7b_alpha",
    "skywork_qwen3_4b",
)
DEFAULT_ALPHA = 0.1
DEFAULT_REPLICATIONS = 200
DEFAULT_BUDGETS = (25, 50, 100, 200, 400)
SMOKE_BUDGETS = (25, 50)
SMOKE_PPE_TARGETS = 2
PPE_TARGET_MODELS = 8
SMOKE_PPE_ROWS_PER_TARGET = 256
SMOKE_REWARDBENCH_ROWS = 512
SMOKE_PPE_CORRECTNESS_ROWS = 256

MAIN_TEXT_ESTIMATORS = (
    "classical",
    "aipw",
    "ppi",
    "ppi_plus_plus",
    "aipw_em",
    "auto_calibration",
    "linear_calibration",
)
APPENDIX_ESTIMATORS = (
    "monotone_spline",
    "isotonic_calibration_min10",
    "platt_calibration",
)
ALL_ESTIMATORS = MAIN_TEXT_ESTIMATORS + APPENDIX_ESTIMATORS
STAGES_DIRNAME = "stages"
RUN_STATUS_FILENAME = "run_status.json"
PPE_CORRECTNESS_AUDIT_FILENAME = "ppe_correctness_dedup_audit.json"
ROOT_OUTPUT_FILENAMES = (
    "llm_eval_raw_results.csv",
    "llm_eval_summary_by_target.csv",
    "llm_eval_summary_by_benchmark.csv",
    "llm_eval_summary_by_evaluator.csv",
    "llm_eval_summary_main.csv",
    "llm_eval_ppe_ranking_raw.csv",
    "llm_eval_ppe_ranking_summary.csv",
    "llm_eval_ppe_subgroup_summary.csv",
    "fig_llm_eval_main.pdf",
    "fig_llm_eval_by_evaluator.pdf",
    "fig_llm_eval_by_benchmark.pdf",
    "fig_llm_ppe_ranking.pdf",
    "table_llm_summary.csv",
    "table_llm_summary.tex",
    "table_llm_ppe_ranking.csv",
    "table_llm_ppe_ranking.tex",
)


def stable_seed_from_text(text: str) -> int:
    digest = md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stage_key(*, track: str, evaluator: str, benchmark: str, target_model: str) -> str:
    return f"{track}|{evaluator}|{benchmark}|{target_model}"


def stage_hash(*, track: str, evaluator: str, benchmark: str, target_model: str) -> str:
    return md5(stage_key(track=track, evaluator=evaluator, benchmark=benchmark, target_model=target_model).encode("utf-8")).hexdigest()[:12]


def stage_dir(
    output_dir: Path,
    *,
    track: str,
    evaluator: str,
    benchmark: str,
    target_model: str,
) -> Path:
    return output_dir / STAGES_DIRNAME / stage_hash(
        track=track,
        evaluator=evaluator,
        benchmark=benchmark,
        target_model=target_model,
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def stage_files_exist(stage_path: Path) -> bool:
    return (stage_path / "raw_results.csv").exists() and (stage_path / "stage_meta.json").exists()


def clear_llm_root_outputs(output_dir: Path) -> None:
    for filename in ROOT_OUTPUT_FILENAMES + (RUN_STATUS_FILENAME,):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _load_hf_dataset(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only in optional runtime
        raise RuntimeError(
            "Loading official benchmark data requires the optional `datasets` dependency. "
            "Install the LLM benchmark extras or pre-populate the row caches."
        ) from exc
    return load_dataset(dataset_name, split=split)


def stable_binary_assignment(unit_ids: pd.Series, *, salt: str) -> np.ndarray:
    return np.asarray(
        [
            bool(stable_seed_from_text(f"{salt}|{unit_id}") % 2)
            for unit_id in unit_ids.astype(str).tolist()
        ],
        dtype=bool,
    )


def select_representative_n_values(values: Sequence[object]) -> list[int]:
    unique_values = sorted({int(value) for value in values})
    if not unique_values:
        return []
    chosen = [
        unique_values[0],
        unique_values[len(unique_values) // 2],
        unique_values[-1],
    ]
    return list(dict.fromkeys(chosen))


def build_rewardbench_unit_ids(frame: pd.DataFrame) -> pd.Series:
    base_ids = (
        frame["id"].astype(str)
        if "id" in frame.columns
        else pd.Series(frame.index.astype(str), index=frame.index)
    )
    content_columns = ("prompt", "chosen", "rejected", "chosen_model", "rejected_model", "subset")
    content_frame = pd.DataFrame(
        {
            column: _series_or_default(frame, column, "").astype(str)
            for column in content_columns
        },
        index=frame.index,
    )
    signatures = pd.Series(
        [
            md5(
                json.dumps(
                    {column: row[idx] for idx, column in enumerate(content_columns)},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12]
            for row in content_frame.itertuples(index=False, name=None)
        ],
        index=frame.index,
    )
    disambiguated = base_ids.copy()
    duplicated_base = disambiguated.duplicated(keep=False)
    disambiguated.loc[duplicated_base] = (
        disambiguated.loc[duplicated_base]
        + "::"
        + signatures.loc[duplicated_base]
    )
    if disambiguated.duplicated(keep=False).any():
        ordered = (
            pd.DataFrame(
                {
                    "unit_id": disambiguated,
                    "_orig_pos": np.arange(len(disambiguated)),
                    **{column: content_frame[column] for column in content_columns},
                }
            )
            .sort_values(["unit_id", *content_columns, "_orig_pos"], kind="mergesort")
        )
        ordered["_occurrence"] = ordered.groupby("unit_id").cumcount()
        occurrence = (
            ordered.sort_values("_orig_pos")["_occurrence"]
            .astype(int)
            .to_numpy()
        )
        duplicated_unit_ids = disambiguated.duplicated(keep=False).to_numpy()
        disambiguated = pd.Series(
            np.where(
                duplicated_unit_ids,
                disambiguated.astype(str) + "::dup" + occurrence.astype(str),
                disambiguated.astype(str),
            ),
            index=frame.index,
        )
    return disambiguated.astype(str)


def normalize_rewardbench_rows(frame: pd.DataFrame) -> pd.DataFrame:
    unit_ids = build_rewardbench_unit_ids(frame)
    swap_order = stable_binary_assignment(unit_ids, salt="rewardbench_order")
    chosen = _series_or_default(frame, "chosen", "").astype(str)
    rejected = _series_or_default(frame, "rejected", "").astype(str)
    chosen_model = _series_or_default(frame, "chosen_model", "").astype(str)
    rejected_model = _series_or_default(frame, "rejected_model", "").astype(str)
    normalized = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "track": "rewardbench_v1",
            "benchmark": REWARDBENCH_BENCHMARK,
            "prompt": _series_or_default(frame, "prompt", "").astype(str),
            "response_a": np.where(swap_order, rejected, chosen),
            "response_b": np.where(swap_order, chosen, rejected),
            "model_a": np.where(swap_order, rejected_model, chosen_model),
            "model_b": np.where(swap_order, chosen_model, rejected_model),
            "subset": _series_or_default(frame, "subset", "unknown").astype(str),
            "y": np.where(swap_order, 0.0, 1.0),
        }
    )
    return normalized.sort_values("unit_id").reset_index(drop=True)


def build_ppe_correctness_rows_from_result_records(
    benchmark: str,
    records: list[dict[str, object]],
    *,
    audit_summary: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    local_audit = empty_ppe_correctness_audit()
    for record in records:
        question_id = str(record["question_id"])
        target_model = str(record["model_name"])
        prompt = str(record.get("question", ""))
        parsed_outputs = record.get("parsed_outputs", []) or []
        correctness = record.get("scores", []) or []
        category = str(record.get("category", "unknown"))
        deduped_pairs, pair_audit = iter_deduped_ppe_correctness_pairs(
            benchmark=benchmark,
            record=record,
        )
        local_audit = merge_ppe_correctness_audit(local_audit, pair_audit)
        for pair_a, pair_b in deduped_pairs:
            if pair_a >= len(correctness) or pair_b >= len(correctness):
                continue
            first_correct = bool(correctness[pair_a])
            second_correct = bool(correctness[pair_b])
            if first_correct == second_correct:
                continue
            rows.append(
                {
                    "unit_id": build_ppe_correctness_unit_id(
                        benchmark,
                        target_model,
                        question_id,
                        pair_a,
                        pair_b,
                    ),
                    "track": "ppe_correctness",
                    "benchmark": benchmark,
                    "prompt": prompt,
                    "response_a": str(parsed_outputs[pair_a]) if pair_a < len(parsed_outputs) else "",
                    "response_b": str(parsed_outputs[pair_b]) if pair_b < len(parsed_outputs) else "",
                    "model_a": target_model,
                    "model_b": target_model,
                    "target_model": target_model,
                    "subset": category,
                    "y": 1.0 if first_correct and not second_correct else 0.0,
                }
            )
    if audit_summary is not None:
        audit_summary.update(merge_ppe_correctness_audit(audit_summary, local_audit))
    return pd.DataFrame.from_records(rows)


def build_split_schedule(
    *,
    num_rows: int,
    budgets: Sequence[int],
    replications: int,
    seed: int,
    schedule_key: str,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    if num_rows <= 0:
        return {}
    rng = np.random.default_rng(seed + stable_seed_from_text(schedule_key))
    schedule: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for n_labeled in budgets:
        if n_labeled >= num_rows:
            continue
        for replicate in range(replications):
            permutation = rng.permutation(num_rows)
            schedule[(int(n_labeled), int(replicate))] = (
                permutation[:n_labeled],
                permutation[n_labeled:],
            )
    return schedule


def build_ppe_rows_from_hf() -> pd.DataFrame:
    dataset = _load_hf_dataset("lmarena-ai/PPE-Human-Preference-V1", split="test")
    frame = dataset.to_pandas()
    language = _series_or_default(frame, "language", "unknown").astype(str)
    normalized = pd.DataFrame(
        {
            "unit_id": (
                frame["question_id"].astype(str)
                if "question_id" in frame.columns
                else pd.Series(frame.index.astype(str), index=frame.index)
            ),
            "track": "ppe_human",
            "benchmark": PPE_HUMAN_BENCHMARK,
            "prompt": _series_or_default(frame, "prompt", "").astype(str),
            "response_a": _series_or_default(frame, "response_1", "").astype(str),
            "response_b": _series_or_default(frame, "response_2", "").astype(str),
            "model_a": _series_or_default(frame, "model_a", "").astype(str),
            "model_b": _series_or_default(frame, "model_b", "").astype(str),
            "winner": _series_or_default(frame, "winner", "tie").astype(str),
            "language": language,
            "language_group": np.where(language.str.lower() == "english", "english", "non_english"),
            "hard_prompt": _series_or_default(frame, "hard_prompt", False).astype(bool),
            "math_prompt": _series_or_default(frame, "math_prompt", False).astype(bool),
            "is_code": _series_or_default(frame, "is_code", False).astype(bool),
        }
    )
    normalized = normalized.sort_values("unit_id").reset_index(drop=True)
    return normalized


def build_rewardbench_rows_from_hf(split: str = "filtered") -> pd.DataFrame:
    dataset = _load_hf_dataset("allenai/reward-bench", split=split)
    return normalize_rewardbench_rows(dataset.to_pandas())


def build_ppe_correctness_rows_from_archive(cache_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    audit_payload: dict[str, dict[str, int]] = {}
    for benchmark in PPE_CORRECTNESS_BENCHMARKS:
        records = load_ppe_result_json(
            cache_dir,
            f"data/{benchmark}/Skywork-Reward-Llama-3.1-8B.json",
        )
        benchmark_audit = empty_ppe_correctness_audit()
        frames.append(
            build_ppe_correctness_rows_from_result_records(
                benchmark,
                records,
                audit_summary=benchmark_audit,
            )
        )
        audit_payload[benchmark] = benchmark_audit
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    atomic_write_json(cache_dir / PPE_CORRECTNESS_AUDIT_FILENAME, audit_payload)
    if combined.empty:
        return combined
    return combined.sort_values(["benchmark", "target_model", "unit_id"]).reset_index(drop=True)


def ensure_track_rows(track: str, cache_dir: Path, rewardbench_split: str = "filtered") -> pd.DataFrame:
    cache_file = row_cache_path(cache_dir, track)
    if cache_file.exists():
        try:
            frame = load_track_rows_cache(cache_dir, track)
        except ValueError:
            if track != "ppe_correctness":
                raise
            frame = build_ppe_correctness_rows_from_archive(cache_dir)
            write_track_rows_cache(cache_dir, track, frame)
    elif track == "ppe_human":
        frame = build_ppe_rows_from_hf()
        write_track_rows_cache(cache_dir, track, frame)
    elif track == "ppe_correctness":
        frame = build_ppe_correctness_rows_from_archive(cache_dir)
        write_track_rows_cache(cache_dir, track, frame)
    elif track == "rewardbench_v1":
        frame = build_rewardbench_rows_from_hf(split=rewardbench_split)
        write_track_rows_cache(cache_dir, track, frame)
    else:  # pragma: no cover - guarded by CLI validation
        raise ValueError(f"Unknown track '{track}'.")
    if frame.empty:
        raise ValueError(f"Track '{track}' has no rows available.")
    return frame


def select_top_ppe_models(rows: pd.DataFrame, top_k: int = PPE_TARGET_MODELS) -> list[str]:
    counts: Counter[str] = Counter()
    for model in rows["model_a"].astype(str):
        counts[model] += 1
    for model in rows["model_b"].astype(str):
        counts[model] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [model for model, _ in ordered[:top_k]]


def ppe_target_outcome(winner: object, *, target_is_model_a: bool) -> float:
    winner_norm = str(winner).strip().lower()
    if winner_norm == "model_a":
        return 1.0 if target_is_model_a else 0.0
    if winner_norm == "model_b":
        return 0.0 if target_is_model_a else 1.0
    return 0.5


def orient_ppe_rows_for_target(rows: pd.DataFrame, target_model: str) -> pd.DataFrame:
    involved = rows[(rows["model_a"] == target_model) | (rows["model_b"] == target_model)].copy()
    if involved.empty:
        return pd.DataFrame()
    target_is_a = involved["model_a"] == target_model
    involved["target_model"] = target_model
    involved["y"] = [
        ppe_target_outcome(winner, target_is_model_a=is_a)
        for winner, is_a in zip(involved["winner"], target_is_a)
    ]
    return involved


def probability_score(raw_score: np.ndarray) -> np.ndarray:
    raw_score = np.asarray(raw_score, dtype=float)
    transformed = 1.0 / (1.0 + np.exp(-raw_score))
    return np.clip(transformed, 1e-6, 1.0 - 1e-6)


def merge_scores(rows: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        scores,
        on="unit_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Could not match any score rows to benchmark rows.")
    if merged.shape[0] != rows.shape[0]:
        missing = sorted(set(rows["unit_id"].astype(str)) - set(merged["unit_id"].astype(str)))
        preview = missing[:5]
        raise ValueError(
            "Scores did not cover the full benchmark row set. "
            f"Missing {len(missing)} rows; examples: {preview}"
        )
    return merged


def build_ppe_panel(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    target_models: Sequence[str],
    smoke_rows_per_target: Optional[int] = None,
) -> pd.DataFrame:
    merged = merge_scores(rows, scores)
    panels: list[pd.DataFrame] = []
    for target_model in target_models:
        oriented = orient_ppe_rows_for_target(merged, target_model)
        if oriented.empty:
            continue
        oriented = oriented.copy()
        target_is_a = oriented["model_a"] == target_model
        oriented["raw_score"] = np.where(target_is_a, oriented["margin"], -oriented["margin"])
        oriented["benchmark"] = PPE_HUMAN_BENCHMARK
        oriented["subset"] = "ppe_human"
        keep_columns = [
            "unit_id",
            "benchmark",
            "target_model",
            "prompt",
            "response_a",
            "response_b",
            "raw_score",
            "y",
            "score_source",
            "language_group",
            "hard_prompt",
            "math_prompt",
            "is_code",
            "subset",
        ]
        oriented = oriented.loc[:, keep_columns].sort_values("unit_id").reset_index(drop=True)
        if smoke_rows_per_target is not None:
            oriented = oriented.head(smoke_rows_per_target).reset_index(drop=True)
        panels.append(oriented)
    if not panels:
        raise ValueError("No PPE target panels could be constructed from the provided rows and scores.")
    return pd.concat(panels, ignore_index=True)


def build_ppe_correctness_panel(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    max_rows_per_group: Optional[int] = None,
) -> pd.DataFrame:
    merged = merge_scores(rows, scores).copy()
    merged["raw_score"] = merged["margin"].astype(float)
    keep_columns = [
        "unit_id",
        "benchmark",
        "target_model",
        "prompt",
        "response_a",
        "response_b",
        "raw_score",
        "y",
        "score_source",
        "subset",
    ]
    panel = merged.loc[:, keep_columns].sort_values(["benchmark", "target_model", "unit_id"]).reset_index(drop=True)
    if max_rows_per_group is not None:
        panel = (
            panel.groupby(["benchmark", "target_model"], sort=False, group_keys=False)
            .head(max_rows_per_group)
            .reset_index(drop=True)
        )
    return panel


def build_rewardbench_panel(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    merged = merge_scores(rows, scores).copy()
    merged["benchmark"] = REWARDBENCH_BENCHMARK
    merged["target_model"] = "__overall__"
    merged["raw_score"] = merged["margin"].astype(float)
    keep_columns = [
        "unit_id",
        "benchmark",
        "target_model",
        "prompt",
        "response_a",
        "response_b",
        "raw_score",
        "y",
        "score_source",
        "subset",
    ]
    panel = merged.loc[:, keep_columns].sort_values("unit_id").reset_index(drop=True)
    if max_rows is not None:
        panel = panel.head(max_rows).reset_index(drop=True)
    return panel


def _estimator_runners(is_binary: bool) -> dict[str, Callable[..., dict[str, float]]]:
    runners: dict[str, Callable[..., dict[str, float]]] = {
        "classical": run_classical_estimator,
        "aipw": run_aipw_estimator,
        "ppi": run_ppi_estimator,
        "ppi_plus_plus": run_ppi_plus_plus_estimator,
        "aipw_em": run_aipw_em_estimator,
        "auto_calibration": run_auto_calibration_estimator,
        "monotone_spline": run_monotone_spline_estimator,
        "linear_calibration": run_linear_calibration_estimator,
        "isotonic_calibration_min10": run_isotonic_min10_calibration_estimator,
    }
    if is_binary:
        runners["platt_calibration"] = run_platt_calibration_estimator
    return runners


def summarize_result(
    *,
    track: str,
    benchmark: str,
    evaluator: str,
    target_model: str,
    alpha: float,
    n_labeled: int,
    n_unlabeled: int,
    replicate: int,
    estimator: str,
    true_theta: float,
    estimate: float,
    se: float,
    ci_lower: float,
    ci_upper: float,
) -> dict[str, float | str]:
    return {
        "track": track,
        "benchmark": benchmark,
        "evaluator": evaluator,
        "target_model": target_model,
        "alpha": float(alpha),
        "n_labeled": int(n_labeled),
        "n_unlabeled": int(n_unlabeled),
        "replicate": int(replicate),
        "estimator": estimator,
        "true_theta": float(true_theta),
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_upper - ci_lower),
        "bias": float(estimate - true_theta),
        "sq_error": float((estimate - true_theta) ** 2),
        "covered": float(ci_lower <= true_theta <= ci_upper),
    }


def run_target_experiment(
    *,
    track: str,
    benchmark: str,
    evaluator: str,
    target_model: str,
    y_total: np.ndarray,
    score_total: np.ndarray,
    budgets: Sequence[int],
    replications: int,
    alpha: float,
    seed: int,
    selected_estimators: Sequence[str],
    split_schedule: Optional[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = None,
) -> list[dict[str, float | str]]:
    y_total = np.asarray(y_total, dtype=float)
    score_total = np.asarray(score_total, dtype=float)
    if y_total.shape != score_total.shape:
        raise ValueError("Expected aligned outcome and score arrays.")
    if y_total.ndim != 1:
        raise ValueError("Expected one-dimensional outcome and score arrays.")
    true_theta = float(np.mean(y_total))
    is_binary = bool(np.all(np.isin(np.unique(y_total), [0.0, 1.0])))
    available_runners = _estimator_runners(is_binary=is_binary)
    requested = [name for name in selected_estimators if name in available_runners]
    if split_schedule is None:
        split_schedule = build_split_schedule(
            num_rows=y_total.shape[0],
            budgets=budgets,
            replications=replications,
            seed=seed,
            schedule_key=f"{track}|{benchmark}|{target_model}",
        )

    rows: list[dict[str, float | str]] = []
    for n_labeled in budgets:
        for replicate in range(replications):
            split = split_schedule.get((int(n_labeled), int(replicate)))
            if split is None:
                continue
            labeled_idx, unlabeled_idx = split
            y_l = y_total[labeled_idx]
            score_l = score_total[labeled_idx]
            score_u = score_total[unlabeled_idx]
            n_unlabeled = int(unlabeled_idx.shape[0])

            for estimator in requested:
                runner = available_runners[estimator]
                if estimator == "classical":
                    result = runner(y_l, alpha)
                elif estimator == "platt_calibration":
                    result = runner(y_l, probability_score(score_l), probability_score(score_u), alpha)
                else:
                    result = runner(y_l, score_l, score_u, alpha)
                rows.append(
                    summarize_result(
                        track=track,
                        benchmark=benchmark,
                        evaluator=evaluator,
                        target_model=target_model,
                        alpha=alpha,
                        n_labeled=n_labeled,
                        n_unlabeled=n_unlabeled,
                        replicate=replicate,
                        estimator=estimator,
                        true_theta=true_theta,
                        estimate=float(result["estimate"]),
                        se=float(result["se"]),
                        ci_lower=float(result["ci_lower"]),
                        ci_upper=float(result["ci_upper"]),
                    )
                )
    return rows


def summarize_raw_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    group_cols = [
        "track",
        "benchmark",
        "evaluator",
        "target_model",
        "alpha",
        "n_labeled",
        "n_unlabeled",
        "estimator",
        "true_theta",
    ]
    target_summary = (
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
        .sort_values(["track", "evaluator", "target_model", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )

    ppi_variance = (
        target_summary[target_summary["estimator"] == "ppi"][
            ["track", "benchmark", "evaluator", "target_model", "n_labeled", "emp_variance", "mse"]
        ]
        .rename(
            columns={
                "emp_variance": "emp_variance_ppi",
                "mse": "mse_ppi",
            }
        )
        .drop_duplicates()
    )
    classical_variance = (
        target_summary[target_summary["estimator"] == "classical"][
            ["track", "benchmark", "evaluator", "target_model", "n_labeled", "emp_variance"]
        ]
        .rename(columns={"emp_variance": "emp_variance_labeled_only"})
        .drop_duplicates()
    )
    target_summary = target_summary.merge(
        ppi_variance,
        on=["track", "benchmark", "evaluator", "target_model", "n_labeled"],
        how="left",
    )
    target_summary = target_summary.merge(
        classical_variance,
        on=["track", "benchmark", "evaluator", "target_model", "n_labeled"],
        how="left",
    )
    target_summary["rel_eff_vs_ppi"] = target_summary["emp_variance_ppi"] / target_summary["emp_variance"]
    target_summary["mse_ratio_vs_ppi"] = target_summary["mse"] / target_summary["mse_ppi"]
    target_summary["rel_eff_vs_labeled_only"] = (
        target_summary["emp_variance_labeled_only"] / target_summary["emp_variance"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        target_summary["label_savings"] = 1.0 - 1.0 / target_summary["rel_eff_vs_labeled_only"]
    target_summary["label_savings"] = np.clip(target_summary["label_savings"], a_min=0.0, a_max=None)

    benchmark_summary = (
        target_summary.groupby(
            ["track", "benchmark", "evaluator", "alpha", "n_labeled", "estimator"],
            as_index=False,
        )
        .agg(
            mean_estimate=("mean_estimate", "mean"),
            mean_bias=("mean_bias", "mean"),
            emp_variance=("emp_variance", "mean"),
            mse=("mse", "mean"),
            rmse=("rmse", "mean"),
            mean_se=("mean_se", "mean"),
            coverage=("coverage", "mean"),
            mean_ci_width=("mean_ci_width", "mean"),
            rel_eff_vs_ppi=("rel_eff_vs_ppi", "mean"),
            mse_ratio_vs_ppi=("mse_ratio_vs_ppi", "mean"),
            rel_eff_vs_labeled_only=("rel_eff_vs_labeled_only", "mean"),
            label_savings=("label_savings", "mean"),
        )
        .sort_values(["track", "benchmark", "evaluator", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )

    macro_summary = (
        benchmark_summary.groupby(["track", "evaluator", "alpha", "n_labeled", "estimator"], as_index=False)
        .agg(
            mean_estimate=("mean_estimate", "mean"),
            mean_bias=("mean_bias", "mean"),
            emp_variance=("emp_variance", "mean"),
            mse=("mse", "mean"),
            rmse=("rmse", "mean"),
            mean_se=("mean_se", "mean"),
            coverage=("coverage", "mean"),
            mean_ci_width=("mean_ci_width", "mean"),
            rel_eff_vs_ppi=("rel_eff_vs_ppi", "mean"),
            mse_ratio_vs_ppi=("mse_ratio_vs_ppi", "mean"),
            rel_eff_vs_labeled_only=("rel_eff_vs_labeled_only", "mean"),
            label_savings=("label_savings", "mean"),
        )
        .sort_values(["track", "evaluator", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )

    overall_summary = (
        macro_summary.groupby(["track", "alpha", "n_labeled", "estimator"], as_index=False)
        .agg(
            mean_estimate=("mean_estimate", "mean"),
            mean_bias=("mean_bias", "mean"),
            emp_variance=("emp_variance", "mean"),
            mse=("mse", "mean"),
            rmse=("rmse", "mean"),
            mean_se=("mean_se", "mean"),
            coverage=("coverage", "mean"),
            mean_ci_width=("mean_ci_width", "mean"),
            rel_eff_vs_ppi=("rel_eff_vs_ppi", "mean"),
            mse_ratio_vs_ppi=("mse_ratio_vs_ppi", "mean"),
            rel_eff_vs_labeled_only=("rel_eff_vs_labeled_only", "mean"),
            label_savings=("label_savings", "mean"),
        )
        .sort_values(["track", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )
    return target_summary, benchmark_summary, macro_summary, overall_summary


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Expected aligned arrays for Spearman correlation.")
    if x.size < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.allclose(x_rank, x_rank[0]) or np.allclose(y_rank, y_rank[0]):
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def top1_regret(estimated: np.ndarray, truth: np.ndarray) -> float:
    if estimated.shape != truth.shape:
        raise ValueError("Expected aligned arrays for top-1 regret.")
    if truth.size == 0:
        return float("nan")
    best_truth_idx = int(np.argmax(truth))
    selected_idx = int(np.argmax(estimated))
    return float(truth[best_truth_idx] - truth[selected_idx])


def summarize_ppe_rankings(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ppe = raw_df[raw_df["track"] == "ppe_human"].copy()
    if ppe.empty:
        empty = pd.DataFrame()
        return empty, empty

    records: list[dict[str, float | str]] = []
    grouped = ppe.groupby(["evaluator", "n_labeled", "replicate", "estimator"], sort=False)
    for (evaluator, n_labeled, replicate, estimator), frame in grouped:
        local = frame.sort_values("target_model").reset_index(drop=True)
        if local["target_model"].nunique() < 2:
            continue
        estimated = local["estimate"].to_numpy(dtype=float)
        truth = local["true_theta"].to_numpy(dtype=float)
        ranking = local[["target_model", "estimate", "true_theta"]].copy()
        top_true = ranking.sort_values(["true_theta", "target_model"], ascending=[False, True]).iloc[0]["target_model"]
        top_est = ranking.sort_values(["estimate", "target_model"], ascending=[False, True]).iloc[0]["target_model"]
        records.append(
            {
                "track": "ppe_human",
                "evaluator": evaluator,
                "n_labeled": int(n_labeled),
                "replicate": int(replicate),
                "estimator": estimator,
                "spearman": spearman_correlation(estimated, truth),
                "top1_accuracy": float(top_true == top_est),
                "top1_regret": top1_regret(estimated, truth),
            }
        )
    raw_ranking = pd.DataFrame.from_records(records)
    if raw_ranking.empty:
        empty = pd.DataFrame()
        return empty, empty
    summary = (
        raw_ranking.groupby(["evaluator", "n_labeled", "estimator"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            top1_regret=("top1_regret", "mean"),
        )
        .sort_values(["evaluator", "n_labeled", "estimator"])
        .reset_index(drop=True)
    )
    macro = (
        summary.groupby(["n_labeled", "estimator"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            top1_regret=("top1_regret", "mean"),
        )
        .sort_values(["n_labeled", "estimator"])
        .reset_index(drop=True)
    )
    return raw_ranking, macro


def build_ppe_subgroup_summary(
    rows: pd.DataFrame,
    *,
    target_models: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    subgroup_columns = {
        "hard_prompt": "hard_prompt",
        "math_prompt": "math_prompt",
        "is_code": "is_code",
        "language_group": "language_group",
    }
    for target_model in target_models:
        oriented = orient_ppe_rows_for_target(rows, target_model)
        if oriented.empty:
            continue
        for subgroup_name, column in subgroup_columns.items():
            if column not in oriented.columns:
                continue
            for subgroup_value, frame in oriented.groupby(column, sort=False):
                records.append(
                    {
                        "target_model": target_model,
                        "subgroup": subgroup_name,
                        "value": str(subgroup_value),
                        "n_rows": int(frame.shape[0]),
                        "true_theta": float(frame["y"].mean()),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_legend(order: Sequence[str], available: set[str]) -> tuple[list[Line2D], list[str]]:
    legend_estimators = [name for name in order if name in available]
    handles = [
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
    labels = [ESTIMATOR_LABELS[name] for name in legend_estimators]
    return handles, labels


def plot_llm_eval_main(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("mse_ratio_vs_ppi", "MSE / PPI MSE"),
        ("rel_eff_vs_ppi", "Rel. eff. vs PPI"),
        ("coverage", "Coverage"),
    ]
    active_tracks = [track for track in TRACK_ORDER if track in set(summary_df["track"].unique())]
    if not active_tracks:
        active_tracks = ["ppe_human", "ppe_correctness"]
    fig, axes = plt.subplots(len(active_tracks), len(metrics), figsize=(10.0, 5.7), sharex=True, squeeze=False)
    for row_idx, track in enumerate(active_tracks):
        frame = summary_df[summary_df["track"] == track].copy()
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            order = [name for name in MAIN_TEXT_ESTIMATORS if name in frame["estimator"].unique()]
            for estimator in plot_draw_order(order):
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
            else:
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            if row_idx == 0:
                ax.set_title(ylabel)
            if col_idx == 0:
                ax.set_ylabel(TRACK_LABELS[track])
            if row_idx == len(active_tracks) - 1:
                ax.set_xlabel("Labeled sample size n")
    handles, labels = build_legend(MAIN_TEXT_ESTIMATORS, set(summary_df["estimator"].unique()))
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_llm_eval_by_evaluator(summary_df: pd.DataFrame, output_path: Path) -> None:
    evaluators = list(summary_df["evaluator"].drop_duplicates()) or list(DEFAULT_EVALUATORS)
    active_tracks = [track for track in TRACK_ORDER if track in set(summary_df["track"].unique())]
    if not active_tracks:
        active_tracks = ["ppe_human", "ppe_correctness"]
    fig, axes = plt.subplots(
        len(active_tracks),
        len(evaluators),
        figsize=(4.5 * len(evaluators), 3.3 * len(active_tracks)),
        sharex=True,
        squeeze=False,
    )
    for row_idx, track in enumerate(active_tracks):
        for col_idx, evaluator in enumerate(evaluators):
            ax = axes[row_idx, col_idx]
            frame = summary_df[(summary_df["track"] == track) & (summary_df["evaluator"] == evaluator)].copy()
            order = [name for name in MAIN_TEXT_ESTIMATORS if name in frame["estimator"].unique()]
            for estimator in plot_draw_order(order):
                sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
                ax.plot(
                    sub["n_labeled"],
                    sub["mse_ratio_vs_ppi"],
                    marker="o",
                    linewidth=plot_line_width(estimator),
                    color=ESTIMATOR_COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                    alpha=plot_line_alpha(estimator),
                    zorder=plot_line_zorder(estimator),
                )
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            if row_idx == 0:
                ax.set_title(get_evaluator_spec(evaluator).display_name)
            if col_idx == 0:
                ax.set_ylabel(f"{TRACK_LABELS[track]}\nMSE / PPI MSE")
            if row_idx == len(active_tracks) - 1:
                ax.set_xlabel("Labeled sample size n")
    handles, labels = build_legend(MAIN_TEXT_ESTIMATORS, set(summary_df["estimator"].unique()))
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_llm_eval_by_benchmark(summary_df: pd.DataFrame, output_path: Path) -> None:
    available_benchmarks = set(summary_df["benchmark"].unique()) if not summary_df.empty else set()
    available_estimators = set(summary_df["estimator"].unique()) if not summary_df.empty else set()
    benchmark_order = [benchmark for benchmark in PPE_CORRECTNESS_BENCHMARKS if benchmark in available_benchmarks]
    if not benchmark_order:
        benchmark_order = list(PPE_CORRECTNESS_BENCHMARKS)
    ncols = min(3, len(benchmark_order))
    nrows = int(np.ceil(len(benchmark_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, benchmark in zip(axes, benchmark_order):
        frame = summary_df[summary_df["benchmark"] == benchmark].copy()
        order = [name for name in MAIN_TEXT_ESTIMATORS if name in frame["estimator"].unique()]
        for estimator in plot_draw_order(order):
            sub = frame[frame["estimator"] == estimator].sort_values("n_labeled")
            ax.plot(
                sub["n_labeled"],
                sub["mse_ratio_vs_ppi"],
                marker="o",
                linewidth=plot_line_width(estimator),
                color=ESTIMATOR_COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
                alpha=plot_line_alpha(estimator),
                zorder=plot_line_zorder(estimator),
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark))
        ax.set_xlabel("Labeled sample size n")
    for ax in axes[::ncols]:
        ax.set_ylabel("MSE / PPI MSE")
    for ax in axes[len(benchmark_order):]:
        ax.set_axis_off()
    handles, labels = build_legend(MAIN_TEXT_ESTIMATORS, available_estimators)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ppe_ranking(ranking_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True)
    if ranking_df.empty:
        for ax, title in zip(
            axes,
            ["Spearman rank correlation", "Top-1 identification", "Top-1 regret"],
        ):
            ax.text(0.5, 0.5, "PPE track not run", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    metrics = [
        ("spearman", "Spearman rank correlation"),
        ("top1_accuracy", "Top-1 identification rate"),
        ("top1_regret", "Top-1 regret"),
    ]
    order = [name for name in MAIN_TEXT_ESTIMATORS if name in ranking_df["estimator"].unique()]
    for ax, (metric, title) in zip(axes, metrics):
        for estimator in plot_draw_order(order):
            sub = ranking_df[ranking_df["estimator"] == estimator].sort_values("n_labeled")
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
        ax.set_title(title)
        ax.set_xlabel("Labeled sample size n")
        if metric == "top1_regret":
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Metric value")
    handles, labels = build_legend(MAIN_TEXT_ESTIMATORS, set(ranking_df["estimator"].unique()))
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_ppe_ranking_table(ranking_df: pd.DataFrame, output_dir: Path) -> None:
    if ranking_df.empty or "n_labeled" not in ranking_df.columns:
        representative_n: list[int] = []
    else:
        representative_n = select_representative_n_values(ranking_df["n_labeled"].tolist())
    if ranking_df.empty or not representative_n:
        table = pd.DataFrame(columns=["n_labeled", "estimator", "spearman", "top1_accuracy", "top1_regret"])
    else:
        available_estimators = set(ranking_df["estimator"].tolist())
        estimator_order = [name for name in MAIN_TEXT_ESTIMATORS if name in available_estimators]
        table = (
            ranking_df[
                ranking_df["n_labeled"].isin(representative_n)
                & ranking_df["estimator"].isin(estimator_order)
            ][["n_labeled", "estimator", "spearman", "top1_accuracy", "top1_regret"]]
            .sort_values(["n_labeled", "estimator"])
            .reset_index(drop=True)
        )
    atomic_write_csv(output_dir / "table_llm_ppe_ranking.csv", table)

    latex_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"$n$ & Estimator & Spearman & Top-1 & Top-1 regret \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{int(row['n_labeled'])} & "
            f"{latex_escape(ESTIMATOR_LABELS.get(row['estimator'], row['estimator']))} & "
            f"{row['spearman']:.3f} & "
            f"{row['top1_accuracy']:.3f} & "
            f"{row['top1_regret']:.3f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_write_text(output_dir / "table_llm_ppe_ranking.tex", "\n".join(latex_lines) + "\n")


def write_llm_summary_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    selected_rows: list[pd.DataFrame] = []
    for track, frame in summary_df.groupby("track", sort=False):
        n_values = select_representative_n_values(frame["n_labeled"].unique().tolist())
        if not n_values:
            continue
        selected_rows.append(frame[frame["n_labeled"].isin(n_values)].copy())
    if selected_rows:
        table = pd.concat(selected_rows, ignore_index=True)
    else:
        table = pd.DataFrame(columns=["track", "n_labeled", "estimator", "mse_ratio_vs_ppi", "rel_eff_vs_ppi", "coverage", "label_savings"])

    table = table[
        [
            "track",
            "n_labeled",
            "estimator",
            "mse_ratio_vs_ppi",
            "rel_eff_vs_ppi",
            "coverage",
            "label_savings",
        ]
    ].sort_values(["track", "n_labeled", "estimator"])
    atomic_write_csv(output_dir / "table_llm_summary.csv", table)

    latex_lines = [
        r"\begin{tabular}{lllrccc}",
        r"\toprule",
        r"Track & $n$ & Estimator & MSE / PPI & RelEff / PPI & Coverage & Label savings \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{latex_escape(TRACK_LABELS.get(row['track'], row['track']))} & "
            f"{int(row['n_labeled'])} & "
            f"{latex_escape(ESTIMATOR_LABELS.get(row['estimator'], row['estimator']))} & "
            f"{row['mse_ratio_vs_ppi']:.3f} & "
            f"{row['rel_eff_vs_ppi']:.3f} & "
            f"{row['coverage']:.3f} & "
            f"{row['label_savings']:.3f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_write_text(output_dir / "table_llm_summary.tex", "\n".join(latex_lines) + "\n")


def write_run_status(
    output_dir: Path,
    *,
    expected_stage_count: int,
    completed_stage_count: int,
    skipped_stage_count: int,
    complete: bool,
    failed_stage: Optional[dict[str, str]] = None,
) -> None:
    atomic_write_json(
        output_dir / RUN_STATUS_FILENAME,
        {
            "expected_stage_count": int(expected_stage_count),
            "completed_stage_count": int(completed_stage_count),
            "skipped_stage_count": int(skipped_stage_count),
            "complete": bool(complete),
            "failed_stage": failed_stage,
        },
    )


def aggregate_completed_outputs(
    *,
    output_dir: Path,
    completed_stage_frames: Sequence[pd.DataFrame],
    ppe_subgroup_summary: pd.DataFrame,
) -> None:
    if not completed_stage_frames:
        return

    raw_df = (
        pd.concat(list(completed_stage_frames), ignore_index=True)
        .sort_values(
            ["track", "benchmark", "evaluator", "target_model", "n_labeled", "replicate", "estimator"]
        )
        .reset_index(drop=True)
    )
    target_summary, benchmark_summary, macro_summary, overall_summary = summarize_raw_results(raw_df)
    ppe_ranking_raw, ppe_ranking_macro = summarize_ppe_rankings(raw_df)

    atomic_write_csv(output_dir / "llm_eval_raw_results.csv", raw_df)
    atomic_write_csv(output_dir / "llm_eval_summary_by_target.csv", target_summary)
    atomic_write_csv(output_dir / "llm_eval_summary_by_benchmark.csv", benchmark_summary)
    atomic_write_csv(output_dir / "llm_eval_summary_by_evaluator.csv", macro_summary)
    atomic_write_csv(output_dir / "llm_eval_summary_main.csv", overall_summary)
    atomic_write_csv(output_dir / "llm_eval_ppe_ranking_raw.csv", ppe_ranking_raw)
    atomic_write_csv(output_dir / "llm_eval_ppe_ranking_summary.csv", ppe_ranking_macro)

    subgroup_path = output_dir / "llm_eval_ppe_subgroup_summary.csv"
    if not ppe_subgroup_summary.empty:
        atomic_write_csv(subgroup_path, ppe_subgroup_summary)
    elif subgroup_path.exists():
        subgroup_path.unlink()

    plot_llm_eval_main(overall_summary, output_dir / "fig_llm_eval_main.pdf")
    plot_llm_eval_by_evaluator(macro_summary, output_dir / "fig_llm_eval_by_evaluator.pdf")
    plot_llm_eval_by_benchmark(benchmark_summary, output_dir / "fig_llm_eval_by_benchmark.pdf")
    plot_ppe_ranking(ppe_ranking_macro, output_dir / "fig_llm_ppe_ranking.pdf")
    write_llm_summary_table(overall_summary, output_dir)
    write_ppe_ranking_table(ppe_ranking_macro, output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="*", default=list(DEFAULT_TRACKS), help="Benchmark tracks to run.")
    parser.add_argument(
        "--evaluators",
        nargs="*",
        default=list(DEFAULT_EVALUATORS),
        help="Evaluator models to use. Prometheus 2 is only valid on PPE.",
    )
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_eval/default_run"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache/llm_eval"),
    )
    parser.add_argument(
        "--rewardbench-split",
        type=str,
        default="filtered",
        help="Hugging Face split to use when building the RewardBench row cache.",
    )
    parser.add_argument(
        "--estimators",
        nargs="*",
        default=list(ALL_ESTIMATORS),
        help="Optional estimator subset to run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke configuration using cached rows/scores when available.",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Ignore existing evaluator score caches and recompute them.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse completed stage checkpoints when present.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Discard any existing LLM stage checkpoints and rerun them from scratch.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    tracks = args.tracks or list(DEFAULT_TRACKS)
    unknown_tracks = sorted(set(tracks) - set(TRACK_ORDER))
    if unknown_tracks:
        raise ValueError(f"Unknown tracks: {unknown_tracks}")

    evaluators = args.evaluators or list(DEFAULT_EVALUATORS)
    unknown_evaluators = sorted(set(evaluators) - set(EVALUATOR_SPECS))
    if unknown_evaluators:
        raise ValueError(f"Unknown evaluators: {unknown_evaluators}")

    unknown_estimators = sorted(set(args.estimators) - set(ALL_ESTIMATORS))
    if unknown_estimators:
        raise ValueError(f"Unknown estimators: {unknown_estimators}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    budgets = SMOKE_BUDGETS if args.smoke else DEFAULT_BUDGETS
    replications = 2 if args.smoke else args.replications
    ppe_subgroup_summary = pd.DataFrame()
    track_contexts: dict[str, dict[str, Any]] = {}
    stage_specs: list[dict[str, Any]] = []

    for track in tracks:
        rows = ensure_track_rows(track, args.cache_dir, rewardbench_split=args.rewardbench_split)
        context: dict[str, Any] = {"rows": rows}
        if track == "ppe_human":
            target_count = SMOKE_PPE_TARGETS if args.smoke else PPE_TARGET_MODELS
            target_models = select_top_ppe_models(rows, top_k=target_count)
            context["target_models"] = tuple(target_models)
            ppe_subgroup_summary = build_ppe_subgroup_summary(rows, target_models=target_models)
            stage_groups = [(PPE_HUMAN_BENCHMARK, model) for model in target_models]
        elif track == "ppe_correctness":
            stage_groups = [
                (str(benchmark), str(target_model))
                for benchmark, target_model in (
                    rows.loc[:, ["benchmark", "target_model"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
            ]
        else:
            stage_groups = [(REWARDBENCH_BENCHMARK, "__overall__")]
        track_contexts[track] = context

        for evaluator in evaluators:
            spec = get_evaluator_spec(evaluator)
            if track not in spec.supported_tracks:
                continue
            for benchmark_str, target_model_str in stage_groups:
                stage_specs.append(
                    {
                        "track": track,
                        "benchmark": benchmark_str,
                        "evaluator": evaluator,
                        "target_model": target_model_str,
                        "stage_key": stage_key(
                            track=track,
                            evaluator=evaluator,
                            benchmark=benchmark_str,
                            target_model=target_model_str,
                        ),
                        "stage_hash": stage_hash(
                            track=track,
                            evaluator=evaluator,
                            benchmark=benchmark_str,
                            target_model=target_model_str,
                        ),
                    }
                )

    if not stage_specs:
        raise RuntimeError("No benchmark results were generated.")

    stage_keys = [spec["stage_key"] for spec in stage_specs]
    if len(stage_keys) != len(set(stage_keys)):
        stage_key_counts = Counter(stage_keys)
        duplicates = sorted([key for key, count in stage_key_counts.items() if count > 1])
        raise RuntimeError(f"Duplicate LLM stage definitions detected: {duplicates[:5]}")
    stage_spec_lookup = {
        (spec["track"], spec["evaluator"], spec["benchmark"], spec["target_model"]): spec
        for spec in stage_specs
    }

    stages_root = args.output_dir / STAGES_DIRNAME
    if not args.resume:
        shutil.rmtree(stages_root, ignore_errors=True)
        clear_llm_root_outputs(args.output_dir)

    completed_stage_frames: dict[str, pd.DataFrame] = {}
    skipped_stage_count = 0
    if args.resume:
        for spec in stage_specs:
            stage_path = stage_dir(
                args.output_dir,
                track=spec["track"],
                evaluator=spec["evaluator"],
                benchmark=spec["benchmark"],
                target_model=spec["target_model"],
            )
            if not stage_files_exist(stage_path):
                continue
            try:
                completed_stage_frames[spec["stage_hash"]] = pd.read_csv(stage_path / "raw_results.csv")
                skipped_stage_count += 1
            except Exception:
                continue

    if completed_stage_frames:
        aggregate_completed_outputs(
            output_dir=args.output_dir,
            completed_stage_frames=list(completed_stage_frames.values()),
            ppe_subgroup_summary=ppe_subgroup_summary,
        )

    expected_stage_count = len(stage_specs)
    write_run_status(
        args.output_dir,
        expected_stage_count=expected_stage_count,
        completed_stage_count=len(completed_stage_frames),
        skipped_stage_count=skipped_stage_count,
        complete=len(completed_stage_frames) == expected_stage_count,
        failed_stage=None,
    )

    for track in tracks:
        context = track_contexts[track]
        rows = context["rows"]
        target_models = context.get("target_models", ())
        for evaluator in evaluators:
            spec = get_evaluator_spec(evaluator)
            if track not in spec.supported_tracks:
                continue

            relevant_stage_hashes = [
                candidate["stage_hash"]
                for candidate in stage_specs
                if candidate["track"] == track and candidate["evaluator"] == evaluator
            ]
            if args.resume and relevant_stage_hashes and all(
                stage_hash_value in completed_stage_frames for stage_hash_value in relevant_stage_hashes
            ):
                continue

            score_frame = resolve_score_frame(
                rows=rows,
                cache_dir=args.cache_dir,
                track=track,
                evaluator=evaluator,
                force_rescore=args.force_rescore,
            )
            if track == "ppe_human":
                panel = build_ppe_panel(
                    rows,
                    score_frame,
                    target_models=target_models,
                    smoke_rows_per_target=SMOKE_PPE_ROWS_PER_TARGET if args.smoke else None,
                )
            elif track == "ppe_correctness":
                panel = build_ppe_correctness_panel(
                    rows,
                    score_frame,
                    max_rows_per_group=SMOKE_PPE_CORRECTNESS_ROWS if args.smoke else None,
                )
            else:
                panel = build_rewardbench_panel(
                    rows,
                    score_frame,
                    max_rows=SMOKE_REWARDBENCH_ROWS if args.smoke else None,
                )

            for (benchmark, target_model), frame in panel.groupby(["benchmark", "target_model"], sort=False):
                spec_key = (track, evaluator, str(benchmark), str(target_model))
                spec = stage_spec_lookup[spec_key]
                if args.resume and spec["stage_hash"] in completed_stage_frames:
                    continue

                split_schedule = build_split_schedule(
                    num_rows=frame.shape[0],
                    budgets=budgets,
                    replications=replications,
                    seed=int(args.seed),
                    schedule_key=f"{spec['track']}|{spec['benchmark']}|{spec['target_model']}",
                )
                stage_path = stage_dir(
                    args.output_dir,
                    track=spec["track"],
                    evaluator=spec["evaluator"],
                    benchmark=spec["benchmark"],
                    target_model=spec["target_model"],
                )

                try:
                    stage_rows = run_target_experiment(
                        track=spec["track"],
                        benchmark=spec["benchmark"],
                        evaluator=spec["evaluator"],
                        target_model=spec["target_model"],
                        y_total=frame["y"].to_numpy(dtype=float),
                        score_total=frame["raw_score"].to_numpy(dtype=float),
                        budgets=budgets,
                        replications=replications,
                        alpha=float(args.alpha),
                        seed=int(args.seed),
                        selected_estimators=tuple(args.estimators),
                        split_schedule=split_schedule,
                    )
                    stage_raw_df = pd.DataFrame.from_records(stage_rows)
                    if stage_raw_df.empty:
                        raise RuntimeError(f"LLM stage '{spec['stage_key']}' produced no results.")
                    atomic_write_csv(stage_path / "raw_results.csv", stage_raw_df)
                    atomic_write_json(
                        stage_path / "stage_meta.json",
                        {
                            "stage_hash": spec["stage_hash"],
                            "stage_key": spec["stage_key"],
                            "track": spec["track"],
                            "benchmark": spec["benchmark"],
                            "evaluator": spec["evaluator"],
                            "target_model": spec["target_model"],
                            "n_rows": int(frame.shape[0]),
                            "replications": int(replications),
                            "budgets": [int(value) for value in budgets],
                            "alpha": float(args.alpha),
                            "estimators": list(args.estimators),
                        },
                    )
                except Exception as exc:
                    write_run_status(
                        args.output_dir,
                        expected_stage_count=expected_stage_count,
                        completed_stage_count=len(completed_stage_frames),
                        skipped_stage_count=skipped_stage_count,
                        complete=False,
                        failed_stage={
                            "stage_hash": spec["stage_hash"],
                            "stage_key": spec["stage_key"],
                            "error": str(exc),
                        },
                    )
                    raise

                completed_stage_frames[spec["stage_hash"]] = stage_raw_df
                aggregate_completed_outputs(
                    output_dir=args.output_dir,
                    completed_stage_frames=list(completed_stage_frames.values()),
                    ppe_subgroup_summary=ppe_subgroup_summary,
                )
                write_run_status(
                    args.output_dir,
                    expected_stage_count=expected_stage_count,
                    completed_stage_count=len(completed_stage_frames),
                    skipped_stage_count=skipped_stage_count,
                    complete=len(completed_stage_frames) == expected_stage_count,
                    failed_stage=None,
                )


if __name__ == "__main__":
    main()
