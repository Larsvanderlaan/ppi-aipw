from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import zipfile

import numpy as np
import pandas as pd


PPE_HUMAN_BENCHMARK = "ppe_human_v1"
PPE_CORRECTNESS_BENCHMARKS = (
    "mmlu_pro_best_of_k",
    "math_best_of_k",
    "gpqa_best_of_k",
    "ifeval_best_of_k",
    "mbpp_plus_best_of_k",
)
REWARDBENCH_BENCHMARK = "rewardbench_v1"
PPE_FAMILY_TRACKS = ("ppe_human", "ppe_correctness")
PPE_RESULT_DATASET_REPO = "lmarena-ai/ppe-result-data"
PPE_RESULT_ARCHIVE_NAME = "data.zip"

TRACK_ROW_CACHE_FILENAMES = {
    "ppe_human": "ppe_human_rows.jsonl.gz",
    "ppe_correctness": "ppe_correctness_rows.jsonl.gz",
    "rewardbench_v1": "rewardbench_rows.jsonl.gz",
}

TRACK_SCORE_CACHE_PREFIXES = {
    "ppe_human": "ppe_human",
    "ppe_correctness": "ppe_correctness",
    "rewardbench_v1": "rewardbench",
}

PPE_RESULT_ARCHIVE_DIRS = {
    PPE_HUMAN_BENCHMARK: "human_preference_v1",
    "mmlu_pro_best_of_k": "mmlu_pro_best_of_k",
    "math_best_of_k": "math_best_of_k",
    "gpqa_best_of_k": "gpqa_best_of_k",
    "ifeval_best_of_k": "ifeval_best_of_k",
    "mbpp_plus_best_of_k": "mbpp_plus_best_of_k",
}

PPE_PRECOMPUTED_EVALUATOR_FILES = {
    "skywork_reward_llama3_1_8b": "Skywork-Reward-Llama-3.1-8B.json",
    "armorm_llama3_8b_v0_1": "ArmoRM-Llama3-8B-v0.1.json",
    "athene_rm_8b": "Athene-RM-8B.json",
    "starling_rm_7b_alpha": "Starling-RM-7B-alpha.json",
}

REQUIRED_SCORE_COLUMNS = (
    "unit_id",
    "evaluator",
    "score_a",
    "score_b",
    "margin",
    "score_source",
)

TRACK_REQUIRED_ROW_COLUMNS = {
    "ppe_human": (
        "unit_id",
        "track",
        "benchmark",
        "prompt",
        "response_a",
        "response_b",
        "model_a",
        "model_b",
        "winner",
    ),
    "ppe_correctness": (
        "unit_id",
        "track",
        "benchmark",
        "prompt",
        "response_a",
        "response_b",
        "model_a",
        "model_b",
        "target_model",
        "subset",
        "y",
    ),
    "rewardbench_v1": (
        "unit_id",
        "track",
        "benchmark",
        "prompt",
        "response_a",
        "response_b",
        "model_a",
        "model_b",
        "subset",
        "y",
    ),
}

PPE_CORRECTNESS_AUDIT_KEYS = (
    "records_scanned",
    "unique_pairs_kept",
    "exact_duplicate_pairs_dropped",
    "reversed_pair_conflicts",
)


@dataclass(frozen=True)
class EvaluatorSpec:
    name: str
    display_name: str
    model_id: str
    supported_tracks: tuple[str, ...]
    scorer: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    requires_precomputed: bool = False


def empty_ppe_correctness_audit() -> dict[str, int]:
    return {key: 0 for key in PPE_CORRECTNESS_AUDIT_KEYS}


def merge_ppe_correctness_audit(
    base: Mapping[str, int],
    update: Mapping[str, int],
) -> dict[str, int]:
    merged = {key: int(base.get(key, 0)) for key in PPE_CORRECTNESS_AUDIT_KEYS}
    for key in PPE_CORRECTNESS_AUDIT_KEYS:
        merged[key] += int(update.get(key, 0))
    return merged


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    return value


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            payload = {str(key): _serialize_value(value) for key, value in row.items()}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def load_jsonl_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find cache file: {path}")
    records = read_jsonl_gz(path)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame()
    return frame


def write_frame_jsonl_gz(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        write_jsonl_gz(path, [])
        return
    write_jsonl_gz(path, frame.to_dict(orient="records"))


def row_cache_path(cache_dir: Path, track: str) -> Path:
    try:
        return cache_dir / TRACK_ROW_CACHE_FILENAMES[track]
    except KeyError as exc:
        valid = ", ".join(sorted(TRACK_ROW_CACHE_FILENAMES))
        raise ValueError(f"Unknown track '{track}'. Expected one of: {valid}.") from exc


def score_cache_path(cache_dir: Path, track: str, evaluator: str) -> Path:
    try:
        prefix = TRACK_SCORE_CACHE_PREFIXES[track]
    except KeyError as exc:
        valid = ", ".join(sorted(TRACK_SCORE_CACHE_PREFIXES))
        raise ValueError(f"Unknown track '{track}'. Expected one of: {valid}.") from exc
    return cache_dir / f"{prefix}_scores_{evaluator}.jsonl.gz"


def build_ppe_correctness_unit_id(
    benchmark: str,
    target_model: str,
    question_id: str,
    pair_a: int,
    pair_b: int,
) -> str:
    return f"{benchmark}::{target_model}::{question_id}::{pair_a}::{pair_b}"


def ppe_result_archive_member_path(benchmark: str, evaluator: str) -> str:
    try:
        benchmark_dir = PPE_RESULT_ARCHIVE_DIRS[benchmark]
        filename = PPE_PRECOMPUTED_EVALUATOR_FILES[evaluator]
    except KeyError as exc:
        raise ValueError(f"Unsupported PPE precomputed source: benchmark={benchmark}, evaluator={evaluator}") from exc
    return f"data/{benchmark_dir}/{filename}"


def _coerce_scalar_score(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected a non-empty score list.")
        return float(value[0])
    return float(value)


def iter_deduped_ppe_correctness_pairs(
    *,
    benchmark: str,
    record: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    deduped_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    audit = empty_ppe_correctness_audit()
    audit["records_scanned"] = 1

    for pair in record.get("sampled_conflict_pairs", []) or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        pair_a = int(pair[0])
        pair_b = int(pair[1])
        ordered_pair = (pair_a, pair_b)
        reversed_pair = (pair_b, pair_a)
        if reversed_pair in seen_pairs:
            audit["reversed_pair_conflicts"] += 1
            question_id = str(record.get("question_id", "unknown"))
            target_model = str(record.get("model_name", "unknown"))
            raise ValueError(
                "PPE correctness archive contains conflicting ordered and reversed conflict pairs "
                f"for benchmark='{benchmark}', question_id='{question_id}', "
                f"target_model='{target_model}', pair={ordered_pair}."
            )
        if ordered_pair in seen_pairs:
            audit["exact_duplicate_pairs_dropped"] += 1
            continue
        seen_pairs.add(ordered_pair)
        deduped_pairs.append(ordered_pair)

    audit["unique_pairs_kept"] = len(deduped_pairs)
    return deduped_pairs, audit


def ensure_ppe_result_archive_path(cache_dir: Path) -> Path:
    local_dir = cache_dir / "_ppe_result_data"
    archive_path = local_dir / PPE_RESULT_ARCHIVE_NAME
    if archive_path.exists():
        return archive_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "Downloading PPE precomputed results requires the optional `huggingface_hub` dependency."
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=PPE_RESULT_DATASET_REPO,
        filename=PPE_RESULT_ARCHIVE_NAME,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def load_ppe_result_json(cache_dir: Path, member_path: str) -> list[dict[str, Any]]:
    archive_path = ensure_ppe_result_archive_path(cache_dir)
    with zipfile.ZipFile(archive_path, "r") as handle:
        return json.loads(handle.read(member_path))


def build_ppe_human_score_frame_from_records(
    *,
    evaluator: str,
    records: list[dict[str, Any]],
    allowed_unit_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        unit_id = str(record["question_id"])
        if allowed_unit_ids is not None and unit_id not in allowed_unit_ids:
            continue
        score_a = _coerce_scalar_score(record["score_1"])
        score_b = _coerce_scalar_score(record["score_2"])
        rows.append(
            {
                "unit_id": unit_id,
                "evaluator": evaluator,
                "score_a": score_a,
                "score_b": score_b,
                "margin": score_a - score_b,
                "score_source": "ppe_result_data",
            }
        )
    return pd.DataFrame.from_records(rows, columns=REQUIRED_SCORE_COLUMNS)


def build_ppe_correctness_score_frame_from_records(
    *,
    benchmark: str,
    evaluator: str,
    records: list[dict[str, Any]],
    allowed_unit_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record["question_id"])
        target_model = str(record["model_name"])
        deduped_pairs, _ = iter_deduped_ppe_correctness_pairs(
            benchmark=benchmark,
            record=record,
        )
        for pair_a, pair_b in deduped_pairs:
            unit_id = build_ppe_correctness_unit_id(
                benchmark,
                target_model,
                question_id,
                pair_a,
                pair_b,
            )
            if allowed_unit_ids is not None and unit_id not in allowed_unit_ids:
                continue
            score_a = _coerce_scalar_score(record[f"score_{pair_a + 1}"])
            score_b = _coerce_scalar_score(record[f"score_{pair_b + 1}"])
            rows.append(
                {
                    "unit_id": unit_id,
                    "evaluator": evaluator,
                    "score_a": score_a,
                    "score_b": score_b,
                    "margin": score_a - score_b,
                    "score_source": "ppe_result_data",
                }
            )
    return pd.DataFrame.from_records(rows, columns=REQUIRED_SCORE_COLUMNS)


def bootstrap_ppe_family_score_cache(
    *,
    rows: pd.DataFrame,
    cache_dir: Path,
    track: str,
    evaluator: str,
) -> pd.DataFrame:
    if track not in PPE_FAMILY_TRACKS:
        raise ValueError(f"Track '{track}' is not part of the PPE family bootstrap path.")
    allowed_unit_ids = set(rows["unit_id"].astype(str))
    if not allowed_unit_ids:
        raise ValueError(f"No unit ids provided for PPE-family bootstrap on track '{track}'.")

    if track == "ppe_human":
        records = load_ppe_result_json(
            cache_dir,
            ppe_result_archive_member_path(PPE_HUMAN_BENCHMARK, evaluator),
        )
        frame = build_ppe_human_score_frame_from_records(
            evaluator=evaluator,
            records=records,
            allowed_unit_ids=allowed_unit_ids,
        )
    else:
        frames: list[pd.DataFrame] = []
        benchmarks = sorted(rows["benchmark"].astype(str).unique().tolist())
        for benchmark in benchmarks:
            records = load_ppe_result_json(
                cache_dir,
                ppe_result_archive_member_path(benchmark, evaluator),
            )
            frames.append(
                build_ppe_correctness_score_frame_from_records(
                    benchmark=benchmark,
                    evaluator=evaluator,
                    records=records,
                    allowed_unit_ids=allowed_unit_ids,
                )
            )
        frame = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=REQUIRED_SCORE_COLUMNS)
        )

    matched_unit_ids = set(frame["unit_id"].astype(str))
    missing = sorted(allowed_unit_ids - matched_unit_ids)
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"PPE-family precomputed scores for '{evaluator}' are missing {len(missing)} rows. "
            f"Examples: {preview}"
        )
    write_score_cache(cache_dir, track, evaluator, frame)
    return frame


def _validate_unique_unit_ids(frame: pd.DataFrame, *, label: str) -> None:
    duplicated = frame["unit_id"].astype(str).duplicated(keep=False)
    if not duplicated.any():
        return
    duplicate_ids = (
        frame.loc[duplicated, "unit_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    preview = duplicate_ids[:5]
    raise ValueError(
        f"{label} contains {len(duplicate_ids)} duplicated unit_id values. "
        f"Examples: {preview}"
    )


def validate_track_rows(frame: pd.DataFrame, *, track: str) -> pd.DataFrame:
    try:
        required_columns = TRACK_REQUIRED_ROW_COLUMNS[track]
    except KeyError as exc:
        valid = ", ".join(sorted(TRACK_REQUIRED_ROW_COLUMNS))
        raise ValueError(f"Unknown track '{track}'. Expected one of: {valid}.") from exc

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Row cache for '{track}' is missing required columns: {missing}")

    validated = frame.copy()
    validated["unit_id"] = validated["unit_id"].astype(str)
    validated["track"] = validated["track"].astype(str)
    if not np.all(validated["track"] == track):
        raise ValueError(
            f"Row cache for '{track}' contains mismatched track labels: "
            f"{sorted(validated['track'].dropna().unique().tolist())}"
        )
    _validate_unique_unit_ids(validated, label=f"Row cache for '{track}'")
    return validated


def load_track_rows_cache(cache_dir: Path, track: str) -> pd.DataFrame:
    frame = load_jsonl_frame(row_cache_path(cache_dir, track))
    if frame.empty:
        return pd.DataFrame()
    return validate_track_rows(frame, track=track)


def write_track_rows_cache(cache_dir: Path, track: str, frame: pd.DataFrame) -> Path:
    path = row_cache_path(cache_dir, track)
    validated = validate_track_rows(frame, track=track)
    write_frame_jsonl_gz(path, validated)
    return path


def validate_score_frame(frame: pd.DataFrame, *, evaluator: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Score cache for '{evaluator}' is missing required columns: {missing}")

    validated = frame.loc[:, REQUIRED_SCORE_COLUMNS].copy()
    validated["unit_id"] = validated["unit_id"].astype(str)
    validated["evaluator"] = validated["evaluator"].astype(str)
    if not np.all(validated["evaluator"] == evaluator):
        raise ValueError(
            f"Score cache for '{evaluator}' contains mismatched evaluator labels: "
            f"{sorted(validated['evaluator'].dropna().unique().tolist())}"
        )
    for column in ("score_a", "score_b", "margin"):
        validated[column] = pd.to_numeric(validated[column], errors="raise").astype(float)
    validated["score_source"] = validated["score_source"].astype(str)
    _validate_unique_unit_ids(validated, label=f"Score cache for '{evaluator}'")
    return validated


def load_score_cache(cache_dir: Path, track: str, evaluator: str) -> pd.DataFrame:
    frame = load_jsonl_frame(score_cache_path(cache_dir, track, evaluator))
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_SCORE_COLUMNS)
    return validate_score_frame(frame, evaluator=evaluator)


def write_score_cache(cache_dir: Path, track: str, evaluator: str, frame: pd.DataFrame) -> Path:
    path = score_cache_path(cache_dir, track, evaluator)
    validated = validate_score_frame(frame, evaluator=evaluator)
    write_frame_jsonl_gz(path, validated)
    return path


def _normalize_pairrm_scores(result: Any) -> np.ndarray:
    if isinstance(result, tuple) and len(result) == 2:
        candidate = np.asarray(result[1], dtype=float)
    else:
        candidate = np.asarray(result, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 2:
        raise RuntimeError(
            "Unexpected PairRM rank output shape. Expected an (n, 2) score matrix "
            f"but received {candidate.shape}."
        )
    return candidate


def score_with_pairrm(rows: pd.DataFrame, model_id: str) -> pd.DataFrame:
    try:
        import llm_blender
    except ImportError as exc:  # pragma: no cover - exercised only in optional runtime
        raise RuntimeError(
            "PairRM scoring requires the optional `llm_blender` dependency. "
            "Install the LLM benchmark extras or provide a precomputed score cache."
        ) from exc

    blender = llm_blender.Blender()
    blender.loadranker(model_id, device="cpu")
    prompts = rows["prompt"].astype(str).tolist()
    candidates = rows[["response_a", "response_b"]].astype(str).values.tolist()
    raw_scores = blender.rank(
        prompts,
        candidates,
        return_scores=True,
        batch_size=4,
    )
    scores = _normalize_pairrm_scores(raw_scores)
    frame = pd.DataFrame(
        {
            "unit_id": rows["unit_id"].astype(str).tolist(),
            "evaluator": "pairrm",
            "score_a": scores[:, 0],
            "score_b": scores[:, 1],
        }
    )
    frame["margin"] = frame["score_a"] - frame["score_b"]
    frame["score_source"] = "local_pairrm"
    return frame


def _reward_text(prompt: str, response: str) -> str:
    return (
        "Instruction:\n"
        f"{prompt.strip()}\n\n"
        "Response:\n"
        f"{response.strip()}"
    )


def _extract_reward_scores(logits: Any) -> np.ndarray:
    array = np.asarray(logits, dtype=float)
    if array.ndim == 1:
        return array.astype(float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0].astype(float)
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, -1].astype(float)
    raise RuntimeError(f"Unexpected reward-model logit shape: {array.shape}")


def configure_offline_hf_runtime() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_hf_snapshot_path(model_id: str) -> str:
    configure_offline_hf_runtime()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only in optional runtime
        raise RuntimeError(
            "Offline Hugging Face model loading requires the optional `huggingface_hub` dependency. "
            "Install the LLM benchmark extras or provide a precomputed score cache."
        ) from exc

    try:
        snapshot_path = snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on local cache state
        raise RuntimeError(
            f"Model '{model_id}' is not available in the local Hugging Face cache. "
            "Download it before running the offline LLM benchmark, or provide a precomputed score cache."
        ) from exc
    return str(Path(snapshot_path))


def score_with_skywork(rows: pd.DataFrame, model_id: str) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only in optional runtime
        raise RuntimeError(
            "Skywork reward-model scoring requires optional `torch` and `transformers` "
            "dependencies. Install the LLM benchmark extras or provide a precomputed score cache."
        ) from exc

    model_path = resolve_local_hf_snapshot_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    texts_a = [
        _reward_text(prompt, response)
        for prompt, response in rows[["prompt", "response_a"]].astype(str).itertuples(index=False, name=None)
    ]
    texts_b = [
        _reward_text(prompt, response)
        for prompt, response in rows[["prompt", "response_b"]].astype(str).itertuples(index=False, name=None)
    ]

    def _batched_scores(texts: list[str], batch_size: int = 8) -> np.ndarray:
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded)
                scores.append(_extract_reward_scores(output.logits.detach().cpu().numpy()))
        return np.concatenate(scores, axis=0)

    score_a = _batched_scores(texts_a)
    score_b = _batched_scores(texts_b)
    frame = pd.DataFrame(
        {
            "unit_id": rows["unit_id"].astype(str).tolist(),
            "evaluator": "skywork_qwen3_4b",
            "score_a": score_a.astype(float),
            "score_b": score_b.astype(float),
        }
    )
    frame["margin"] = frame["score_a"] - frame["score_b"]
    frame["score_source"] = "local_skywork"
    return frame


def score_with_precomputed_only(rows: pd.DataFrame, model_id: str) -> pd.DataFrame:
    raise RuntimeError(
        "This evaluator is configured as a precomputed-cache-only evaluator in this benchmark path. "
        f"Populate the score cache for model '{model_id}' before requesting it."
    )


EVALUATOR_SPECS: dict[str, EvaluatorSpec] = {
    "pairrm": EvaluatorSpec(
        name="pairrm",
        display_name="PairRM",
        model_id="llm-blender/PairRM",
        supported_tracks=("ppe_human", "rewardbench_v1"),
        scorer=lambda rows: score_with_pairrm(rows, model_id="llm-blender/PairRM"),
    ),
    "skywork_qwen3_4b": EvaluatorSpec(
        name="skywork_qwen3_4b",
        display_name="Skywork-Reward-Qwen3-4B",
        model_id="Skywork/Skywork-Reward-V2-Qwen3-4B",
        supported_tracks=("rewardbench_v1",),
        scorer=lambda rows: score_with_skywork(rows, model_id="Skywork/Skywork-Reward-V2-Qwen3-4B"),
    ),
    "prometheus2_7b": EvaluatorSpec(
        name="prometheus2_7b",
        display_name="Prometheus 2 7B",
        model_id="prometheus-eval/prometheus-7b-v2.0",
        supported_tracks=("ppe_human",),
        scorer=lambda rows: score_with_precomputed_only(
            rows,
            model_id="prometheus-eval/prometheus-7b-v2.0",
        ),
        requires_precomputed=True,
    ),
    "skywork_reward_llama3_1_8b": EvaluatorSpec(
        name="skywork_reward_llama3_1_8b",
        display_name="Skywork-Reward-Llama-3.1-8B",
        model_id="Skywork/Skywork-Reward-Llama-3.1-8B",
        supported_tracks=("ppe_human", "ppe_correctness"),
        scorer=lambda rows: score_with_precomputed_only(
            rows,
            model_id="Skywork/Skywork-Reward-Llama-3.1-8B",
        ),
        requires_precomputed=True,
    ),
    "armorm_llama3_8b_v0_1": EvaluatorSpec(
        name="armorm_llama3_8b_v0_1",
        display_name="ArmoRM-Llama3-8B-v0.1",
        model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        supported_tracks=("ppe_human", "ppe_correctness"),
        scorer=lambda rows: score_with_precomputed_only(
            rows,
            model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        ),
        requires_precomputed=True,
    ),
    "athene_rm_8b": EvaluatorSpec(
        name="athene_rm_8b",
        display_name="Athene-RM-8B",
        model_id="Nexusflow/Athene-RM-8B",
        supported_tracks=("ppe_human", "ppe_correctness"),
        scorer=lambda rows: score_with_precomputed_only(
            rows,
            model_id="Nexusflow/Athene-RM-8B",
        ),
        requires_precomputed=True,
    ),
    "starling_rm_7b_alpha": EvaluatorSpec(
        name="starling_rm_7b_alpha",
        display_name="Starling-RM-7B-alpha",
        model_id="berkeley-nest/Starling-RM-7B-alpha",
        supported_tracks=("ppe_human", "ppe_correctness"),
        scorer=lambda rows: score_with_precomputed_only(
            rows,
            model_id="berkeley-nest/Starling-RM-7B-alpha",
        ),
        requires_precomputed=True,
    ),
}


def get_evaluator_spec(evaluator: str) -> EvaluatorSpec:
    try:
        return EVALUATOR_SPECS[evaluator]
    except KeyError as exc:
        valid = ", ".join(sorted(EVALUATOR_SPECS))
        raise ValueError(f"Unknown evaluator '{evaluator}'. Expected one of: {valid}.") from exc


def resolve_score_frame(
    *,
    rows: pd.DataFrame,
    cache_dir: Path,
    track: str,
    evaluator: str,
    force_rescore: bool = False,
) -> pd.DataFrame:
    cache_path = score_cache_path(cache_dir, track, evaluator)
    spec = get_evaluator_spec(evaluator)
    if cache_path.exists() and not force_rescore:
        try:
            cached = load_score_cache(cache_dir, track, evaluator)
            if spec.requires_precomputed and track in PPE_FAMILY_TRACKS:
                expected_unit_ids = set(rows["unit_id"].astype(str))
                cached_unit_ids = set(cached["unit_id"].astype(str))
                if cached_unit_ids != expected_unit_ids:
                    missing = sorted(expected_unit_ids - cached_unit_ids)
                    extra = sorted(cached_unit_ids - expected_unit_ids)
                    raise ValueError(
                        f"Cached PPE-family score frame for evaluator '{evaluator}' and track '{track}' "
                        f"does not match the deduplicated row set. Missing={len(missing)}, extra={len(extra)}."
                    )
            return cached
        except ValueError:
            if not (spec.requires_precomputed and track in PPE_FAMILY_TRACKS):
                raise

    if track not in spec.supported_tracks:
        raise ValueError(f"Evaluator '{evaluator}' does not support track '{track}'.")
    if spec.requires_precomputed and track in PPE_FAMILY_TRACKS:
        return bootstrap_ppe_family_score_cache(
            rows=rows,
            cache_dir=cache_dir,
            track=track,
            evaluator=evaluator,
        )
    if spec.scorer is None:
        raise RuntimeError(f"No scorer is configured for evaluator '{evaluator}'.")
    scored = spec.scorer(rows)
    write_score_cache(cache_dir, track, evaluator, scored)
    return scored
