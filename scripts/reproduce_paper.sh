#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
OUTPUT_ROOT="$ROOT_DIR/outputs"
PAPER_ASSETS_DIR="$ROOT_DIR/paper/assets"
CACHE_DIR="$OUTPUT_ROOT/cache/ppi_datasets"
LLM_CACHE_DIR="$OUTPUT_ROOT/cache/llm_eval"
PAPER_TEX="$ROOT_DIR/paper/main.tex"
SEED="20260411"
SMOKE=0
MAIN_TEXT_ONLY=0
REPLICATIONS=""
INCLUDE_LLM_BENCHMARK=0
ONLY_LLM_BENCHMARK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      shift
      ;;
    --main-text-only)
      MAIN_TEXT_ONLY=1
      shift
      ;;
    --replications)
      REPLICATIONS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --paper-assets-dir)
      PAPER_ASSETS_DIR="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --llm-cache-dir)
      LLM_CACHE_DIR="$2"
      shift 2
      ;;
    --include-llm-benchmark)
      INCLUDE_LLM_BENCHMARK=1
      shift
      ;;
    --only-llm-benchmark)
      ONLY_LLM_BENCHMARK=1
      INCLUDE_LLM_BENCHMARK=1
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$DEFAULT_VENV_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_VENV_PYTHON"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Could not find a usable Python executable. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Could not find Python executable at $PYTHON_BIN" >&2
  exit 1
fi

SIM_RECOMMENDATIONS="$ROOT_DIR/experiments/paper_simulation_recommendations.json"
SIM_OUTPUT_DIR="$OUTPUT_ROOT/simulations/final"
REAL_OUTPUT_DIR="$OUTPUT_ROOT/real_data/final"

if [[ "$SMOKE" -eq 1 ]]; then
  SIM_PROFILE="quick"
  SIM_REPLICATIONS="2"
  REAL_REPLICATIONS="2"
  TOY_REPLICATIONS="25"
  SIM_OUTPUT_DIR="$OUTPUT_ROOT/simulations/smoke"
  REAL_OUTPUT_DIR="$OUTPUT_ROOT/real_data/smoke"
else
  SIM_PROFILE="paper"
  SIM_REPLICATIONS="${REPLICATIONS:-200}"
  REAL_REPLICATIONS="${REPLICATIONS:-200}"
  TOY_REPLICATIONS="${REPLICATIONS:-500}"
fi

if [[ "$MAIN_TEXT_ONLY" -eq 1 && "$SMOKE" -eq 0 ]]; then
  SIM_OUTPUT_DIR="$OUTPUT_ROOT/simulations/main_text"
  REAL_OUTPUT_DIR="$OUTPUT_ROOT/real_data/main_text"
fi

mkdir -p "$OUTPUT_ROOT" "$PAPER_ASSETS_DIR" "$CACHE_DIR"
export MPLCONFIGDIR="$OUTPUT_ROOT/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

run_cmd() {
  echo "+ $*"
  "$@"
}

if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  run_cmd rm -rf "$SIM_OUTPUT_DIR"
  run_cmd rm -rf "$REAL_OUTPUT_DIR"
  run_cmd mkdir -p "$SIM_OUTPUT_DIR" "$REAL_OUTPUT_DIR"
fi

SIM_ARGS=(
  "$PYTHON_BIN" "$ROOT_DIR/experiments/simulate.py"
  --profile "$SIM_PROFILE" \
  --replications "$SIM_REPLICATIONS" \
  --seed "$SEED" \
  --recommendations-file "$SIM_RECOMMENDATIONS" \
  --output-dir "$SIM_OUTPUT_DIR"
)
if [[ "$MAIN_TEXT_ONLY" -eq 1 ]]; then
  SIM_ARGS+=(--main-text-only)
fi
if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  run_cmd "${SIM_ARGS[@]}"
fi

PLOT_ARGS=(
  "$PYTHON_BIN" "$ROOT_DIR/experiments/plot_results.py"
  --input-dir "$SIM_OUTPUT_DIR"
)
if [[ "$MAIN_TEXT_ONLY" -eq 1 ]]; then
  PLOT_ARGS+=(--main-text-only)
fi
if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  run_cmd "${PLOT_ARGS[@]}"
fi

REAL_ARGS=(
  "$PYTHON_BIN" "$ROOT_DIR/experiments/ppi_mean_reproduction.py"
  --replications "$REAL_REPLICATIONS"
  --seed "$SEED"
  --output-dir "$REAL_OUTPUT_DIR"
  --cache-dir "$CACHE_DIR"
)
if [[ "$SMOKE" -eq 1 ]]; then
  REAL_ARGS+=(--smoke)
fi
if [[ "$MAIN_TEXT_ONLY" -eq 1 ]]; then
  REAL_ARGS+=(--main-text-only)
fi
if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  run_cmd "${REAL_ARGS[@]}"
fi

LLM_OUTPUT_DIR="$OUTPUT_ROOT/llm_eval/final"
if [[ "$SMOKE" -eq 1 ]]; then
  LLM_REPLICATIONS="2"
  LLM_OUTPUT_DIR="$OUTPUT_ROOT/llm_eval/smoke"
else
  LLM_REPLICATIONS="${REPLICATIONS:-200}"
fi

if [[ "$INCLUDE_LLM_BENCHMARK" -eq 1 ]]; then
  run_cmd mkdir -p "$LLM_OUTPUT_DIR"
  if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
    run_cmd rm -rf "$LLM_OUTPUT_DIR"
    run_cmd mkdir -p "$LLM_OUTPUT_DIR"
  fi
  LLM_ARGS=(
    "$PYTHON_BIN" "$ROOT_DIR/experiments/llm_eval_benchmark.py"
    --replications "$LLM_REPLICATIONS"
    --seed "$SEED"
    --output-dir "$LLM_OUTPUT_DIR"
    --cache-dir "$LLM_CACHE_DIR"
    --resume
    --tracks ppe_human ppe_correctness
  )
  TOY_ARGS=(
    "$PYTHON_BIN" "$ROOT_DIR/experiments/toy_calibration_study.py"
    --seed "$SEED"
    --replications "$TOY_REPLICATIONS"
    --output-dir "$SIM_OUTPUT_DIR"
    --llm-cache-dir "$LLM_CACHE_DIR"
    --dataset-cache-dir "$CACHE_DIR"
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    LLM_ARGS+=(--smoke)
  fi
  run_cmd "${TOY_ARGS[@]}"
  run_cmd env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR="$MPLCONFIGDIR" \
    "${LLM_ARGS[@]}"
  if ! "$PYTHON_BIN" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1], "r", encoding="utf-8")).get("complete") else 1)' "$LLM_OUTPUT_DIR/run_status.json"; then
    echo "LLM benchmark run did not complete successfully; leaving staged outputs in $LLM_OUTPUT_DIR" >&2
    exit 1
  fi
fi

if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  run_cmd cp "$SIM_OUTPUT_DIR/fig_simulation_main.pdf" "$PAPER_ASSETS_DIR/sim_fig_main.pdf"
  run_cmd cp "$REAL_OUTPUT_DIR/fig_paper_main_grid.pdf" "$PAPER_ASSETS_DIR/fig_paper_main_grid.pdf"
fi

if [[ "$ONLY_LLM_BENCHMARK" -eq 0 && "$MAIN_TEXT_ONLY" -eq 0 ]]; then
  run_cmd cp "$SIM_OUTPUT_DIR/fig_bias_coverage.pdf" "$PAPER_ASSETS_DIR/sim_fig_bias_coverage.pdf"
  run_cmd cp "$REAL_OUTPUT_DIR/fig_paper_metric_grid.pdf" "$PAPER_ASSETS_DIR/fig_paper_metric_grid.pdf"
  run_cmd cp "$REAL_OUTPUT_DIR/table_paper_summary.tex" "$PAPER_ASSETS_DIR/table_paper_summary.tex"
fi
if [[ "$INCLUDE_LLM_BENCHMARK" -eq 1 ]]; then
  run_cmd cp "$SIM_OUTPUT_DIR/fig_toy_calibration_ppi.pdf" "$PAPER_ASSETS_DIR/fig_toy_calibration_ppi.pdf"
  run_cmd cp "$LLM_OUTPUT_DIR/fig_llm_eval_main.pdf" "$PAPER_ASSETS_DIR/fig_llm_eval_main.pdf"
  run_cmd cp "$LLM_OUTPUT_DIR/fig_llm_eval_by_evaluator.pdf" "$PAPER_ASSETS_DIR/fig_llm_eval_by_evaluator.pdf"
  run_cmd cp "$LLM_OUTPUT_DIR/fig_llm_ppe_ranking.pdf" "$PAPER_ASSETS_DIR/fig_llm_ppe_ranking.pdf"
  run_cmd cp "$LLM_OUTPUT_DIR/table_llm_summary.tex" "$PAPER_ASSETS_DIR/table_llm_summary.tex"
  run_cmd cp "$LLM_OUTPUT_DIR/table_llm_ppe_ranking.tex" "$PAPER_ASSETS_DIR/table_llm_ppe_ranking.tex"
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is required to rebuild paper/main.tex" >&2
  exit 1
fi
run_cmd latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir="$ROOT_DIR/paper" "$PAPER_TEX"

echo
echo "Paper assets refreshed:"
if [[ "$ONLY_LLM_BENCHMARK" -eq 0 ]]; then
  echo "  $PAPER_ASSETS_DIR/sim_fig_main.pdf"
  echo "  $PAPER_ASSETS_DIR/fig_paper_main_grid.pdf"
fi
if [[ "$ONLY_LLM_BENCHMARK" -eq 0 && "$MAIN_TEXT_ONLY" -eq 0 ]]; then
  echo "  $PAPER_ASSETS_DIR/sim_fig_bias_coverage.pdf"
  echo "  $PAPER_ASSETS_DIR/fig_paper_metric_grid.pdf"
  echo "  $PAPER_ASSETS_DIR/table_paper_summary.tex"
fi
if [[ "$INCLUDE_LLM_BENCHMARK" -eq 1 ]]; then
  echo "  $PAPER_ASSETS_DIR/fig_toy_calibration_ppi.pdf"
  echo "  $PAPER_ASSETS_DIR/fig_llm_eval_main.pdf"
  echo "  $PAPER_ASSETS_DIR/fig_llm_eval_by_evaluator.pdf"
  echo "  $PAPER_ASSETS_DIR/fig_llm_ppe_ranking.pdf"
  echo "  $PAPER_ASSETS_DIR/table_llm_summary.tex"
  echo "  $PAPER_ASSETS_DIR/table_llm_ppe_ranking.tex"
fi
echo "  $ROOT_DIR/paper/main.pdf"
echo "  $ROOT_DIR/docs/paper.pdf"
