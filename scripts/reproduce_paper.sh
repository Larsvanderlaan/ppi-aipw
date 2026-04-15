#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
OUTPUT_ROOT="$ROOT_DIR/outputs"
PAPER_ASSETS_DIR="$ROOT_DIR/paper/assets"
CACHE_DIR="$OUTPUT_ROOT/cache/ppi_datasets"
SEED="20260411"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      shift
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
  SIM_OUTPUT_DIR="$OUTPUT_ROOT/simulations/smoke"
  REAL_OUTPUT_DIR="$OUTPUT_ROOT/real_data/smoke"
else
  SIM_PROFILE="paper"
  SIM_REPLICATIONS="5000"
  REAL_REPLICATIONS="5000"
fi

mkdir -p "$OUTPUT_ROOT" "$PAPER_ASSETS_DIR" "$CACHE_DIR"
export MPLCONFIGDIR="$OUTPUT_ROOT/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

run_cmd() {
  echo "+ $*"
  "$@"
}

run_cmd rm -rf "$SIM_OUTPUT_DIR"
run_cmd rm -rf "$REAL_OUTPUT_DIR"
run_cmd mkdir -p "$SIM_OUTPUT_DIR" "$REAL_OUTPUT_DIR"

run_cmd "$PYTHON_BIN" "$ROOT_DIR/experiments/simulate.py" \
  --profile "$SIM_PROFILE" \
  --replications "$SIM_REPLICATIONS" \
  --seed "$SEED" \
  --recommendations-file "$SIM_RECOMMENDATIONS" \
  --output-dir "$SIM_OUTPUT_DIR"

run_cmd "$PYTHON_BIN" "$ROOT_DIR/experiments/plot_results.py" \
  --input-dir "$SIM_OUTPUT_DIR"

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
run_cmd "${REAL_ARGS[@]}"

run_cmd cp "$SIM_OUTPUT_DIR/fig_simulation_main.pdf" "$PAPER_ASSETS_DIR/sim_fig_main.pdf"
run_cmd cp "$SIM_OUTPUT_DIR/fig_bias_coverage.pdf" "$PAPER_ASSETS_DIR/sim_fig_bias_coverage.pdf"
run_cmd cp "$REAL_OUTPUT_DIR/fig_paper_main_grid.pdf" "$PAPER_ASSETS_DIR/fig_paper_main_grid.pdf"
run_cmd cp "$REAL_OUTPUT_DIR/fig_paper_metric_grid.pdf" "$PAPER_ASSETS_DIR/fig_paper_metric_grid.pdf"
run_cmd cp "$REAL_OUTPUT_DIR/table_paper_summary.tex" "$PAPER_ASSETS_DIR/table_paper_summary.tex"

echo
echo "Paper assets refreshed:"
echo "  $PAPER_ASSETS_DIR/sim_fig_main.pdf"
echo "  $PAPER_ASSETS_DIR/sim_fig_bias_coverage.pdf"
echo "  $PAPER_ASSETS_DIR/fig_paper_main_grid.pdf"
echo "  $PAPER_ASSETS_DIR/fig_paper_metric_grid.pdf"
echo "  $PAPER_ASSETS_DIR/table_paper_summary.tex"
