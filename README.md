# Prediction-Powered Inference via Calibration

This repository contains the manuscript, package code, and reproduction pipeline for the paper on calibration-based semisupervised mean inference. The public paper is built from [`paper/main.tex`](paper/main.tex), the canonical local PDF is `paper/main.pdf`, and the website copy is `docs/paper.pdf`.

The repo also ships the `ppi_aipw` Python package and an R package in `r/ppiAIPW`, but the top-level workflow is paper-first: if you want to reproduce the manuscript or refresh paper assets, start here.

[Package website](https://larsvanderlaan.github.io/ppi-aipw/)

## Canonical paper files

- Canonical manuscript source: `paper/main.tex`
- Canonical local manuscript PDF: `paper/main.pdf`
- Published website PDF: `docs/paper.pdf`
- Secondary submission artifact: `paper/main_neurips.tex`

Additional build notes live in [paper/README.md](paper/README.md).

## Repo layout

- `paper/`: manuscript sources and tracked paper assets
- `scripts/`: public reproduction entrypoints
- `experiments/`: experiment drivers used to refresh manuscript assets
- `src/ppi_aipw/`: Python package used by the experiments and public API
- `r/ppiAIPW/`: R package and tests
- `tests/`: Python tests for the package and experiment pipelines
- `docs/`: static package website and published `paper.pdf`
- `outputs/`: local regenerated runs, caches, and intermediates

## Environment setup

Core paper and package workflow:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Optional PPE/LLM refresh workflow:

```bash
.venv/bin/pip install -r requirements-llm-eval.txt
```

You also need `latexmk` on your `PATH` to rebuild the manuscript PDF.

## Smoke checks

```bash
.venv/bin/pytest -q
./scripts/reproduce_paper.sh --smoke
```

This validates the core public workflow: synthetic simulations, reproduced `ppi_py` benchmarks, and a rebuild of `paper/main.pdf` plus `docs/paper.pdf`.

## Core paper reproduction

```bash
./scripts/reproduce_paper.sh
```

The default command refreshes the non-LLM paper assets used by `paper/main.tex` and then rebuilds the canonical paper.

It does **not** recompute the LLM-backed paper assets by default. The tracked toy and PPE figures/tables already in `paper/assets/` are reused unless you explicitly refresh them.

## Optional PPE/LLM asset refresh

```bash
./scripts/reproduce_paper.sh --include-llm-benchmark
```

This additionally refreshes the LLM-backed assets used by `paper/main.tex`:

- the grounded toy calibration figure
- PPE Human and PPE Correctness summary figures/tables
- evaluator-specific PPE appendix figure
- PPE ranking appendix figure/table

This path expects local or precomputed caches under `outputs/cache/llm_eval/` (or a custom `--llm-cache-dir`). The public paper workflow intentionally does not refresh RewardBench.

## Reproduction outputs

- Heavy regenerated outputs go under `outputs/`
- Manuscript-ready assets live under `paper/assets/`
- The canonical manuscript PDF is `paper/main.pdf`
- `paper/latexmkrc` syncs `paper/main.pdf` to `docs/paper.pdf`

The repo intentionally tracks only lightweight paper assets, not the full raw experimental outputs.

## Package quickstart

Install from PyPI:

```bash
python -m pip install ppi-aipw
```

Or from this checkout:

```bash
python -m pip install -e .
```

Minimal example:

```python
from ppi_aipw import mean_inference

result = mean_inference(
    Y,
    Yhat,
    Yhat_unlabeled,
    method="monotone_spline",
    alpha=0.1,
)

print(result.pointestimate)
print(result.ci)
```

The fuller package guide lives in [docs/ppi_aipw_vignette.md](docs/ppi_aipw_vignette.md), with the static site entrypoint at [docs/index.html](docs/index.html).
