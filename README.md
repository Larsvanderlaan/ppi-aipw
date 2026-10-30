# Prediction-Powered Inference via Calibration

This repository contains the manuscript, package code, and reproduction pipeline for the paper on calibration-based semisupervised mean inference. The public paper is built from [`paper/main.tex`](paper/main.tex), the canonical local PDF is `paper/main.pdf`, and the website copy is `docs/paper.pdf`.

The repo also ships the `ppi_aipw` Python package and an R package in `r/ppiAIPW`, but the top-level workflow is paper-first: if you want to reproduce the manuscript or refresh paper assets, start here.

[Package website](https://larsvanderlaan.github.io/ppi-aipw/)

## Canonical paper files

- Canonical manuscript source: `paper/main.tex`
- Canonical local manuscript PDF: `paper/main.pdf`
- Published website PDF: `docs/paper.pdf`
- NeurIPS manuscript source: `paper/main_neurips.tex`
- NeurIPS manuscript PDF: `paper/main_neurips.pdf`

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

## Reproduce the final paper experiments

The current reproduction instructions are in
[reproduction/README.md](reproduction/README.md). They cover both manuscripts,
including the 1,000-repetition small-sample and RePPI comparisons, leave-one-out
diagnostics, tabular benchmarks, LLM evaluation and ranking, and the held-out
MATH calibration example.

```bash
python3 -m venv .venv
.venv/bin/pip install -r reproduction/requirements.txt
./scripts/reproduce_paper.sh --prepare-data
./scripts/reproduce_paper.sh
```

Use `--smoke` for a short execution check, `--figures-only` to regenerate all
15 manuscript figures and tables from the included final summaries, or
`--study small_sample` to run one study. Outputs go to `outputs/reproduction/`.

The standalone code archive is
[reproduction/calibeating-reproduction.zip](reproduction/calibeating-reproduction.zip).
It includes only the experiment dependencies and final summaries, without
pilot runs, screening studies, rebuttal text, raw datasets, or model weights.

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
