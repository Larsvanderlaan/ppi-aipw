# Reproducing the paper

This code reproduces the experiments in `paper/main.tex` (the extended arXiv
manuscript) and `paper/main_neurips.tex` (the NeurIPS manuscript). The asset
list is checked against both sources before each run.

## Setup

Use Python 3.9–3.12. From the repository or extracted archive:

```bash
python3 -m venv .venv
.venv/bin/pip install -r reproduction/requirements.txt
```

The experiment commands load the included Python sources directly; no package
installation is needed. The LLM experiments use public precomputed evaluator
scores and do not require model weights, PyTorch, or a GPU.

## Run the experiments

```bash
.venv/bin/python reproduction/reproduce.py --prepare-data
.venv/bin/python reproduction/reproduce.py
```

The first command downloads missing public inputs and checks them against
`reproduction/data.json`. Existing caches under `outputs/cache/ppi_datasets`
and `outputs/cache/llm_eval` are reused. Custom cache locations can be supplied
with `--ppi-cache-dir` and `--llm-cache-dir`.

The full run uses the following settings:

| Study | Repetitions | Seed | Design |
|---|---:|---:|---|
| Main simulation | 500 | 20260411 | Both fixed score settings; seven sample sizes; N/n = 1 and 16 |
| Small-sample comparison | 1,000 | 20260411 | Two score settings; n = 20, 50, 100, 200, 400, 800, 1200; N = 16n; all 11 methods, including LOO and both RePPI learners |
| Tabular benchmarks | 500 | 20260411 | Forest, galaxies, and census income; original sample-size grids |
| LLM benchmarks | 200 | 20260411 | PPE Human Preference and five PPE correctness benchmarks; four evaluators; eight human-preference target models |
| MATH calibration example | 500 | 20260420 | ArmoRM; 100 labels for the held-out calibration diagnostic; n = 25, 50, 100, 200 for the interval-width comparison |

The main simulation retains the original ordering of the two score settings
so that the seeded random-number streams match the reported experiment.
The prediction-score settings and RePPI architecture grid are fixed in the
two JSON files under `experiments/`; the reproduction does not tune these
settings against benchmark results.

Results go to `outputs/reproduction/full/experiments/`. The 15 manuscript
figures and tables go to `outputs/reproduction/full/assets/`. The output
manifest checks the complete design, replication counts, unique result keys,
and finite estimates. The full small-sample run also replays the largest
RePPI errors and saves their coefficients and prediction variances.

To run one study or choose an output directory:

```bash
.venv/bin/python reproduction/reproduce.py --study small_sample --jobs 8
.venv/bin/python reproduction/reproduce.py --output-dir /path/to/results
```

The five study names are `simulation`, `small_sample`, `real_data`, `llm`,
and `calibration`. Use `--plan` to inspect the commands before running them.

## Regenerate the published figures and tables

The archive includes the final numerical summaries, with checksums and their
original result paths in `reproduction/published/manifest.json`.

```bash
.venv/bin/python reproduction/reproduce.py --figures-only
```

This regenerates all 15 manuscript assets in
`outputs/reproduction/figures/assets/` from the saved summaries. It does not
rerun the experiments or download data. The full-run command above recomputes
the estimates from the inputs.

## Smoke check

```bash
.venv/bin/python reproduction/reproduce.py --smoke
```

Smoke mode uses two repetitions and reduced grids for the small-sample,
tabular, and LLM studies. Results go to `outputs/reproduction/smoke/`.
These outputs are for checking execution and are not the paper results.
Neither mode changes `paper/assets` or the manuscript PDFs.

## Public data and reference implementation

The tabular data are the public mean-estimation examples from
[ppi_py](https://github.com/aangelopoulos/ppi_py). The LLM inputs come from
[PPE Human Preference V1](https://huggingface.co/datasets/lmarena-ai/PPE-Human-Preference-V1)
and [PPE precomputed results](https://huggingface.co/datasets/lmarena-ai/ppe-result-data).
`data.json` identifies the normalized inputs used for the reported results.
The two PPE downloads are pinned to the revisions listed there.
Raw datasets and model weights are not included in the code archive.

The scalar RePPI implementations and frozen neural-network grid are included.
To repeat the public-code checks described in the appendix:

```bash
bash scripts/fetch_reppi_official.sh
.venv/bin/python scripts/validate_reppi_official_parity.py \
  --source-dir outputs/external/RePPI-60924986fa23ee53af5500104e5821f16dd3a9e5
```

These checks compare the shared neural prediction path and single-fold scalar
calculation with the pinned public source. The experiment's three-way scaling
and aggregation follow Algorithm 1 of the cited paper.

## Build the code archive

```bash
python3 reproduction/build_archive.py
```

This writes `reproduction/calibeating-reproduction.zip` from an explicit file
list. It includes the required experiment modules, mean-estimation library
code, manuscript sources and styles, final summaries, and the reproduction
instructions. It excludes pilot runs, screening studies, rebuttal text, raw
datasets, model weights, alternate manuscripts, and the general R package.
