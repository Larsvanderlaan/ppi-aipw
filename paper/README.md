# Paper Build Notes

This directory contains the manuscript sources for the public paper release.

## Canonical files

- Canonical manuscript source: [`main.tex`](./main.tex)
- Canonical local PDF: `main.pdf`
- Published website PDF: [`../docs/paper.pdf`](../docs/paper.pdf)
- Secondary submission artifact: [`main_neurips.tex`](./main_neurips.tex)

`main_neurips.tex` is retained as a secondary submission/archive version. The public website and public reproduction flow should be treated as `main.tex`-first.

## Build commands

From the repo root:

```bash
./scripts/reproduce_paper.sh --smoke
```

This refreshes the core non-LLM paper assets and rebuilds `paper/main.pdf`. `paper/latexmkrc` then syncs the result to `docs/paper.pdf`.

If you also want to refresh the LLM-backed paper assets used in `main.tex`:

```bash
./scripts/reproduce_paper.sh --smoke --include-llm-benchmark
```

## Expected outputs

- `paper/main.pdf`: canonical local manuscript PDF
- `docs/paper.pdf`: website copy of the canonical PDF
- `paper/assets/*.pdf` and `paper/assets/*.tex`: manuscript-ready assets referenced by `main.tex`

The reproduction script uses the tracked paper assets already in `paper/assets/` unless you explicitly request the optional LLM refresh path.
