# Manuscripts and builds

- Extended arXiv source: `main.tex`; PDF: `main.pdf`.
- NeurIPS source: `main_neurips.tex`; PDF: `main_neurips.pdf`.
- Website PDF: `../docs/paper.pdf`, identical to `main.pdf`.

The NeurIPS source currently uses anonymous review formatting and a local
style file. It requires the official style, final mode, and author block
before a camera-ready submission.

In the standalone archive, regenerate and stage the figures first:

```bash
.venv/bin/python reproduction/reproduce.py --figures-only
mkdir -p paper/assets
cp outputs/reproduction/figures/assets/* paper/assets/
```

To build either source without changing experimental results:

```bash
cd paper
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error main_neurips.tex
```

To reproduce the experiments and their 15 figures and tables, see
[`../reproduction/README.md`](../reproduction/README.md). The reproduction
commands write to `outputs/reproduction/`; they do not replace the published
paper assets or PDFs. The saved final summaries are included for regenerating
the figures without rerunning the experiments.
