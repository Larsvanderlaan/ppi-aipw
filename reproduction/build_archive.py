#!/usr/bin/env python3
"""Package only the code and summaries needed for the paper experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "reproduction/README.md", "reproduction/requirements.txt",
    "reproduction/reproduce.py", "reproduction/build_archive.py",
    "reproduction/assets.json", "reproduction/data.json",
    "experiments/__init__.py", "experiments/estimators.py",
    "experiments/simulate.py", "experiments/plot_results.py",
    "experiments/rebuttal_sample_size.py", "experiments/reppi_baseline_config.json",
    "experiments/paper_simulation_recommendations.json",
    "experiments/ppi_mean_reproduction.py", "experiments/llm_eval_benchmark.py",
    "experiments/llm_eval_models.py", "experiments/toy_calibration_study.py",
    "experiments/calibration_example.py",
    "src/ppi_aipw/_api.py", "src/ppi_aipw/_calibration.py", "src/ppi_aipw/_utils.py",
    "scripts/reproduce_paper.sh", "scripts/audit_reppi_tail.py",
    "scripts/fetch_reppi_official.sh", "scripts/validate_reppi_official_parity.py",
    "paper/README.md", "paper/main.tex", "paper/main_neurips.tex",
    "paper/paper_style.sty", "paper/neurips_2026.sty", "paper/references.bib",
]


def main() -> None:
    files = {name: (ROOT / name).read_bytes() for name in FILES}
    # Match the bibliography used for the checked manuscript builds.
    if (ROOT / ".git").exists():
        files["paper/references.bib"] = subprocess.check_output(
            ["git", "show", "HEAD:paper/references.bib"], cwd=ROOT)
    # Experiments import only this public entry point; the causal and diagnostic
    # APIs are not part of the reproduction dependency tree.
    files["src/ppi_aipw/__init__.py"] = b'"""Mean-estimation entry point used by the paper experiments."""\nfrom ._api import mean_inference\n'
    for path in sorted((ROOT / "reproduction/published").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = path.read_bytes()
    files["README.md"] = b"# Calibeating reproduction\n\nSee reproduction/README.md for setup and experiment commands.\n"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    manifest = {
        "base_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination = ROOT / "reproduction/calibeating-reproduction.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo("calibeating-reproduction/" + name, date_time=(2026, 10, 30, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(entry, data)
    print(f"{destination}: {len(files)} files, {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
