from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "60924986fa23ee53af5500104e5821f16dd3a9e5"


def official_source_dir() -> Path:
    configured = os.environ.get("REPPI_OFFICIAL_DIR")
    if configured:
        return Path(configured)
    return ROOT / "outputs" / "external" / f"RePPI-{COMMIT}"


def test_pinned_official_reppi_source_and_scalar_parity(tmp_path: Path) -> None:
    source_dir = official_source_dir()
    if not (source_dir / "reppi.py").is_file():
        pytest.skip(
            "Pinned external RePPI checkout is absent; run "
            "scripts/fetch_reppi_official.sh."
        )
    report_path = tmp_path / "official_reppi_parity.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_reppi_official_parity.py"),
            "--source-dir",
            str(source_dir),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["source"]["commit"] == COMMIT
    assert (
        report["public_mlp_path"]["maximum_absolute_prediction_error"]
        == pytest.approx(0.0, abs=1e-12)
    )
    assert (
        report["single_fold_scalar_algebra"]["estimate_absolute_error"]
        < 1e-12
    )
    assert report["public_crossfit_wrapper_used_as_production_oracle"] is False
