from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction import reproduce
from experiments.ppi_mean_reproduction import write_paper_summary_table


def test_plan_covers_current_manuscripts_and_final_settings() -> None:
    manifest = reproduce.verify_asset_manifest()
    assert len(manifest["assets"]) == 15
    studies = {item["name"]: item["study"] for item in manifest["assets"]}
    assert studies["fig_small_n_reppi.pdf"] == "small_sample"
    assert studies["fig_small_n_diagnostics.pdf"] == "small_sample"
    args = type("Args", (), {"smoke": False, "jobs": 8,
                             "ppi_cache_dir": Path("ppi"), "llm_cache_dir": Path("llm")})()
    commands = reproduce.commands(args, Path("results"))
    expected = {"simulation": (500, 20260411), "small_sample": (1000, 20260411),
                "real_data": (500, 20260411), "llm": (200, 20260411),
                "calibration": (500, 20260420)}
    for study, (repetitions, seed) in expected.items():
        command = commands[study]
        assert command[command.index("--replications") + 1] == str(repetitions)
        assert command[command.index("--seed") + 1] == str(seed)
    assert "--no-resume" in commands["llm"]


def test_modified_saved_summary_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    published = tmp_path / "reproduction/published/simulation"
    published.mkdir(parents=True)
    (published / "summary.csv").write_text("changed\n")
    (published.parent / "manifest.json").write_text(json.dumps({
        "files": [{"path": "simulation/summary.csv", "sha256": "not-the-saved-hash"}]}))
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        reproduce.verify_saved_summaries(["simulation"])


def test_table_omits_zero_variance_efficiency_without_changing_results(tmp_path: Path) -> None:
    rows = []
    for n in [50, 100, 200]:
        for method in ["imputation", "ppi"]:
            rows.append({"dataset": "forest_mean", "n_labeled": n, "n_unlabeled": 1000,
                         "estimator": method, "mean_bias": 0.1, "mse": 0.01,
                         "emp_variance": 1e-32 if method == "imputation" else 0.01,
                         "coverage": 0.0, "rel_eff_vs_ppi": 1e30 if method == "imputation" else 1.0})
    frame = pd.DataFrame(rows)
    before = frame.copy(deep=True)
    write_paper_summary_table(frame, tmp_path)
    table = (tmp_path / "table_paper_summary.tex").read_text()
    imputation = [line for line in table.splitlines() if "Imputation" in line]
    assert len(imputation) == 3
    assert all("& --" in line for line in imputation)
    assert "& 1.000" in table
    pd.testing.assert_frame_equal(frame, before)


def test_public_data_downloads_use_fixed_revisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import llm_eval_benchmark, llm_eval_models
    hub = pytest.importorskip("huggingface_hub")
    datasets = pytest.importorskip("datasets")
    calls = {}

    def fake_hub(**kwargs):
        calls["archive"] = kwargs
        return str(tmp_path / "data.zip")

    def fake_dataset(name, **kwargs):
        calls["human"] = kwargs
        return name

    monkeypatch.setattr(hub, "hf_hub_download", fake_hub)
    monkeypatch.setattr(datasets, "load_dataset", fake_dataset)
    llm_eval_models.ensure_ppe_result_archive_path(tmp_path)
    llm_eval_benchmark._load_hf_dataset("lmarena-ai/PPE-Human-Preference-V1", "test")
    assert calls["archive"]["revision"] == llm_eval_models.PPE_RESULT_DATA_REVISION
    assert calls["human"]["revision"] == llm_eval_models.PPE_HUMAN_DATA_REVISION
