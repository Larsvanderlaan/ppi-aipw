#!/usr/bin/env python3
"""Reproduce the experiments and figures in the two manuscript versions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("simulation", "small_sample", "real_data", "llm", "calibration")
EVALUATORS = (
    "skywork_reward_llama3_1_8b", "armorm_llama3_8b_v0_1",
    "athene_rm_8b", "starling_rm_7b_alpha",
)
SEED = 20260411


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def required_assets() -> dict[str, list[str]]:
    """Read the asset names directly from each manuscript."""
    result = {}
    for name in ("main.tex", "main_neurips.tex"):
        source = (ROOT / "paper" / name).read_text()
        result[name] = sorted(set(re.findall(r"assets/([^{}]+\.(?:pdf|tex))", source)))
    return result


def verify_asset_manifest() -> dict:
    manifest = load_json(ROOT / "reproduction/assets.json")
    required = set().union(*map(set, required_assets().values()))
    supplied = {asset["name"] for asset in manifest["assets"]}
    if required != supplied:
        raise ValueError(f"Asset manifest differs from manuscripts: missing={required-supplied}, extra={supplied-required}")
    return manifest


def commands(args: argparse.Namespace, results: Path) -> dict[str, list[str]]:
    """Return commands with the published settings or reduced smoke settings."""
    py = sys.executable
    experiment = ROOT / "experiments"
    recommendations = str(experiment / "paper_simulation_recommendations.json")
    small = [py, str(experiment / "simulate.py"), "--rebuttal-sample-size",
             "--replications", "2" if args.smoke else "1000", "--seed", str(SEED),
             "--jobs", str(args.jobs), "--recommendations-file", recommendations,
             "--output-dir", str(results / "small_sample")]
    if args.smoke:
        small += ["--n-grid", "20", "50"]
    real = [py, str(experiment / "ppi_mean_reproduction.py"),
            "--replications", "2" if args.smoke else "500", "--seed", str(SEED),
            "--cache-dir", str(args.ppi_cache_dir), "--output-dir", str(results / "real_data")]
    llm = [py, str(experiment / "llm_eval_benchmark.py"),
           "--replications", "2" if args.smoke else "200", "--seed", str(SEED),
           "--cache-dir", str(args.llm_cache_dir), "--output-dir", str(results / "llm"),
           "--tracks", "ppe_human", "ppe_correctness", "--evaluators",
           *(EVALUATORS[1:2] if args.smoke else EVALUATORS), "--no-resume"]
    if args.smoke:
        real += ["--smoke"]
        llm += ["--smoke"]
    return {
        # Both score settings retain the original order of random-number streams.
        "simulation": [py, str(experiment / "simulate.py"), "--profile", "paper",
                       "--replications", "2" if args.smoke else "500", "--seed", str(SEED),
                       "--recommendations-file", recommendations,
                       "--output-dir", str(results / "simulation")],
        "small_sample": small,
        "real_data": real,
        "llm": llm,
        "calibration": [py, str(experiment / "toy_calibration_study.py"),
                        "--replications", "2" if args.smoke else "500", "--seed", "20260420",
                        "--llm-cache-dir", str(args.llm_cache_dir),
                        "--dataset-cache-dir", str(args.ppi_cache_dir),
                        "--output-dir", str(results / "calibration")],
    }


def verify_inputs(args: argparse.Namespace, studies: list[str]) -> list[dict]:
    """Verify downloaded or existing inputs against the published data hashes."""
    import pandas as pd

    manifest = load_json(ROOT / "reproduction/data.json")
    reports = []
    for item in manifest["files"]:
        if not set(item["studies"]).intersection(studies):
            continue
        cache = args.ppi_cache_dir if item["kind"] == "npz" else args.llm_cache_dir
        path = cache / item["name"]
        if item["kind"] == "npz":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            with gzip.open(path, "rt") as stream:
                frame = pd.read_json(stream, lines=True, dtype={"unit_id": str})
            canonical = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
            digest = hashlib.sha256(canonical).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Data differ from the published inputs: {path}")
        reports.append({"file": item["name"], "sha256": digest})
    return reports


def prepare_inputs(args: argparse.Namespace, studies: list[str]) -> list[dict]:
    """Obtain the public datasets and precomputed evaluator scores used in the paper."""
    if "real_data" in studies:
        from experiments.ppi_mean_reproduction import ensure_dataset_cached
        for name in ("forest", "galaxies", "census_income"):
            ensure_dataset_cached(args.ppi_cache_dir, name)
    if set(studies).intersection(("llm", "calibration")):
        from experiments.llm_eval_benchmark import ensure_track_rows
        from experiments.llm_eval_models import resolve_score_frame
        tracks = ("ppe_human", "ppe_correctness") if "llm" in studies else ("ppe_correctness",)
        evaluators = EVALUATORS if "llm" in studies else EVALUATORS[1:2]
        for track in tracks:
            rows = ensure_track_rows(track, args.llm_cache_dir)
            for evaluator in evaluators:
                resolve_score_frame(rows=rows, cache_dir=args.llm_cache_dir,
                                    track=track, evaluator=evaluator)
    return verify_inputs(args, studies)


def verify_saved_summaries(studies: list[str]) -> None:
    """Reject modified or missing published summaries before plotting."""
    published = ROOT / "reproduction/published"
    manifest = load_json(published / "manifest.json")
    for item in manifest["files"]:
        if Path(item["path"]).parts[0] in studies:
            digest = hashlib.sha256((published / item["path"]).read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Published summary checksum differs: {item['path']}")


def verify_run(study: str, directory: Path, smoke: bool) -> dict:
    """Check the complete design, unique replications, and finite estimates."""
    import numpy as np
    import pandas as pd

    if study == "calibration":
        payload = load_json(directory / "toy_calibration_summary.json")
        if len(payload["monte_carlo"]) != 12:
            raise ValueError("Calibration example did not complete all four sample sizes")
        return {"summary_rows": 12, "reference_fit_n": payload["reference"]["reference_fit_n"]}
    specs = {
        "simulation": ("raw_results.csv", ["score_quality", "n_labeled", "unlabeled_ratio", "estimator"], 308, 308, 500),
        "small_sample": ("raw_results.csv", ["score_quality", "n_labeled", "estimator"], 154, 44, 1000),
        "real_data": ("raw_results.csv", ["dataset", "n_labeled", "estimator"], 340, 68, 500),
        "llm": ("llm_eval_raw_results.csv", ["track", "benchmark", "evaluator", "target_model", "n_labeled", "estimator"], 5320, 436, 200),
    }
    name, groups, full_groups, smoke_groups, repetitions = specs[study]
    frame = pd.read_csv(directory / name, usecols=[*groups, "replicate", "estimate", "se"])
    counts = frame.groupby(groups)["replicate"].nunique()
    expected = 2 if smoke else repetitions
    if len(counts) != (smoke_groups if smoke else full_groups) or not (counts == expected).all():
        raise ValueError(f"Incomplete {study} design: groups={len(counts)}, repetitions={counts.unique()}")
    if frame.duplicated([*groups, "replicate"]).any():
        raise ValueError(f"Duplicate {study} replications")
    if not np.isfinite(frame[["estimate", "se"]]).all().all():
        raise ValueError(f"Nonfinite {study} estimates or standard errors")
    if study == "llm":
        status = load_json(directory / "run_status.json")
        expected_stages = 22 if smoke else 112
        if not status["complete"] or status["completed_stage_count"] != expected_stages:
            raise ValueError(f"Incomplete LLM stages: {status}")
    return {"rows": len(frame), "groups": len(counts), "replications": expected}


def make_figures(study: str, source: Path, output: Path) -> None:
    """Generate only manuscript figures and tables from a study's saved summaries."""
    import pandas as pd

    output.mkdir(parents=True, exist_ok=True)
    if study == "simulation":
        from experiments.plot_results import plot_simulation_main
        plot_simulation_main(pd.read_csv(source / "summary.csv", float_precision="round_trip"), output / "sim_fig_main.pdf")
    elif study == "small_sample":
        from experiments.rebuttal_sample_size import plot_rebuttal_figures
        plot_rebuttal_figures(pd.read_csv(source / "summary.csv", float_precision="round_trip"), output)
    elif study == "real_data":
        from experiments.ppi_mean_reproduction import (
            plot_main_text_grid, plot_appendix_calibration_grid,
            plot_overall_metric_grid, write_paper_summary_table,
        )
        summary = pd.read_csv(source / "summary.csv", float_precision="round_trip")
        plot_main_text_grid(summary, output)
        plot_appendix_calibration_grid(summary, output)
        plot_overall_metric_grid(summary, output)
        write_paper_summary_table(summary, output)
    elif study == "llm":
        from experiments.llm_eval_benchmark import (
            plot_llm_eval_main, plot_llm_eval_by_evaluator, plot_ppe_ranking,
            write_llm_summary_table, write_ppe_ranking_table,
        )
        summary = pd.read_csv(source / "llm_eval_summary_main.csv", float_precision="round_trip")
        ranking = pd.read_csv(source / "llm_eval_ppe_ranking_summary.csv", float_precision="round_trip")
        plot_llm_eval_main(summary, output / "fig_llm_eval_main.pdf")
        plot_llm_eval_by_evaluator(pd.read_csv(source / "llm_eval_summary_by_evaluator.csv", float_precision="round_trip"),
                                  output / "fig_llm_eval_by_evaluator.pdf")
        plot_ppe_ranking(ranking, output / "fig_llm_ppe_ranking.pdf")
        write_llm_summary_table(summary, output)
        write_ppe_ranking_table(ranking, output)
    else:
        from experiments.toy_calibration_study import ToyStudyConfig, plot_toy_study
        payload = load_json(source / "toy_calibration_summary.json")
        config = dict(payload["config"])
        config["n_grid"] = tuple(config["n_grid"])
        config["llm_cache_dir"] = Path(config["llm_cache_dir"])
        config["dataset_cache_dir"] = Path(config["dataset_cache_dir"])
        plot_toy_study(pd.DataFrame(payload["monte_carlo"]), payload["reference"],
                       ToyStudyConfig(**config), output / "fig_toy_calibration_ppi.pdf")


def stage_assets(manifest: dict, studies: list[str], results: Path, output: Path) -> list[dict]:
    """Stage the exact set of manuscript assets produced by the selected studies."""
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for item in manifest["assets"]:
        if item["study"] not in studies:
            continue
        source = results / item["study"] / item["name"]
        destination = output / item["name"]
        shutil.copy2(source, destination)
        reports.append({"name": item["name"], "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run two repetitions with reduced small-sample, real-data, and LLM grids.")
    parser.add_argument("--study", choices=("all", *STUDIES), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ppi-cache-dir", type=Path, default=ROOT / "outputs/cache/ppi_datasets")
    parser.add_argument("--llm-cache-dir", type=Path, default=ROOT / "outputs/cache/llm_eval")
    parser.add_argument("--jobs", type=int, default=8, help="Workers for the small-sample experiment.")
    parser.add_argument("--prepare-data", action="store_true", help="Download and verify inputs, then stop.")
    parser.add_argument("--figures-only", action="store_true", help="Use the included final summaries to regenerate figures and tables.")
    parser.add_argument("--plan", action="store_true", help="Print the commands and asset list without running experiments.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.figures_only and (args.smoke or args.prepare_data):
        parser.error("--figures-only cannot be combined with --smoke or --prepare-data")
    args.ppi_cache_dir = args.ppi_cache_dir.resolve()
    args.llm_cache_dir = args.llm_cache_dir.resolve()
    studies = list(STUDIES) if args.study == "all" else [args.study]
    mode = "figures" if args.figures_only else "smoke" if args.smoke else "full"
    output = (args.output_dir or ROOT / "outputs/reproduction" / mode).resolve()
    results = output / "experiments"
    manifest = verify_asset_manifest()
    run_commands = commands(args, results)
    if args.plan:
        print(json.dumps({"mode": mode, "output": str(output), "commands":
                          {s: run_commands[s] for s in studies}, "assets":
                          [a["name"] for a in manifest["assets"] if a["study"] in studies]}, indent=2))
        return
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    output.mkdir(parents=True, exist_ok=True)
    # Bound numerical libraries before importing the scientific dependencies.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    if args.figures_only:
        inputs = []
        verify_saved_summaries(studies)
    else:
        inputs = prepare_inputs(args, studies)
        if args.prepare_data:
            print(json.dumps({"status": "inputs verified", "files": inputs}, indent=2))
            return
    checks = {}
    for study in studies:
        print(f"{study}: {'plotting saved results' if args.figures_only else 'running experiment'}", flush=True)
        if not args.figures_only:
            subprocess.run(run_commands[study], cwd=ROOT, check=True)
            checks[study] = verify_run(study, results / study, args.smoke)
            if study == "small_sample" and not args.smoke:
                subprocess.run([sys.executable, str(ROOT / "scripts/audit_reppi_tail.py"),
                                "--raw-results", str(results / study / "raw_results.csv"),
                                "--output-prefix", str(results / study / "reppi_tail_audit")],
                               cwd=ROOT, check=True)
        source = ROOT / "reproduction/published" / study if args.figures_only else results / study
        make_figures(study, source, results / study)
    staged = stage_assets(manifest, studies, results, output / "assets")
    report = {"status": "complete", "mode": mode, "studies": studies,
              "commands": {s: run_commands[s] for s in studies}, "inputs": inputs,
              "checks": checks, "assets": staged, "manuscripts": required_assets()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
