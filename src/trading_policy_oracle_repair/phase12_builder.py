from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .hashing import sha256_file
from .publication import (
    build_results_readme,
    claim_evidence_table,
    compact_meta_table,
    compact_selective_table,
    plot_annual_heatmaps,
    plot_continuous_relative_wealth,
    plot_coverage_harm,
    plot_forest,
    plot_prediction_deciles,
    paper_meta_table,
    paper_selective_table,
    paper_synthetic_table,
    plot_switching_frontier,
    plot_synthetic_recoverability,
    write_publication_manifest,
    write_table_bundle,
    zip_publication_bundle,
)
from .synthetic_validation import (
    run_phase12_synthetic_toy_validation,
    run_synthetic_grid,
    summarize_synthetic,
    synthetic_sanity_report,
)


@dataclass(frozen=True)
class Phase12Outputs:
    synthetic_runs: Path
    synthetic_summary: Path
    claim_evidence: Path
    compact_selective_repair: Path
    compact_meta: Path
    publication_manifest: Path
    publication_bundle_dir: Path
    publication_bundle_zip: Path
    audit_json: Path
    figure_files: tuple[Path, ...]
    table_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase11_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase11_*.json"))
    return files[-1] if files else None


def require_phase11_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase12 is not None
    cp = _latest_phase11_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase12.require_phase11_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 11 checkpoint found. Run/copy frozen Phase 11 outputs first.")
    return cp


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Phase-12 input: {path}")
    return pd.read_parquet(path)


def _index_frame(paths: list[Path], base: Path, kind: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "kind": kind,
            "artifact": str(p.relative_to(base)),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        }
        for p in paths if p.exists()
    ])


def _input_contract(cfg: AppConfig, root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    assert cfg.phase12 is not None
    p12 = cfg.phase12
    p11dir = root / cfg.paths.processed_dir / p12.phase11_processed_subdir
    p10dir = root / cfg.paths.processed_dir / p12.phase10_processed_subdir
    required = {
        "publication_selective_repair": p11dir / "publication_selective_repair.parquet",
        "publication_meta": p11dir / "publication_meta.parquet",
        "annual_stability": p11dir / "annual_stability.parquet",
        "asset_stability": p11dir / "asset_stability.parquet",
        "continuous_bootstrap_inference": p11dir / "continuous_bootstrap_inference.parquet",
        "continuous_ledger": p10dir / "continuous_ledger.parquet",
        "selective_repair_oos": p10dir / "selective_repair_oos.parquet",
    }
    optional = {
        "phase9_coverage_harm": root / cfg.paths.processed_dir / p12.phase9_processed_subdir / "coverage_harm_summary.parquet",
        "phase8_prediction_deciles": root / cfg.paths.processed_dir / p12.phase8_processed_subdir / "prediction_deciles.parquet",
    }
    return required, optional


def run_phase12_toy_validation() -> dict[str, Any]:
    return run_phase12_synthetic_toy_validation()


def build_phase12(cfg: AppConfig, root: Path) -> Phase12Outputs:
    if cfg.phase != 12 or cfg.phase12 is None:
        raise ValueError("build_phase12 requires phase=12 config")
    p12 = cfg.phase12
    cp = require_phase11_checkpoint(root, cfg)
    logger = logging.getLogger("trading_policy_oracle_repair")

    required_paths, optional_paths = _input_contract(cfg, root)
    required_frames = {name: _read_required(path) for name, path in required_paths.items()}
    optional_frames: dict[str, pd.DataFrame] = {}
    for name, path in optional_paths.items():
        if path.exists():
            optional_frames[name] = pd.read_parquet(path)
        else:
            optional_frames[name] = pd.DataFrame()

    # -------- Synthetic ground-truth validation --------
    logger.info("Phase 12 synthetic ground-truth grid")
    synthetic_runs = run_synthetic_grid(
        worlds=p12.synthetic_worlds,
        signal_strengths=p12.synthetic_signal_strengths,
        replicates=p12.synthetic_replicates,
        n_days=p12.synthetic_n_days,
        horizon_days=p12.synthetic_horizon_days,
        cost_bps_per_side=p12.synthetic_cost_bps_per_side,
        train_fraction=p12.synthetic_train_fraction,
        calibration_fraction=p12.synthetic_calibration_fraction,
        alpha=p12.synthetic_alpha,
        ridge_alpha=p12.synthetic_ridge_alpha,
        seed=cfg.seed + 120000,
    )
    synthetic_summary = summarize_synthetic(synthetic_runs)
    synthetic_report = synthetic_sanity_report(synthetic_summary)
    if synthetic_report["recoverable_high_signal_gain"] <= 0.0:
        raise AssertionError("Phase-12 recoverable world did not produce positive deployable gain")
    if synthetic_report["nonrecoverable_oracle_gain"] <= 0.0:
        raise AssertionError("Phase-12 clairvoyance-only world lacks hindsight opportunity; sanity test is invalid")
    if synthetic_report["nonrecoverable_max_coverage"] > float(p12.max_nonrecoverable_coverage):
        raise AssertionError("Phase-12 clairvoyance-only gate deploys too often; inspect leakage/overfitting")

    # -------- Frozen real-data publication tables --------
    selective_raw = required_frames["publication_selective_repair"]
    meta_raw = required_frames["publication_meta"]
    annual = required_frames["annual_stability"]
    assets = required_frames["asset_stability"]
    continuous_inf = required_frames["continuous_bootstrap_inference"]
    continuous_ledger = required_frames["continuous_ledger"]
    repair_oos = required_frames["selective_repair_oos"]
    selective = compact_selective_table(selective_raw)
    meta = compact_meta_table(meta_raw)
    claims = claim_evidence_table(selective_raw, meta_raw)

    processed_dir = root / cfg.paths.processed_dir / p12.processed_subdir
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_paths = {
        "synthetic_runs": processed_dir / "synthetic_runs.parquet",
        "synthetic_summary": processed_dir / "synthetic_summary.parquet",
        "claims": processed_dir / "claim_evidence.parquet",
        "selective": processed_dir / "publication_selective_repair_compact.parquet",
        "meta": processed_dir / "publication_meta_compact.parquet",
        "audit": processed_dir / "phase12_audit.json",
    }
    for path in processed_paths.values():
        if path.exists():
            path.unlink()
    synthetic_runs.to_parquet(processed_paths["synthetic_runs"], index=False)
    synthetic_summary.to_parquet(processed_paths["synthetic_summary"], index=False)
    claims.to_parquet(processed_paths["claims"], index=False)
    selective.to_parquet(processed_paths["selective"], index=False)
    meta.to_parquet(processed_paths["meta"], index=False)

    # -------- Publication bundle --------
    bundle_dir = root / cfg.paths.outputs_dir / p12.publication_output_subdir
    _clean_dir(bundle_dir)
    table_dir = bundle_dir / "tables"
    figure_dir = bundle_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table_files: list[Path] = []
    table_files += write_table_bundle(
        paper_selective_table(selective_raw), stem="table_main_selective_repair", table_dir=table_dir,
        caption="Selective repair: paired block-bootstrap inference and decision evidence.",
        label="tab:selective_repair", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        paper_meta_table(meta_raw), stem="table_meta_stress_test", table_dir=table_dir,
        caption="Meta-policy selection and naive meta stress tests.",
        label="tab:meta_stress", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        paper_synthetic_table(synthetic_summary), stem="table_synthetic_validation", table_dir=table_dir,
        caption="Synthetic recoverable and clairvoyance-only ground-truth validation.",
        label="tab:synthetic", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        claims, stem="table_claim_evidence_map", table_dir=table_dir,
        caption="Automatically generated mapping from manuscript claims to frozen evidence.",
        label="tab:claim_evidence", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        annual, stem="table_annual_stability", table_dir=table_dir,
        caption="Annual stability of selective-repair utility improvements.",
        label="tab:annual_stability", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        assets, stem="table_asset_stability", table_dir=table_dir,
        caption="Asset-level stability of selective-repair utility improvements.",
        label="tab:asset_stability", write_latex=p12.write_latex_tables,
    )
    table_files += write_table_bundle(
        continuous_inf, stem="table_continuous_robustness", table_dir=table_dir,
        caption="Continuous 2016-2025 pseudo-live paired block-bootstrap robustness.",
        label="tab:continuous_robustness", write_latex=p12.write_latex_tables,
    )

    figure_files: list[Path] = []
    figure_files += plot_synthetic_recoverability(synthetic_summary, figure_dir, write_pdf=p12.write_pdf_figures)
    figure_files += plot_forest(
        selective_raw, metric_prefix="log_utility_improvement",
        title="Selective repair: post-cost OOS utility", stem_prefix="forest_oos_utility",
        figure_dir=figure_dir, write_pdf=p12.write_pdf_figures,
    )
    figure_files += plot_forest(
        selective_raw, metric_prefix="mean_actual_advantage",
        title="Selective repair: realized continuation advantage", stem_prefix="forest_decision_advantage",
        figure_dir=figure_dir, write_pdf=p12.write_pdf_figures,
    )
    figure_files += plot_forest(
        selective_raw, metric_prefix="continuous_log_utility_improvement",
        title="Continuous pseudo-live utility improvement", stem_prefix="forest_continuous_utility",
        figure_dir=figure_dir, write_pdf=p12.write_pdf_figures,
    )
    figure_files += plot_annual_heatmaps(annual, figure_dir, write_pdf=p12.write_pdf_figures)
    figure_files += plot_switching_frontier(repair_oos, figure_dir, write_pdf=p12.write_pdf_figures)
    figure_files += plot_continuous_relative_wealth(
        continuous_ledger,
        baselines=p12.continuous_figure_baselines,
        models=p12.continuous_figure_models,
        lcb_method=p12.continuous_figure_lcb_method,
        figure_dir=figure_dir,
        write_pdf=p12.write_pdf_figures,
    )
    if not optional_frames["phase9_coverage_harm"].empty:
        figure_files += plot_coverage_harm(optional_frames["phase9_coverage_harm"], figure_dir, write_pdf=p12.write_pdf_figures)
    if not optional_frames["phase8_prediction_deciles"].empty:
        figure_files += plot_prediction_deciles(optional_frames["phase8_prediction_deciles"], figure_dir, write_pdf=p12.write_pdf_figures)

    optional_present = {k: bool(path.exists()) for k, path in optional_paths.items()}
    readme = build_results_readme(
        selective=selective_raw, meta=meta_raw, synthetic_report=synthetic_report,
        optional_inputs_present=optional_present,
    )
    quality = "publication-grade-required" if cfg.research_contract.publication_data_quality_required else "exploratory-data"
    readme += f"\n## Data-quality flag\n\n`{quality}`. The current research contract sets `publication_data_quality_required={str(cfg.research_contract.publication_data_quality_required).lower()}`. This bundle is publication-formatted evidence; it is not a certification that the raw market source is point-in-time institutional data.\n"
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")

    # Indices are themselves publication artifacts and make bundle inspection deterministic.
    table_index = _index_frame(table_files, bundle_dir, "table")
    figure_index = _index_frame(figure_files, bundle_dir, "figure")
    table_index_path = bundle_dir / "table_index.csv"
    figure_index_path = bundle_dir / "figure_index.csv"
    table_index.to_csv(table_index_path, index=False)
    figure_index.to_csv(figure_index_path, index=False)
    table_files += [table_index_path, figure_index_path]

    all_input_paths = list(required_paths.values()) + [p for p in optional_paths.values() if p.exists()]
    manifest = write_publication_manifest(
        bundle_dir,
        input_files=all_input_paths,
        metadata={
            "phase": 12,
            "scope": "synthetic validation plus frozen publication bundle",
            "phase11_checkpoint": str(cp.relative_to(root)) if cp is not None else None,
            "real_data_models_fit": False,
            "real_data_thresholds_tuned": False,
            "synthetic_model": "Ridge student used only for controlled ground-truth validation",
            "data_quality_flag": quality,
            "synthetic_sanity_report": synthetic_report,
            "optional_inputs_present": optional_present,
        },
    )
    bundle_zip = zip_publication_bundle(bundle_dir, root / cfg.paths.outputs_dir / p12.publication_zip_stem)

    audit = {
        "phase": 12,
        "scope": "final synthetic validation and publication bundle; no real-data model modification",
        "phase11_checkpoint": str(cp.relative_to(root)) if cp is not None else None,
        "real_data_models_fit": False,
        "real_data_hyperparameters_tuned": False,
        "real_data_thresholds_retuned": False,
        "phase10_11_outcomes_frozen": True,
        "synthetic_worlds": list(p12.synthetic_worlds),
        "synthetic_signal_strengths": [float(x) for x in p12.synthetic_signal_strengths],
        "synthetic_replicates": int(p12.synthetic_replicates),
        "synthetic_n_days": int(p12.synthetic_n_days),
        "synthetic_horizon_days": int(p12.synthetic_horizon_days),
        "synthetic_sanity_report": synthetic_report,
        "publication_bundle": str(bundle_dir.relative_to(root)),
        "publication_bundle_zip": str(bundle_zip.relative_to(root)),
        "figure_count": int(len(figure_files)),
        "table_artifact_count": int(len(table_files)),
        "optional_inputs_present": optional_present,
        "data_quality_flag": quality,
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in all_input_paths},
    }
    processed_paths["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase12Outputs(
        synthetic_runs=processed_paths["synthetic_runs"],
        synthetic_summary=processed_paths["synthetic_summary"],
        claim_evidence=processed_paths["claims"],
        compact_selective_repair=processed_paths["selective"],
        compact_meta=processed_paths["meta"],
        publication_manifest=manifest,
        publication_bundle_dir=bundle_dir,
        publication_bundle_zip=bundle_zip,
        audit_json=processed_paths["audit"],
        figure_files=tuple(figure_files),
        table_files=tuple(table_files),
        input_files=tuple(all_input_paths),
    )
