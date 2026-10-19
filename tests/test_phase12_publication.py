from pathlib import Path

import pandas as pd

from trading_policy_oracle_repair.publication import (
    add_support_flags,
    claim_evidence_table,
    plot_synthetic_recoverability,
    support_flag,
    write_table_bundle,
)


def test_support_flag():
    assert support_flag(0.01, 0.03) == "positive_95"
    assert support_flag(-0.04, -0.01) == "negative_95"
    assert support_flag(-0.01, 0.02) == "inconclusive"


def test_claim_map_preserves_negative_meta_result():
    selective = pd.DataFrame({
        "split": ["asset_oos_test"],
        "model_name": ["ridge"],
        "lcb_method": ["empirical"],
        "baseline_policy_id": ["mom_126"],
        "log_utility_improvement_ci_low": [0.001],
        "log_utility_improvement_ci_high": [0.02],
        "mean_actual_advantage_ci_low": [0.01],
        "mean_actual_advantage_ci_high": [0.04],
        "continuous_log_utility_improvement_ci_low": [-0.01],
        "continuous_log_utility_improvement_ci_high": [0.05],
    })
    meta = pd.DataFrame({
        "split": ["same_asset_test"],
        "strategy_type": ["meta_selector"],
        "model_name": ["gradient_boosting"],
        "lcb_method": ["empirical"],
        "log_utility_improvement_ci_low": [-0.03],
        "log_utility_improvement_ci_high": [0.01],
        "continuous_log_utility_improvement_ci_low": [-0.50],
        "continuous_log_utility_improvement_ci_high": [-0.02],
    })
    s = add_support_flags(selective)
    assert s.loc[0, "path_support"] == "positive_95"
    claims = claim_evidence_table(selective, meta)
    row = claims.loc[claims["claim_id"] == "C4_global_meta_selector"].iloc[0]
    assert row["status"] == "not_supported"
    assert "1 continuous meta groups" in row["evidence"]


def test_synthetic_plot_writes_png(tmp_path: Path):
    summary = pd.DataFrame({
        "world": ["recoverable", "recoverable", "clairvoyance_only", "clairvoyance_only"],
        "signal_strength": [0.0, 4.0, 0.0, 4.0],
        "recoverable_fraction_mean": [0.0, 0.2, 0.0, 0.0],
        "recoverable_fraction_std": [0.0, 0.02, 0.0, 0.0],
        "coverage_mean": [0.0, 0.1, 0.0, 0.0],
    })
    paths = plot_synthetic_recoverability(summary, tmp_path, write_pdf=False)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_table_bundle_writes_latex_with_declared_runtime_dependency(tmp_path: Path):
    frame = pd.DataFrame({"policy": ["mom_126"], "estimate": [0.012345]})
    paths = write_table_bundle(
        frame,
        stem="table_dependency_smoke",
        table_dir=tmp_path,
        caption="Dependency smoke test",
        label="tab:dependency_smoke",
        write_latex=True,
    )
    assert len(paths) == 2
    tex = tmp_path / "table_dependency_smoke.tex"
    assert tex.exists()
    body = tex.read_text(encoding="utf-8")
    assert "\\begin{table}" in body
    assert r"mom\_126" in body
