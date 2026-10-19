import numpy as np

from trading_policy_oracle_repair.synthetic_validation import (
    run_phase12_synthetic_toy_validation,
    run_synthetic_grid,
    summarize_synthetic,
    synthetic_sanity_report,
)


def test_phase12_synthetic_toy_validation():
    out = run_phase12_synthetic_toy_validation()
    assert out["status"] == "PASSED"
    assert out["recoverable_high_signal_gain"] > 0.0
    assert out["nonrecoverable_oracle_gain"] > 0.0
    assert out["nonrecoverable_max_coverage"] <= 0.05


def test_recoverable_signal_strength_improves_recovered_fraction():
    runs = run_synthetic_grid(
        worlds=["recoverable", "clairvoyance_only"],
        signal_strengths=[0.0, 1.0, 4.0],
        replicates=4,
        n_days=1800,
        horizon_days=10,
        cost_bps_per_side=2.0,
        train_fraction=0.60,
        calibration_fraction=0.20,
        alpha=0.10,
        ridge_alpha=1.0,
        seed=91,
    )
    summary = summarize_synthetic(runs)
    report = synthetic_sanity_report(summary)
    assert report["recoverable_high_signal_fraction"] > 0.03
    assert report["recoverable_high_signal_gain"] > report["recoverable_low_signal_gain"]
    assert report["nonrecoverable_max_abs_gain"] < 0.001


def test_clairvoyance_world_has_hindsight_opportunity_without_deployability():
    runs = run_synthetic_grid(
        worlds=["recoverable", "clairvoyance_only"],
        signal_strengths=[0.0, 4.0],
        replicates=3,
        n_days=1600,
        horizon_days=10,
        cost_bps_per_side=2.0,
        train_fraction=0.60,
        calibration_fraction=0.20,
        alpha=0.10,
        ridge_alpha=1.0,
        seed=19,
    )
    non = runs.loc[runs["world"] == "clairvoyance_only"]
    assert float(non["oracle_positive_gain_per_decision"].mean()) > 0.0
    assert float(non["coverage"].max()) <= 0.05
    assert np.isfinite(non["deployed_gain_per_decision"]).all()
