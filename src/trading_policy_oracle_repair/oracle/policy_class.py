from __future__ import annotations

import numpy as np
import pandas as pd


def select_policy_class_oracle(performance: pd.DataFrame) -> pd.DataFrame:
    """Choose the ex-post best frozen policy separately for each asset."""
    required = {"asset_id", "policy_id", "initial_wealth", "final_wealth"}
    missing = required - set(performance.columns)
    if missing:
        raise ValueError(f"Performance table missing required columns: {sorted(missing)}")
    p = performance.copy()
    p["initial_wealth"] = pd.to_numeric(p["initial_wealth"], errors="raise").astype(float)
    p["final_wealth"] = pd.to_numeric(p["final_wealth"], errors="raise").astype(float)
    if (p["initial_wealth"] <= 0).any() or (p["final_wealth"] <= 0).any():
        raise ValueError("Policy-class oracle requires positive wealth")
    p["log_utility"] = np.log(p["final_wealth"] / p["initial_wealth"])
    # Stable sorting gives deterministic tie-breaking by policy_id.
    p = p.sort_values(["asset_id", "log_utility", "policy_id"], ascending=[True, False, True], kind="mergesort")
    best = p.groupby("asset_id", as_index=False, sort=True).head(1).copy()
    cols = ["asset_id", "policy_id", "initial_wealth", "final_wealth", "log_utility"]
    optional = [c for c in ["CAGR", "MaxDD", "total_cost", "trade_count", "cost_bps_per_side"] if c in best.columns]
    best = best[cols + optional].rename(columns={"policy_id": "oracle_policy_id"})
    best["oracle_type"] = "policy_class"
    return best.reset_index(drop=True)
