from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .hashing import sha256_file
from .policies import (
    BuyHoldPolicy,
    DonchianBreakoutPolicy,
    MomentumPolicy,
    RSIMeanReversionPolicy,
    SMACrossPolicy,
    VolTargetPolicy,
    load_policies,
)
from .policy_runner import evaluate_policy_at, generate_signal_audit


@dataclass(frozen=True)
class Phase3Outputs:
    policy_definitions: Path
    signal_audit: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase2_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase2_*.json"))
    return files[-1] if files else None


def require_phase2_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase3 is not None
    cp = _latest_phase2_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase3.require_phase2_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 2 checkpoint found. Run Phase 0, Phase 1 and Phase 2 first.")
    return cp


def _toy_market(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    dates = pd.bdate_range("2026-01-02", periods=n)
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "asset_id": ["TOY"] * n,
        "open": p,
        "high": p + 0.5,
        "low": np.maximum(p - 0.5, 0.01),
        "close": p,
        "signal_price": p,
    })


def run_phase3_toy_validation() -> dict[str, float | str]:
    rising = _toy_market([100, 101, 102, 103, 104, 105])
    falling = _toy_market([105, 104, 103, 102, 101, 100])

    bh = BuyHoldPolicy()
    a_bh = evaluate_policy_at(policy=bh, full_market=rising, decision_date=rising.iloc[-1]["date"], current_exposure=0.0)
    if a_bh.target_exposure != 1.0:
        raise AssertionError("BuyHold toy failed")

    sma = SMACrossPolicy(short_window=2, long_window=4)
    a_sma = evaluate_policy_at(policy=sma, full_market=rising, decision_date=rising.iloc[-1]["date"], current_exposure=0.0)
    if a_sma.target_exposure != 1.0:
        raise AssertionError("SMA toy failed")

    mom = MomentumPolicy(lookback=3)
    a_mom_up = evaluate_policy_at(policy=mom, full_market=rising, decision_date=rising.iloc[-1]["date"], current_exposure=0.0)
    a_mom_down = evaluate_policy_at(policy=mom, full_market=falling, decision_date=falling.iloc[-1]["date"], current_exposure=1.0)
    if a_mom_up.target_exposure != 1.0 or a_mom_down.target_exposure != 0.0:
        raise AssertionError("Momentum toy failed")

    breakout = DonchianBreakoutPolicy(entry_window=3, exit_window=2)
    a_breakout = evaluate_policy_at(policy=breakout, full_market=rising, decision_date=rising.iloc[-1]["date"], current_exposure=0.0)
    if a_breakout.target_exposure != 1.0:
        raise AssertionError("Donchian entry toy failed")
    a_breakdown = evaluate_policy_at(policy=breakout, full_market=falling, decision_date=falling.iloc[-1]["date"], current_exposure=1.0)
    if a_breakdown.target_exposure != 0.0:
        raise AssertionError("Donchian exit toy failed")

    rsi = RSIMeanReversionPolicy(window=3, entry=30, exit=55)
    a_rsi_low = evaluate_policy_at(policy=rsi, full_market=falling, decision_date=falling.iloc[-1]["date"], current_exposure=0.0)
    a_rsi_high = evaluate_policy_at(policy=rsi, full_market=rising, decision_date=rising.iloc[-1]["date"], current_exposure=1.0)
    if a_rsi_low.target_exposure != 1.0 or a_rsi_high.target_exposure != 0.0:
        raise AssertionError("RSI toy failed")

    vol_market = _toy_market([100, 101, 99, 102, 100, 103])
    vol = VolTargetPolicy(target_ann_vol=0.15, lookback=4, max_exposure=1.0)
    a_vol = evaluate_policy_at(policy=vol, full_market=vol_market, decision_date=vol_market.iloc[-1]["date"], current_exposure=0.0)
    if not 0.0 <= a_vol.target_exposure <= 1.0 or not a_vol.ready:
        raise AssertionError("VolTarget toy failed")

    warm = SMACrossPolicy(short_window=2, long_window=5)
    a_warm = evaluate_policy_at(policy=warm, full_market=rising.iloc[:3], decision_date=rising.iloc[2]["date"], current_exposure=0.25)
    if a_warm.ready or abs(a_warm.target_exposure - 0.25) > 1e-12:
        raise AssertionError("Canonical warmup-hold behavior failed")

    return {
        "status": "PASSED",
        "buy_hold_target": float(a_bh.target_exposure),
        "sma_target": float(a_sma.target_exposure),
        "momentum_up_target": float(a_mom_up.target_exposure),
        "momentum_down_target": float(a_mom_down.target_exposure),
        "donchian_entry_target": float(a_breakout.target_exposure),
        "donchian_exit_target": float(a_breakdown.target_exposure),
        "rsi_oversold_target": float(a_rsi_low.target_exposure),
        "rsi_overbought_target": float(a_rsi_high.target_exposure),
        "voltarget_exposure": float(a_vol.target_exposure),
        "warmup_hold_exposure": float(a_warm.target_exposure),
    }


def build_phase3(cfg: AppConfig, project_root: str | Path) -> Phase3Outputs:
    if cfg.phase != 3 or cfg.phase3 is None:
        raise ValueError("build_phase3 requires phase=3 config")
    root = Path(project_root).resolve()
    require_phase2_checkpoint(root, cfg)
    p3 = cfg.phase3

    canonical_path = root / cfg.paths.processed_dir / p3.phase1_processed_subdir / "canonical_market.parquet"
    policies_path = root / p3.policies_config
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 canonical market: {canonical_path}")
    if not policies_path.exists():
        raise FileNotFoundError(f"Missing policy config: {policies_path}")

    market = pd.read_parquet(canonical_path)
    asset_id = p3.smoke_asset.strip().upper()
    asset = market.loc[market["asset_id"].astype(str).str.upper() == asset_id].copy()
    if asset.empty:
        raise ValueError(f"Smoke asset {asset_id} not found")
    asset = asset.sort_values("date").head(int(p3.smoke_rows)).reset_index(drop=True)

    policies = load_policies(policies_path)
    expected = {"buy_hold", "sma_20_100", "mom_126", "donchian_55_20", "rsi_14_30_55", "voltarget_15_20"}
    if set(policies) != expected:
        raise ValueError(f"Phase 3 requires exactly the six frozen canonical policies: {sorted(expected)}")

    definitions = pd.DataFrame([
        {"policy_id": pid, "definition_json": json.dumps(policy.definition(), sort_keys=True)}
        for pid, policy in policies.items()
    ])
    audits = [
        generate_signal_audit(policy=policy, market=asset, initial_exposure=p3.initial_exposure)
        for policy in policies.values()
    ]
    signal_audit = pd.concat(audits, ignore_index=True).sort_values(["policy_id", "date"]).reset_index(drop=True)

    out_dir = root / cfg.paths.processed_dir / p3.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = out_dir / "policy_definitions.parquet"
    signals_path = out_dir / "policy_signal_smoke.parquet"
    audit_path = out_dir / "phase3_audit.json"
    definitions.to_parquet(definitions_path, index=False)
    signal_audit.to_parquet(signals_path, index=False)

    csv_files: list[Path] = []
    if p3.write_csv_copy:
        d_csv = out_dir / "policy_definitions.csv"
        s_csv = out_dir / "policy_signal_smoke.csv"
        definitions.to_csv(d_csv, index=False)
        signal_audit.to_csv(s_csv, index=False)
        csv_files.extend([d_csv, s_csv])

    summary: dict[str, Any] = {}
    for pid, g in signal_audit.groupby("policy_id"):
        summary[str(pid)] = {
            "rows": int(len(g)),
            "ready_rows": int(g["ready"].sum()),
            "warmup_rows": int((~g["ready"]).sum()),
            "min_target_exposure": float(g["target_exposure"].min()),
            "max_target_exposure": float(g["target_exposure"].max()),
            "target_changes": int((g["target_exposure"].diff().abs().fillna(0.0) > 1e-12).sum()),
        }

    audit = {
        "phase": 3,
        "scope": "policy decision layer only; no performance replay, oracle, regret labels or ML",
        "smoke_asset": asset_id,
        "smoke_rows": int(len(asset)),
        "signal_time": cfg.research_contract.signal_time,
        "execution_time_reserved_for_phase4": cfg.research_contract.execution_time,
        "canonical_warmup_behavior": "hold current exposure",
        "policy_count": int(len(policies)),
        "policy_summary": summary,
        "canonical_market_sha256": sha256_file(canonical_path),
        "policies_config_sha256": sha256_file(policies_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase3Outputs(
        policy_definitions=definitions_path,
        signal_audit=signals_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_files=(canonical_path, policies_path),
    )
