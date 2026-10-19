from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .hashing import sha256_file
from .metrics import summarize_replay
from .policies import load_policies
from .replay import replay_policy


@dataclass(frozen=True)
class Phase4Outputs:
    replay_ledger: Path
    replay_fills: Path
    replay_actions: Path
    performance_summary: Path
    cost_sensitivity: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase3_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase3_*.json"))
    return files[-1] if files else None


def require_phase3_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase4 is not None
    cp = _latest_phase3_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase4.require_phase3_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 3 checkpoint found. Run Phase 0 through Phase 3 first.")
    return cp


def _toy_market() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
        "asset_id": ["TOY"] * 4,
        "open": [100.0, 110.0, 121.0, 121.0],
        "high": [101.0, 122.0, 122.0, 122.0],
        "low": [99.0, 109.0, 120.0, 120.0],
        "close": [100.0, 121.0, 121.0, 121.0],
        "signal_price": [100.0, 121.0, 121.0, 121.0],
        "dividend_cash": [0.0, 0.0, 0.0, 0.0],
        "split_factor": [1.0, 1.0, 1.0, 1.0],
        "cash_daily_return": [0.0, 0.0, 0.0, 0.0],
    })


def run_phase4_toy_validation() -> dict[str, float | str]:
    from .policies import BuyHoldPolicy
    from .execution import run_execution_path

    market = _toy_market()
    policy = BuyHoldPolicy(policy_id="buy_hold")
    replay = replay_policy(
        asset_market=market,
        policy=policy,
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=0.0,
    )
    if replay.fills.empty:
        raise AssertionError("BuyHold toy should create an initial fill")
    first_fill = replay.fills.iloc[0]
    if first_fill["decision_date"] != pd.Timestamp("2026-01-02"):
        raise AssertionError("Replay decision date mismatch")
    if first_fill["execution_date"] != pd.Timestamp("2026-01-05"):
        raise AssertionError("Replay must execute at next open")
    if abs(float(first_fill["execution_price"]) - 110.0) > 1e-12:
        raise AssertionError("Replay next-open execution price mismatch")
    if abs(float(replay.ledger.iloc[1]["wealth_close"]) - 1100.0) > 1e-9:
        raise AssertionError("Replay wealth mismatch on analytical toy")

    schedule = {d: 1.0 for d in market["date"].iloc[:-1]}
    scheduled = run_execution_path(
        asset_market=market,
        target_schedule=schedule,
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=0.0,
    )
    max_diff = float((replay.ledger["wealth_close"] - scheduled.ledger["wealth_close"]).abs().max())
    if max_diff > 1e-9:
        raise AssertionError("Policy replay and Phase-2 execution engine disagree")

    low_cost = replay_policy(
        asset_market=market,
        policy=BuyHoldPolicy(policy_id="buy_hold"),
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=0.0,
    )
    high_cost = replay_policy(
        asset_market=market,
        policy=BuyHoldPolicy(policy_id="buy_hold"),
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=20.0,
    )
    if float(high_cost.ledger.iloc[-1]["wealth_close"]) >= float(low_cost.ledger.iloc[-1]["wealth_close"]):
        raise AssertionError("Higher transaction cost should reduce BuyHold toy wealth")

    return {
        "status": "PASSED",
        "first_execution_open": float(first_fill["execution_price"]),
        "wealth_after_first_close": float(replay.ledger.iloc[1]["wealth_close"]),
        "phase2_equivalence_max_abs_error": max_diff,
        "final_wealth_0bps": float(low_cost.ledger.iloc[-1]["wealth_close"]),
        "final_wealth_20bps": float(high_cost.ledger.iloc[-1]["wealth_close"]),
    }


def build_phase4(cfg: AppConfig, project_root: str | Path) -> Phase4Outputs:
    if cfg.phase != 4 or cfg.phase4 is None:
        raise ValueError("build_phase4 requires phase=4 config")
    root = Path(project_root).resolve()
    require_phase3_checkpoint(root, cfg)
    p4 = cfg.phase4
    if abs(float(p4.primary_cost_bps) - float(cfg.research_contract.cost_bps_per_side)) > 1e-12:
        raise ValueError("phase4.primary_cost_bps must match research_contract.cost_bps_per_side")

    canonical_path = root / cfg.paths.processed_dir / p4.phase1_processed_subdir / "canonical_market.parquet"
    policies_path = root / p4.policies_config
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 canonical market: {canonical_path}")
    if not policies_path.exists():
        raise FileNotFoundError(f"Missing policy config: {policies_path}")

    market = pd.read_parquet(canonical_path)
    available_assets = set(market["asset_id"].astype(str).str.upper().unique())
    assets = [x for x in cfg.raw_data.tickers if x.upper() in available_assets]
    if not assets:
        raise ValueError("No configured risky assets found in canonical market")

    policies = load_policies(policies_path)
    expected = {"buy_hold", "sma_20_100", "mom_126", "donchian_55_20", "rsi_14_30_55", "voltarget_15_20"}
    if set(policies) != expected:
        raise ValueError(f"Phase 4 requires exactly the six frozen canonical policies: {sorted(expected)}")

    all_ledger: list[pd.DataFrame] = []
    all_fills: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for asset_id in assets:
        asset_market = market.loc[market["asset_id"].astype(str).str.upper() == asset_id.upper()].copy()
        asset_market = asset_market.sort_values("date").reset_index(drop=True)
        for policy_id, policy in policies.items():
            result = replay_policy(
                asset_market=asset_market,
                policy=policy,
                initial_wealth=p4.initial_wealth,
                initial_exposure=p4.initial_exposure,
                cost_bps_per_side=p4.primary_cost_bps,
            )
            ledger = result.ledger.copy()
            fills = result.fills.copy()
            actions = result.actions.copy()
            for frame in (ledger, fills, actions):
                frame["cost_bps_per_side"] = float(p4.primary_cost_bps)
            all_ledger.append(ledger)
            all_fills.append(fills)
            all_actions.append(actions)
            row = summarize_replay(
                ledger,
                fills,
                actions,
                annualization_days=p4.annualization_days,
            )
            row["cost_bps_per_side"] = float(p4.primary_cost_bps)
            summaries.append(row)

    replay_ledger = pd.concat(all_ledger, ignore_index=True).sort_values(["asset_id", "policy_id", "date"]).reset_index(drop=True)
    nonempty_fills = [x for x in all_fills if not x.empty]
    if nonempty_fills:
        replay_fills = pd.concat(nonempty_fills, ignore_index=True).sort_values(["asset_id", "policy_id", "execution_date"]).reset_index(drop=True)
    else:
        replay_fills = all_fills[0].copy()
    replay_actions = pd.concat(all_actions, ignore_index=True).sort_values(["asset_id", "policy_id", "date"]).reset_index(drop=True)
    performance = pd.DataFrame(summaries).sort_values(["asset_id", "policy_id"]).reset_index(drop=True)

    sensitivity_asset = p4.sensitivity_asset.strip().upper()
    if sensitivity_asset not in available_assets:
        raise ValueError(f"Sensitivity asset {sensitivity_asset} not found in canonical market")
    sens_market = market.loc[market["asset_id"].astype(str).str.upper() == sensitivity_asset].copy().sort_values("date").reset_index(drop=True)
    sensitivity_rows: list[dict[str, Any]] = []
    for cost_bps in p4.sensitivity_cost_bps:
        for policy_id, policy in policies.items():
            result = replay_policy(
                asset_market=sens_market,
                policy=policy,
                initial_wealth=p4.initial_wealth,
                initial_exposure=p4.initial_exposure,
                cost_bps_per_side=float(cost_bps),
            )
            summary = summarize_replay(result.ledger, result.fills, result.actions, annualization_days=p4.annualization_days)
            summary["cost_bps_per_side"] = float(cost_bps)
            sensitivity_rows.append(summary)
    cost_sensitivity = pd.DataFrame(sensitivity_rows).sort_values(["policy_id", "cost_bps_per_side"]).reset_index(drop=True)

    out_dir = root / cfg.paths.processed_dir / p4.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "replay_ledger.parquet"
    fills_path = out_dir / "replay_fills.parquet"
    actions_path = out_dir / "replay_actions.parquet"
    performance_path = out_dir / "replay_performance.parquet"
    sensitivity_path = out_dir / "cost_sensitivity.parquet"
    audit_path = out_dir / "phase4_audit.json"

    replay_ledger.to_parquet(ledger_path, index=False)
    replay_fills.to_parquet(fills_path, index=False)
    replay_actions.to_parquet(actions_path, index=False)
    performance.to_parquet(performance_path, index=False)
    cost_sensitivity.to_parquet(sensitivity_path, index=False)

    csv_files: list[Path] = []
    if p4.write_csv_summary:
        p_csv = out_dir / "replay_performance.csv"
        s_csv = out_dir / "cost_sensitivity.csv"
        performance.to_csv(p_csv, index=False)
        cost_sensitivity.to_csv(s_csv, index=False)
        csv_files.extend([p_csv, s_csv])

    audit = {
        "phase": 4,
        "scope": "full counterfactual replay and audit outputs only; no oracle, regret labels or ML",
        "assets": assets,
        "policy_ids": sorted(policies),
        "primary_cost_bps_per_side": float(p4.primary_cost_bps),
        "initial_wealth": float(p4.initial_wealth),
        "initial_exposure": float(p4.initial_exposure),
        "rows": {
            "replay_ledger": int(len(replay_ledger)),
            "replay_fills": int(len(replay_fills)),
            "replay_actions": int(len(replay_actions)),
            "performance_summary": int(len(performance)),
            "cost_sensitivity": int(len(cost_sensitivity)),
        },
        "replay_count": int(len(assets) * len(policies)),
        "accounting_max_abs_error": float(replay_ledger["accounting_error"].abs().max()),
        "sensitivity_asset": sensitivity_asset,
        "sensitivity_cost_bps": [float(x) for x in p4.sensitivity_cost_bps],
        "canonical_market_sha256": sha256_file(canonical_path),
        "policies_config_sha256": sha256_file(policies_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase4Outputs(
        replay_ledger=ledger_path,
        replay_fills=fills_path,
        replay_actions=actions_path,
        performance_summary=performance_path,
        cost_sensitivity=sensitivity_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_files=(canonical_path, policies_path),
    )
