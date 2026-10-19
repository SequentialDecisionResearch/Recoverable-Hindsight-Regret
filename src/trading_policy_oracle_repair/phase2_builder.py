from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .execution import run_execution_path
from .hashing import sha256_file


@dataclass(frozen=True)
class Phase2Outputs:
    smoke_ledger: Path
    smoke_fills: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase1_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase1_*.json"))
    return files[-1] if files else None


def require_phase1_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase2 is not None
    cp = _latest_phase1_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase2.require_phase1_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 1 checkpoint found. Run Phase 0 and Phase 1 first.")
    return cp


def run_phase2_toy_validation() -> dict[str, float | str]:
    """Hand-checkable execution toy covering timing, gap, split, dividend and cost."""
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    market = pd.DataFrame({
        "date": dates,
        "asset_id": ["TOY"] * 4,
        "open": [100.0, 110.0, 60.5, 60.5],
        "close": [100.0, 121.0, 60.5, 60.5],
        "dividend_cash": [0.0, 0.0, 0.0, 1.0],
        "split_factor": [1.0, 1.0, 2.0, 1.0],
        "cash_daily_return": [0.0, 0.0, 0.0, 0.0],
    })
    # The only requested trade is chosen at close Jan 2 and must execute Jan 5 open.
    schedule = {dates[0]: 1.0}
    result = run_execution_path(
        asset_market=market,
        target_schedule=schedule,
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=0.0,
    )
    ledger, fills = result.ledger, result.fills
    if len(fills) != 1:
        raise AssertionError("Toy should contain exactly one trade")
    if fills.loc[0, "execution_date"] != pd.Timestamp("2026-01-05"):
        raise AssertionError("close-t action did not execute at next open")
    if abs(float(fills.loc[0, "execution_price"]) - 110.0) > 1e-12:
        raise AssertionError("Toy next-open execution price mismatch")
    if abs(float(ledger.loc[1, "wealth_close"]) - 1100.0) > 1e-10:
        raise AssertionError("Toy overnight/intraday wealth mismatch")
    # 2-for-1 split with price halving must preserve economic value.
    if abs(float(ledger.loc[2, "wealth_pre_trade"]) - 1100.0) > 1e-10:
        raise AssertionError("Toy split changed economic value")
    # Final $1 dividend on 18.1818... post-split shares.
    expected_final = 1100.0 + (1000.0 / 110.0) * 2.0 * 1.0
    if abs(float(ledger.loc[3, "wealth_close"]) - expected_final) > 1e-10:
        raise AssertionError("Toy dividend accounting mismatch")

    # Cost boundary: constant price purchase at 5 bps cannot create wealth.
    cost_market = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-02", "2026-02-03"]),
        "asset_id": ["COST"] * 2,
        "open": [100.0, 100.0],
        "close": [100.0, 100.0],
        "dividend_cash": [0.0, 0.0],
        "split_factor": [1.0, 1.0],
        "cash_daily_return": [0.0, 0.0],
    })
    cost_result = run_execution_path(
        asset_market=cost_market,
        target_schedule={pd.Timestamp("2026-02-02"): 1.0},
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=5.0,
    )
    final_cost_wealth = float(cost_result.ledger.iloc[-1]["wealth_close"])
    if not final_cost_wealth < 1000.0:
        raise AssertionError("Transaction cost toy failed")

    return {
        "status": "PASSED",
        "first_execution_open": float(fills.loc[0, "execution_price"]),
        "wealth_after_gap_and_close": float(ledger.loc[1, "wealth_close"]),
        "wealth_after_split_at_open": float(ledger.loc[2, "wealth_pre_trade"]),
        "wealth_after_dividend": float(ledger.loc[3, "wealth_close"]),
        "constant_price_wealth_after_5bps": final_cost_wealth,
    }


def _smoke_schedule(dates: pd.Series) -> dict[pd.Timestamp, float]:
    """Exogenous schedule used only to exercise the engine; this is not a policy."""
    d = list(pd.to_datetime(dates).dt.normalize())
    if len(d) < 5:
        raise ValueError("Smoke window requires at least 5 rows")
    idx1 = 0
    idx2 = max(1, len(d) // 3)
    idx3 = max(idx2 + 1, (2 * len(d)) // 3)
    idx3 = min(idx3, len(d) - 2)  # every action must have a next open
    return {d[idx1]: 1.0, d[idx2]: 0.5, d[idx3]: 0.0}


def build_phase2(cfg: AppConfig, project_root: str | Path) -> Phase2Outputs:
    if cfg.phase != 2 or cfg.phase2 is None:
        raise ValueError("build_phase2 requires phase=2 config")

    root = Path(project_root).resolve()
    require_phase1_checkpoint(root, cfg)
    p2 = cfg.phase2

    canonical_path = root / cfg.paths.processed_dir / p2.phase1_processed_subdir / "canonical_market.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 canonical market: {canonical_path}")
    market = pd.read_parquet(canonical_path)

    smoke_asset = p2.smoke_asset.strip().upper()
    asset = market.loc[market["asset_id"].astype(str).str.upper() == smoke_asset].copy()
    if asset.empty:
        available = sorted(market["asset_id"].astype(str).unique().tolist())
        raise ValueError(f"Smoke asset {smoke_asset} not found. Available: {available}")
    asset = asset.sort_values("date").head(int(p2.smoke_rows)).reset_index(drop=True)
    if len(asset) < 5:
        raise ValueError("Not enough rows for Phase 2 smoke execution")

    schedule = _smoke_schedule(asset["date"])
    result = run_execution_path(
        asset_market=asset,
        target_schedule=schedule,
        initial_wealth=p2.initial_wealth,
        initial_exposure=p2.initial_exposure,
        cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
    )

    out_dir = root / cfg.paths.processed_dir / p2.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "execution_smoke_ledger.parquet"
    fills_path = out_dir / "execution_smoke_fills.parquet"
    audit_path = out_dir / "phase2_audit.json"
    result.ledger.to_parquet(ledger_path, index=False)
    result.fills.to_parquet(fills_path, index=False)

    csv_files: list[Path] = []
    if p2.write_csv_copy:
        ledger_csv = out_dir / "execution_smoke_ledger.csv"
        fills_csv = out_dir / "execution_smoke_fills.csv"
        result.ledger.to_csv(ledger_csv, index=False)
        result.fills.to_csv(fills_csv, index=False)
        csv_files.extend([ledger_csv, fills_csv])

    conservation = pd.to_numeric(result.ledger["accounting_error"], errors="coerce").abs().max()
    total_cost = float(pd.to_numeric(result.ledger["transaction_cost"], errors="coerce").sum())
    final_wealth = float(result.ledger.iloc[-1]["wealth_close"])
    audit: dict[str, Any] = {
        "phase": 2,
        "engine_scope": "single risky asset + cash; exogenous target schedule only",
        "smoke_asset": smoke_asset,
        "smoke_rows": int(len(asset)),
        "initial_wealth": float(p2.initial_wealth),
        "initial_exposure": float(p2.initial_exposure),
        "cost_bps_per_side": float(cfg.research_contract.cost_bps_per_side),
        "signal_time": cfg.research_contract.signal_time,
        "execution_time": cfg.research_contract.execution_time,
        "schedule": {str(pd.Timestamp(k).date()): float(v) for k, v in schedule.items()},
        "ledger_rows": int(len(result.ledger)),
        "fill_rows": int(len(result.fills)),
        "total_transaction_cost": total_cost,
        "final_wealth": final_wealth,
        "max_abs_accounting_error": float(conservation),
        "canonical_market_sha256": sha256_file(canonical_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase2Outputs(
        smoke_ledger=ledger_path,
        smoke_fills=fills_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_files=(canonical_path,),
    )
