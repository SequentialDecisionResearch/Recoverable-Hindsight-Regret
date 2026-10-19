from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .calendar import build_event_clock
from .cash_curve import build_cash_curve
from .config import AppConfig
from .corporate_actions import normalize_actions
from .data_loader import load_raw_symbol, raw_symbol_path
from .hashing import sha256_file
from .total_return import add_total_return_series


@dataclass(frozen=True)
class Phase1Outputs:
    canonical_market: Path
    cash_curve: Path
    event_clock: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_raw_files: tuple[Path, ...]


def _latest_phase0_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase0_*.json"))
    return files[-1] if files else None


def require_phase0_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase1 is not None
    cp = _latest_phase0_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase1.require_phase0_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 0 checkpoint found. Run Phase 0 first.")
    return cp


def run_phase1_toy_validation() -> dict[str, float | str]:
    toy = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        "asset_id": ["TOY"] * 3,
        "open": [100.0, 50.0, 51.0],
        "high": [101.0, 51.0, 52.0],
        "low": [99.0, 49.0, 50.0],
        "close": [100.0, 50.0, 51.0],
        "volume": [1000, 1000, 1000],
        "dividend_cash_vendor": [0.0, 0.0, 1.0],
        "split_factor_vendor": [0.0, 2.0, 0.0],
    })
    tr = add_total_return_series(normalize_actions(toy, price_basis="raw_native"))
    if abs(float(tr.loc[1, "total_return_gross"]) - 1.0) > 1e-12:
        raise AssertionError("Toy split should not create economic return")
    if abs(float(tr.loc[2, "total_return_gross"]) - 1.04) > 1e-12:
        raise AssertionError("Toy dividend return mismatch")

    proxy = pd.DataFrame({"date": toy["date"], "close": [5.0, 5.1, 5.2]})
    cash = build_cash_curve(proxy, toy["date"], annualization_days=252)
    expected_cash = (1.0 + 0.05) ** (1.0 / 252.0) - 1.0
    if abs(float(cash.loc[1, "cash_daily_return"]) - expected_cash) > 1e-12:
        raise AssertionError("Toy cash timing/conversion mismatch")

    clock = build_event_clock(tr)
    if clock.loc[0, "execution_date"] != pd.Timestamp("2026-01-05"):
        raise AssertionError("Toy event clock date mismatch")
    if float(clock.loc[0, "execution_open"]) != 50.0:
        raise AssertionError("Toy event clock next-open mismatch")

    return {
        "status": "PASSED",
        "split_day_gross": float(tr.loc[1, "total_return_gross"]),
        "dividend_day_gross": float(tr.loc[2, "total_return_gross"]),
        "cash_daily_return_from_5pct": float(cash.loc[1, "cash_daily_return"]),
        "first_execution_open": float(clock.loc[0, "execution_open"]),
    }


def build_phase1(cfg: AppConfig, project_root: str | Path) -> Phase1Outputs:
    if cfg.phase != 1 or cfg.phase1 is None:
        raise ValueError("build_phase1 requires phase=1 config")

    root = Path(project_root).resolve()
    require_phase0_checkpoint(root, cfg)
    p1 = cfg.phase1

    risky_frames: dict[str, pd.DataFrame] = {}
    raw_files: list[Path] = []
    for symbol in cfg.raw_data.tickers:
        path = raw_symbol_path(root, cfg.paths.raw_dir, p1.raw_provider_subdir, symbol)
        raw_files.append(path)
        df = load_raw_symbol(root, cfg.paths.raw_dir, p1.raw_provider_subdir, symbol)
        risky_frames[symbol] = add_total_return_series(normalize_actions(df, price_basis=p1.price_basis), base=p1.total_return_base)

    cash_symbol = cfg.raw_data.cash_proxy
    cash_path_raw = raw_symbol_path(root, cfg.paths.raw_dir, p1.raw_provider_subdir, cash_symbol)
    raw_files.append(cash_path_raw)
    cash_raw = load_raw_symbol(root, cfg.paths.raw_dir, p1.raw_provider_subdir, cash_symbol)

    all_dates = pd.DatetimeIndex(
        pd.concat([df["date"] for df in risky_frames.values()], ignore_index=True)
        .drop_duplicates()
        .sort_values()
    )
    cash = build_cash_curve(cash_raw, all_dates, annualization_days=p1.cash_annualization_days)

    canonical_parts: list[pd.DataFrame] = []
    clock_parts: list[pd.DataFrame] = []
    for symbol, df in risky_frames.items():
        merged = df.merge(cash, on="date", how="left", validate="many_to_one")
        merged["data_quality"] = "exploratory_yfinance"
        keep = [
            "date", "asset_id", "open", "high", "low", "close", "volume",
            "dividend_cash", "vendor_split_factor", "split_factor", "price_basis", "total_return_gross", "total_return",
            "total_return_index", "signal_price", "cash_yield_ann_pct",
            "cash_daily_return", "cash_rate_source_date", "data_quality",
        ]
        if "adj_close_vendor" in merged.columns:
            keep.insert(7, "adj_close_vendor")
        canonical_parts.append(merged[keep])
        clock_parts.append(build_event_clock(merged))

    canonical = (
        pd.concat(canonical_parts, ignore_index=True)
        .sort_values(["asset_id", "date"])
        .reset_index(drop=True)
    )
    event_clock = (
        pd.concat(clock_parts, ignore_index=True)
        .sort_values(["asset_id", "decision_date"])
        .reset_index(drop=True)
    )

    out_dir = root / cfg.paths.processed_dir / p1.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = out_dir / "canonical_market.parquet"
    cash_path = out_dir / "cash_curve.parquet"
    clock_path = out_dir / "event_clock.parquet"
    audit_path = out_dir / "phase1_audit.json"

    canonical.to_parquet(canonical_path, index=False)
    cash.to_parquet(cash_path, index=False)
    event_clock.to_parquet(clock_path, index=False)

    csv_files: list[Path] = []
    if p1.write_csv_copy:
        for frame, path in [
            (canonical, out_dir / "canonical_market.csv"),
            (cash, out_dir / "cash_curve.csv"),
            (event_clock, out_dir / "event_clock.csv"),
        ]:
            frame.to_csv(path, index=False)
            csv_files.append(path)

    audit: dict[str, Any] = {
        "phase": 1,
        "risky_assets": {},
        "cash_proxy": cash_symbol,
        "cash_method": p1.cash_method,
        "cash_annualization_days": p1.cash_annualization_days,
        "event_clock": {
            "signal_time": cfg.research_contract.signal_time,
            "execution_time": cfg.research_contract.execution_time,
        },
        "rows": {
            "canonical_market": int(len(canonical)),
            "cash_curve": int(len(cash)),
            "event_clock": int(len(event_clock)),
        },
        "missing_cash_return_rows": int(canonical["cash_daily_return"].isna().sum()),
        "raw_input_hashes": {str(p.relative_to(root)): sha256_file(p) for p in raw_files},
    }
    for symbol, df in risky_frames.items():
        audit["risky_assets"][symbol] = {
            "rows": int(len(df)),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "vendor_split_events": int((df["vendor_split_factor"] != 1.0).sum()),
            "effective_ledger_split_events": int((df["split_factor"] != 1.0).sum()),
            "price_basis": p1.price_basis,
            "dividend_events": int((df["dividend_cash"] != 0.0).sum()),
        }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase1Outputs(
        canonical_market=canonical_path,
        cash_curve=cash_path,
        event_clock=clock_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_raw_files=tuple(raw_files),
    )
