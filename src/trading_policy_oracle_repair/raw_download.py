from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .dates import normalize_daily_dates
from .hashing import sha256_file


@dataclass(frozen=True)
class DownloadedFile:
    symbol: str
    parquet_path: Path
    csv_path: Path | None
    metadata_path: Path
    n_rows: int
    first_date: str
    last_date: str


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError(
            "yfinance is not installed. Run `python -m pip install -e .[dev]` first."
        ) from e
    return yf


def normalize_yfinance_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert one yfinance daily history frame to the immutable Phase 0 schema."""
    if df is None or df.empty:
        raise ValueError(f"No rows returned for {symbol}")

    out = df.copy()
    out.index.name = "date"
    out = out.reset_index()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})

    out["date"] = normalize_daily_dates(out["date"])

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close_vendor",
        "Volume": "volume",
        "Dividends": "dividend_cash_vendor",
        "Stock Splits": "split_factor_vendor",
        "Capital Gains": "capital_gains_vendor",
        "Repaired?": "repaired_vendor",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    out.insert(1, "asset_id", symbol)

    required = ["date", "asset_id", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for {symbol}: {missing}")

    if "dividend_cash_vendor" not in out.columns:
        out["dividend_cash_vendor"] = 0.0
    if "split_factor_vendor" not in out.columns:
        out["split_factor_vendor"] = 0.0

    out = out.sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        dups = out.loc[out["date"].duplicated(), "date"].astype(str).tolist()[:5]
        raise ValueError(f"Duplicate dates for {symbol}: {dups}")

    all_ohlc_missing = out[["open", "high", "low", "close"]].isna().all(axis=1)
    return out.loc[~all_ohlc_missing].reset_index(drop=True)


def download_symbol(symbol: str, cfg: AppConfig, project_root: str | Path) -> DownloadedFile:
    yf = _import_yfinance()
    raw_cfg = cfg.raw_data
    project_root = Path(project_root).resolve()
    out_dir = project_root / cfg.paths.raw_dir / raw_cfg.provider
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = yf.Ticker(symbol).history(
        start=raw_cfg.start,
        end=raw_cfg.end,
        interval=raw_cfg.interval,
        auto_adjust=raw_cfg.auto_adjust,
        actions=raw_cfg.actions,
        repair=raw_cfg.repair,
        keepna=raw_cfg.keepna,
        timeout=raw_cfg.timeout_seconds,
        raise_errors=True,
    )
    frame = normalize_yfinance_history(hist, symbol)

    safe = symbol.replace("^", "INDEX_").replace("/", "_")
    parquet_path = out_dir / f"{safe}.parquet"
    csv_path = out_dir / f"{safe}.csv" if cfg.phase0 and cfg.phase0.write_csv_copy else None
    metadata_path = out_dir / f"{safe}.metadata.json"

    frame.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        frame.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {
        "provider": "yfinance",
        "symbol": symbol,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "start": raw_cfg.start,
            "end": raw_cfg.end,
            "interval": raw_cfg.interval,
            "auto_adjust": raw_cfg.auto_adjust,
            "actions": raw_cfg.actions,
            "repair": raw_cfg.repair,
            "keepna": raw_cfg.keepna,
        },
        "data_quality": "exploratory",
        "notes": [
            "Daily dates are stored as exchange-calendar dates at midnight, timezone naive.",
            "Raw vendor OHLC are preserved for execution modeling.",
            "Vendor adjusted close is reference-only; Phase 1 builds its own signal total-return series.",
            "Dividend/split fields are normalized in Phase 1.",
        ],
        "rows": int(len(frame)),
        "first_date": frame["date"].min().isoformat(),
        "last_date": frame["date"].max().isoformat(),
        "columns": frame.columns.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return DownloadedFile(
        symbol=symbol,
        parquet_path=parquet_path,
        csv_path=csv_path,
        metadata_path=metadata_path,
        n_rows=len(frame),
        first_date=metadata["first_date"],
        last_date=metadata["last_date"],
    )


def download_all(cfg: AppConfig, project_root: str | Path) -> list[DownloadedFile]:
    symbols = list(cfg.raw_data.tickers)
    if cfg.raw_data.cash_proxy not in symbols:
        symbols.append(cfg.raw_data.cash_proxy)
    return [download_symbol(symbol, cfg, project_root) for symbol in symbols]


def audit_downloads(downloaded: list[DownloadedFile]) -> dict[str, Any]:
    return {
        "n_symbols": len(downloaded),
        "symbols": [
            {
                "symbol": item.symbol,
                "rows": item.n_rows,
                "first_date": item.first_date,
                "last_date": item.last_date,
                "parquet": str(item.parquet_path),
                "sha256": sha256_file(item.parquet_path),
            }
            for item in downloaded
        ],
    }
