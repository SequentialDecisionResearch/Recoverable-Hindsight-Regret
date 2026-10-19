from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dates import normalize_daily_dates


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace("/", "_")


def raw_symbol_path(project_root: str | Path, raw_dir: str | Path, provider: str, symbol: str) -> Path:
    return Path(project_root) / raw_dir / provider / f"{safe_symbol_name(symbol)}.parquet"


def load_raw_symbol(project_root: str | Path, raw_dir: str | Path, provider: str, symbol: str) -> pd.DataFrame:
    path = raw_symbol_path(project_root, raw_dir, provider, symbol)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 0 raw file for {symbol}: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Raw file is empty for {symbol}: {path}")

    out = df.copy()
    out["date"] = normalize_daily_dates(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    if out["date"].duplicated().any():
        dups = out.loc[out["date"].duplicated(), "date"].astype(str).tolist()[:5]
        raise ValueError(f"Duplicate daily dates in raw file for {symbol}: {dups}")
    return out
