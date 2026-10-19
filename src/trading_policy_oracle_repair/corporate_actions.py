from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


PriceBasis = Literal["vendor_split_adjusted", "raw_native"]


def normalize_actions(df: pd.DataFrame, price_basis: PriceBasis = "vendor_split_adjusted") -> pd.DataFrame:
    """Normalize vendor corporate-action fields for research accounting.

    `vendor_split_adjusted` means historical OHLC are already expressed in a
    split-adjusted share basis (the exploratory yfinance/Yahoo convention used
    by this project). In that representation a historical account must *not*
    multiply shares again on split dates, so the effective ledger split factor
    is 1.0 while the vendor event is retained separately for audit.

    `raw_native` is reserved for truly unadjusted execution-price data. There the
    effective split factor equals the vendor split factor and the execution
    engine adjusts shares before the split-date open.
    """
    if price_basis not in {"vendor_split_adjusted", "raw_native"}:
        raise ValueError(f"Unsupported price_basis: {price_basis}")

    out = df.copy()
    div = pd.to_numeric(out.get("dividend_cash_vendor", 0.0), errors="coerce").fillna(0.0)
    split_raw = pd.to_numeric(out.get("split_factor_vendor", 0.0), errors="coerce").fillna(0.0)
    if (split_raw < 0).any():
        raise ValueError("Negative split factor found")

    vendor_split = np.where(split_raw > 0.0, split_raw, 1.0).astype(float)
    out["dividend_cash"] = div.astype(float)
    out["vendor_split_factor"] = vendor_split
    out["split_factor"] = vendor_split if price_basis == "raw_native" else np.ones(len(out), dtype=float)
    out["price_basis"] = price_basis
    return out
