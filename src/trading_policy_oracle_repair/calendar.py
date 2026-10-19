from __future__ import annotations

import pandas as pd

from .dates import normalize_daily_dates


def build_event_clock(asset_df: pd.DataFrame) -> pd.DataFrame:
    """Audit table for the fixed close-t decision -> open-(t+1) execution clock."""
    x = asset_df[["date", "asset_id", "close", "open"]].copy()
    x["date"] = normalize_daily_dates(x["date"])
    x = x.sort_values("date").reset_index(drop=True)
    x["execution_date"] = x["date"].shift(-1)
    x["execution_open"] = x["open"].shift(-1)
    x = x.rename(columns={"date": "decision_date", "close": "decision_close"})
    x["signal_time"] = "close"
    x["execution_time"] = "next_open"
    x["has_next_open"] = x["execution_date"].notna()
    return x[
        [
            "asset_id", "decision_date", "decision_close", "execution_date",
            "execution_open", "signal_time", "execution_time", "has_next_open",
        ]
    ]
