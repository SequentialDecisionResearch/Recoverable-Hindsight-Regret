from __future__ import annotations

import numpy as np
import pandas as pd


def add_total_return_series(df: pd.DataFrame, base: float = 1.0) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="raise").astype(float)
    dividend = pd.to_numeric(out["dividend_cash"], errors="raise").astype(float)
    split = pd.to_numeric(out["split_factor"], errors="raise").astype(float)
    if (close <= 0).any():
        raise ValueError("Non-positive close price found")
    if (split <= 0).any():
        raise ValueError("Non-positive normalized split factor found")

    prev_close = close.shift(1)
    gross = ((close + dividend) * split) / prev_close
    gross.iloc[0] = 1.0
    if (~np.isfinite(gross)).any() or (gross <= 0).any():
        raise ValueError("Invalid total-return gross series")

    out["total_return_gross"] = gross.astype(float)
    out["total_return"] = out["total_return_gross"] - 1.0
    out["total_return_index"] = float(base) * out["total_return_gross"].cumprod()
    first_close = float(close.iloc[0])
    out["signal_price"] = first_close * out["total_return_index"] / float(base)
    return out
