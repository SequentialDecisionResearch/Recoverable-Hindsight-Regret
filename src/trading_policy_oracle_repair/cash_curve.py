from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import normalize_daily_dates


def annual_yield_pct_to_daily_return(yield_pct: pd.Series, annualization_days: int = 252) -> pd.Series:
    y = pd.to_numeric(yield_pct, errors="coerce").astype(float) / 100.0
    if (y.dropna() <= -1.0).any():
        raise ValueError("Annualized cash yield must be greater than -100%")
    return np.expm1(np.log1p(y) / float(annualization_days))


def build_cash_curve(
    cash_proxy_raw: pd.DataFrame,
    target_dates,
    annualization_days: int = 252,
) -> pd.DataFrame:
    """Create a past-only daily cash-return curve on the research trading calendar.

    ^IRX close is interpreted as an annualized percentage yield observation.
    The return credited on target date t uses the yield available on the previous
    target trading date, never the same-day close.
    """
    proxy = cash_proxy_raw[["date", "close"]].copy()
    proxy["date"] = normalize_daily_dates(proxy["date"])
    proxy["cash_yield_ann_pct_observed"] = pd.to_numeric(proxy["close"], errors="raise").astype(float)
    proxy["cash_rate_source_date_observed"] = proxy["date"]
    proxy = proxy[["date", "cash_yield_ann_pct_observed", "cash_rate_source_date_observed"]]
    proxy = proxy.sort_values("date").drop_duplicates("date", keep="last")

    cal = pd.DataFrame({"date": normalize_daily_dates(target_dates)})
    cal = cal.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # For each research date, find the most recent cash observation known by that close.
    observed = pd.merge_asof(cal, proxy, on="date", direction="backward")

    # The cash return realized over the next daily interval must use information
    # available before that interval; therefore shift by one research trading day.
    observed["cash_yield_ann_pct"] = observed["cash_yield_ann_pct_observed"].shift(1)
    observed["cash_rate_source_date"] = observed["cash_rate_source_date_observed"].shift(1)
    observed["cash_daily_return"] = annual_yield_pct_to_daily_return(
        observed["cash_yield_ann_pct"], annualization_days=annualization_days
    )

    return observed[["date", "cash_yield_ann_pct", "cash_daily_return", "cash_rate_source_date"]]
