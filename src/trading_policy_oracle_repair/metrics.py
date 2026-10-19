from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(wealth: pd.Series) -> float:
    w = pd.to_numeric(wealth, errors="raise").astype(float)
    if w.empty or (w <= 0).any():
        raise ValueError("wealth must be nonempty and positive")
    peak = w.cummax()
    dd = w / peak - 1.0
    return float(dd.min())


def summarize_replay(
    ledger: pd.DataFrame,
    fills: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    annualization_days: int = 252,
) -> dict[str, float | int | str]:
    if ledger.empty:
        raise ValueError("ledger may not be empty")
    w = pd.to_numeric(ledger["wealth_close"], errors="raise").astype(float)
    if (w <= 0).any():
        raise ValueError("wealth must stay positive")
    daily = w.pct_change().dropna()
    periods = max(len(w) - 1, 1)
    years = periods / float(annualization_days)
    initial = float(w.iloc[0])
    final = float(w.iloc[-1])
    total_return = final / initial - 1.0
    cagr = (final / initial) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = float(daily.std(ddof=1) * np.sqrt(annualization_days)) if len(daily) > 1 else 0.0
    sharpe_zero_rf = (
        float(daily.mean() / daily.std(ddof=1) * np.sqrt(annualization_days))
        if len(daily) > 1 and float(daily.std(ddof=1)) > 0
        else 0.0
    )
    turnover = pd.to_numeric(ledger["trade_notional"], errors="coerce").fillna(0.0)
    wealth_pre = pd.to_numeric(ledger["wealth_pre_trade"], errors="coerce").replace(0.0, np.nan)
    turnover_fraction = (turnover / wealth_pre).fillna(0.0)
    total_cost = float(pd.to_numeric(ledger["transaction_cost"], errors="coerce").fillna(0.0).sum())
    return {
        "asset_id": str(ledger.iloc[0]["asset_id"]),
        "policy_id": str(ledger.iloc[0].get("policy_id", "unknown")),
        "rows": int(len(ledger)),
        "initial_wealth": initial,
        "final_wealth": final,
        "total_return": float(total_return),
        "CAGR": float(cagr),
        "ann_vol": ann_vol,
        "Sharpe_zero_rf": sharpe_zero_rf,
        "MaxDD": max_drawdown(w),
        "mean_exposure": float(pd.to_numeric(ledger["exposure_close"], errors="raise").mean()),
        "total_turnover": float(turnover_fraction.sum()),
        "total_cost": total_cost,
        "trade_count": int(len(fills)),
        "ready_fraction": float(actions["ready"].mean()) if not actions.empty else 0.0,
        "target_changes": int((actions["target_exposure"].diff().abs().fillna(0.0) > 1e-12).sum()) if not actions.empty else 0,
    }
