from __future__ import annotations

import json

import pandas as pd

from .dates import normalize_daily_dates
from .policies.base import Action, BasePolicy
from .state import make_initial_state


def history_through(market: pd.DataFrame, decision_date) -> pd.DataFrame:
    if market.empty:
        raise ValueError("market may not be empty")
    m = market.copy()
    m["date"] = normalize_daily_dates(m["date"])
    d = pd.Timestamp(decision_date).normalize()
    out = m.loc[m["date"] <= d].sort_values("date").reset_index(drop=True)
    if out.empty or pd.Timestamp(out.iloc[-1]["date"]).normalize() != d:
        raise ValueError(f"decision_date {d.date()} not found in market")
    return out


def evaluate_policy_at(
    *,
    policy: BasePolicy,
    full_market: pd.DataFrame,
    decision_date,
    current_exposure: float,
) -> Action:
    """Past-only adapter used by Phase 3 and future replay code.

    The caller may hold the full realized path, but the policy receives only
    rows with date <= decision_date. This is the Phase 3 non-anticipation wall.
    """
    h = history_through(full_market, decision_date)
    last = h.iloc[-1]
    state = make_initial_state(
        date=last["date"],
        asset_id=str(last["asset_id"]),
        close_price=float(last["close"]),
        initial_wealth=1.0,
        initial_exposure=float(current_exposure),
    )
    return policy.act(h, state)


def generate_signal_audit(
    *,
    policy: BasePolicy,
    market: pd.DataFrame,
    initial_exposure: float = 0.0,
) -> pd.DataFrame:
    """Generate a decision-only audit table; this is NOT a backtest/replay.

    To exercise restartability, each next decision receives the previous target
    as its current exposure. No shares/cash ledger or performance is computed.
    Formal replay remains Phase 4.
    """
    m = market.copy().sort_values("date").reset_index(drop=True)
    current = float(initial_exposure)
    rows: list[dict] = []
    for d in m["date"]:
        action = evaluate_policy_at(
            policy=policy,
            full_market=m,
            decision_date=d,
            current_exposure=current,
        )
        rows.append({
            "date": pd.Timestamp(action.decision_date),
            "asset_id": str(m.loc[m["date"] == d, "asset_id"].iloc[0]),
            "policy_id": action.policy_id,
            "current_exposure_input": current,
            "target_exposure": float(action.target_exposure),
            "ready": bool(action.ready),
            "reason": action.reason,
            "diagnostics_json": json.dumps(dict(action.diagnostics), sort_keys=True, default=float),
        })
        current = float(action.target_exposure)
    return pd.DataFrame(rows)
