from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import ProportionalCostModel
from .execution import execute_next_day, initial_ledger_row, prepare_execution_market
from .policies.base import BasePolicy
from .state import make_initial_state


@dataclass(frozen=True)
class PolicyReplayResult:
    ledger: pd.DataFrame
    fills: pd.DataFrame
    actions: pd.DataFrame


def _prepare_policy_market(asset_market: pd.DataFrame) -> pd.DataFrame:
    market = prepare_execution_market(asset_market)
    required = {"high", "low", "signal_price"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"Policy replay market missing columns: {sorted(missing)}")
    for col in ["high", "low", "signal_price"]:
        market[col] = pd.to_numeric(market[col], errors="raise").astype(float)
        if (~np.isfinite(market[col])).any() or (market[col] <= 0).any():
            raise ValueError(f"Invalid {col} in policy replay market")
    # BasePolicy may use this marker to skip repeated full-history validation.
    market.attrs["_policy_history_prevalidated"] = True
    return market


def _history_view(market: pd.DataFrame, end_exclusive: int) -> pd.DataFrame:
    h = market.iloc[:end_exclusive]
    h.attrs["_policy_history_prevalidated"] = True
    return h


def _action_row(action, state, close_price: float) -> dict:
    return {
        "date": pd.Timestamp(action.decision_date).normalize(),
        "asset_id": str(state.asset_id),
        "policy_id": str(action.policy_id),
        "current_exposure": float(state.exposure(close_price)),
        "target_exposure": float(action.target_exposure),
        "ready": bool(action.ready),
        "reason": str(action.reason),
        "diagnostics_json": json.dumps(dict(action.diagnostics), sort_keys=True, default=float),
    }


def replay_policy(
    *,
    asset_market: pd.DataFrame,
    policy: BasePolicy,
    initial_wealth: float,
    initial_exposure: float,
    cost_bps_per_side: float,
) -> PolicyReplayResult:
    """Replay one deployable policy on one realized market path.

    At each close t, the policy receives only history through t and the actual
    close-t account state. Its target is executed by the Phase-2 accounting
    primitive at open t+1. Future rows are never passed to the policy.
    """
    market = _prepare_policy_market(asset_market)
    first = market.iloc[0]
    asset_id = str(first["asset_id"])
    state = make_initial_state(
        date=first["date"],
        asset_id=asset_id,
        close_price=float(first["close"]),
        initial_wealth=float(initial_wealth),
        initial_exposure=float(initial_exposure),
    )
    cost_model = ProportionalCostModel.from_bps(cost_bps_per_side)

    ledger_rows: list[dict] = []
    first_ledger = initial_ledger_row(state, first)
    first_ledger["policy_id"] = policy.policy_id
    first_ledger["decision_ready"] = False
    first_ledger["decision_reason"] = "initial_close"
    ledger_rows.append(first_ledger)
    fill_rows: list[dict] = []
    action_rows: list[dict] = []

    history = _history_view(market, 1)
    action = policy.act(history, state)
    action_rows.append(_action_row(action, state, float(first["close"])))

    for i in range(1, len(market)):
        prev = market.iloc[i - 1]
        row = market.iloc[i]
        step = execute_next_day(
            state=state,
            previous_row=prev,
            row=row,
            target_exposure=float(action.target_exposure),
            cost_model=cost_model,
        )
        state = step.state
        lr = dict(step.ledger_row)
        lr["policy_id"] = policy.policy_id
        lr["decision_ready"] = bool(action.ready)
        lr["decision_reason"] = str(action.reason)
        ledger_rows.append(lr)
        if step.fill_row is not None:
            fr = dict(step.fill_row)
            fr["policy_id"] = policy.policy_id
            fr["decision_ready"] = bool(action.ready)
            fr["decision_reason"] = str(action.reason)
            fill_rows.append(fr)

        history = _history_view(market, i + 1)
        action = policy.act(history, state)
        action_rows.append(_action_row(action, state, float(row["close"])))

    ledger = pd.DataFrame(ledger_rows)
    fills = pd.DataFrame(fill_rows)
    if fills.empty:
        fills = pd.DataFrame(columns=[
            "asset_id", "decision_date", "execution_date", "execution_price",
            "target_exposure", "shares_before", "shares_after", "trade_shares",
            "trade_notional", "transaction_cost", "wealth_pre_trade",
            "wealth_post_cost", "policy_id", "decision_ready", "decision_reason",
        ])
    actions = pd.DataFrame(action_rows)
    return PolicyReplayResult(ledger=ledger, fills=fills, actions=actions)
