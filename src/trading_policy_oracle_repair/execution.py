from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .costs import ProportionalCostModel, solve_post_cost_wealth
from .dates import normalize_daily_dates
from .state import AccountState, make_initial_state


@dataclass(frozen=True)
class RebalanceResult:
    shares_after: float
    cash_after: float
    wealth_post_cost: float
    trade_shares: float
    trade_notional: float
    transaction_cost: float
    target_exposure: float


@dataclass(frozen=True)
class DailyExecutionResult:
    state: AccountState
    ledger_row: dict
    fill_row: dict | None


@dataclass(frozen=True)
class ExecutionPathResult:
    ledger: pd.DataFrame
    fills: pd.DataFrame
    final_state: AccountState


def rebalance_to_target(
    *,
    shares_before: float,
    cash_before: float,
    open_price: float,
    target_exposure: float,
    cost_model: ProportionalCostModel,
) -> RebalanceResult:
    if open_price <= 0:
        raise ValueError("open_price must be positive")
    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must lie in [0, 1]")

    current_risky_value = float(shares_before) * float(open_price)
    v_pre = current_risky_value + float(cash_before)
    if v_pre < -1e-10:
        raise ValueError("Negative pre-trade wealth")

    v_net = solve_post_cost_wealth(
        v_pre=v_pre,
        current_risky_value=current_risky_value,
        target_exposure=target_exposure,
        cost_rate=cost_model.cost_rate,
    )
    target_risky_value = float(target_exposure) * v_net
    shares_after = target_risky_value / float(open_price)
    cash_after = v_net - target_risky_value
    trade_shares = shares_after - float(shares_before)
    trade_notional = abs(target_risky_value - current_risky_value)
    transaction_cost = cost_model.cost(trade_notional)

    conservation_error = v_pre - (v_net + transaction_cost)
    if abs(conservation_error) > 1e-9 * max(v_pre, 1.0):
        raise AssertionError("Post-cost wealth conservation failed")
    if cash_after < -1e-9:
        raise AssertionError("Long/cash target generated negative cash")

    return RebalanceResult(
        shares_after=shares_after,
        cash_after=max(cash_after, 0.0),
        wealth_post_cost=v_net,
        trade_shares=trade_shares,
        trade_notional=trade_notional,
        transaction_cost=transaction_cost,
        target_exposure=float(target_exposure),
    )


def prepare_execution_market(asset_market: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one-asset canonical market data once."""
    required = {
        "date", "asset_id", "open", "close", "dividend_cash",
        "split_factor", "cash_daily_return",
    }
    missing = required - set(asset_market.columns)
    if missing:
        raise ValueError(f"Market table missing required columns: {sorted(missing)}")

    market = asset_market.copy()
    market["date"] = normalize_daily_dates(market["date"])
    market = market.sort_values("date").reset_index(drop=True)
    if market.empty:
        raise ValueError("asset_market may not be empty")
    if market["asset_id"].nunique() != 1:
        raise ValueError("Execution accepts exactly one asset")
    if market["date"].duplicated().any():
        raise ValueError("Duplicate dates in asset market")
    for c in ["open", "close"]:
        market[c] = pd.to_numeric(market[c], errors="raise").astype(float)
        if (market[c] <= 0).any():
            raise ValueError(f"Non-positive {c} price")
    market["dividend_cash"] = pd.to_numeric(market["dividend_cash"], errors="coerce").fillna(0.0).astype(float)
    market["split_factor"] = pd.to_numeric(market["split_factor"], errors="coerce").fillna(1.0).astype(float)
    if (market["split_factor"] <= 0).any():
        raise ValueError("split_factor must be positive")
    market["cash_daily_return"] = pd.to_numeric(market["cash_daily_return"], errors="coerce").fillna(0.0).astype(float)
    if (market["cash_daily_return"] <= -1.0).any():
        raise ValueError("cash_daily_return must be greater than -100%")
    return market


def initial_ledger_row(state: AccountState, first_row: pd.Series) -> dict:
    close_price = float(first_row["close"])
    return {
        "date": pd.Timestamp(first_row["date"]).normalize(),
        "asset_id": str(first_row["asset_id"]),
        "decision_date": pd.NaT,
        "execution_date": pd.NaT,
        "open": float(first_row["open"]),
        "close": close_price,
        "had_action": False,
        "target_exposure": np.nan,
        "cash_return": 0.0,
        "cash_prior_close": float(state.cash),
        "cash_after_accrual": float(state.cash),
        "split_factor": 1.0,
        "shares_prior_close": float(state.shares),
        "shares_after_split": float(state.shares),
        "dividend_per_share": 0.0,
        "dividend_cash_received": 0.0,
        "wealth_pre_trade": float(state.wealth(close_price)),
        "exposure_pre_trade": float(state.exposure(close_price)),
        "trade_shares": 0.0,
        "trade_notional": 0.0,
        "transaction_cost": 0.0,
        "shares_close": float(state.shares),
        "cash_close": float(state.cash),
        "wealth_close": float(state.wealth(close_price)),
        "exposure_close": float(state.exposure(close_price)),
        "accounting_error": 0.0,
        "event_type": "initial_close",
    }


def execute_next_day(
    *,
    state: AccountState,
    previous_row: pd.Series,
    row: pd.Series,
    target_exposure: float | None,
    cost_model: ProportionalCostModel,
) -> DailyExecutionResult:
    """Advance a close state through the next open and next close.

    This is the single accounting primitive shared by Phase 2 schedules and
    Phase 4 policy replay. `target_exposure` must have been decided using only
    information available at `previous_row` close. None means no rebalance.
    """
    decision_date = pd.Timestamp(previous_row["date"]).normalize()
    execution_date = pd.Timestamp(row["date"]).normalize()
    if pd.Timestamp(state.date).normalize() != decision_date:
        raise ValueError("state date must equal previous close/decision date")
    if str(state.asset_id) != str(row["asset_id"]) or str(state.asset_id) != str(previous_row["asset_id"]):
        raise ValueError("asset_id mismatch in execution step")
    if execution_date <= decision_date:
        raise ValueError("execution date must be after decision date")

    shares_prior_close = float(state.shares)
    cash_prior_close = float(state.cash)

    cash_return = float(row["cash_daily_return"])
    cash_after_accrual = cash_prior_close * (1.0 + cash_return)

    split_factor = float(row["split_factor"])
    shares_after_split = shares_prior_close * split_factor

    dividend_per_share = float(row["dividend_cash"])
    dividend_received = shares_after_split * dividend_per_share
    cash_before_trade = cash_after_accrual + dividend_received

    open_price = float(row["open"])
    close_price = float(row["close"])
    risky_pre = shares_after_split * open_price
    wealth_pre = risky_pre + cash_before_trade
    exposure_pre = 0.0 if wealth_pre <= 0 else risky_pre / wealth_pre

    had_action = target_exposure is not None
    if had_action:
        reb = rebalance_to_target(
            shares_before=shares_after_split,
            cash_before=cash_before_trade,
            open_price=open_price,
            target_exposure=float(target_exposure),
            cost_model=cost_model,
        )
        shares_close = reb.shares_after
        cash_close = reb.cash_after
        trade_shares = reb.trade_shares
        trade_notional = reb.trade_notional
        transaction_cost = reb.transaction_cost
        wealth_post_trade = reb.wealth_post_cost
        fill_row = None
        if trade_notional > 1e-12:
            fill_row = {
                "asset_id": str(state.asset_id),
                "decision_date": decision_date,
                "execution_date": execution_date,
                "execution_price": open_price,
                "target_exposure": float(target_exposure),
                "shares_before": shares_after_split,
                "shares_after": shares_close,
                "trade_shares": trade_shares,
                "trade_notional": trade_notional,
                "transaction_cost": transaction_cost,
                "wealth_pre_trade": wealth_pre,
                "wealth_post_cost": wealth_post_trade,
            }
    else:
        shares_close = shares_after_split
        cash_close = cash_before_trade
        trade_shares = 0.0
        trade_notional = 0.0
        transaction_cost = 0.0
        wealth_post_trade = wealth_pre
        fill_row = None

    wealth_close = shares_close * close_price + cash_close
    exposure_close = 0.0 if wealth_close <= 0 else (shares_close * close_price) / wealth_close
    accounting_error = wealth_close - (shares_close * close_price + cash_close)
    if abs(accounting_error) > 1e-9 * max(wealth_close, 1.0):
        raise AssertionError("Close accounting identity failed")
    if had_action and abs(wealth_pre - (wealth_post_trade + transaction_cost)) > 1e-9 * max(wealth_pre, 1.0):
        raise AssertionError("Trade-boundary accounting identity failed")

    new_state = AccountState(
        date=execution_date,
        asset_id=str(state.asset_id),
        shares=shares_close,
        cash=cash_close,
    )
    ledger_row = {
        "date": execution_date,
        "asset_id": str(state.asset_id),
        "decision_date": decision_date,
        "execution_date": execution_date,
        "open": open_price,
        "close": close_price,
        "had_action": bool(had_action),
        "target_exposure": float(target_exposure) if had_action else np.nan,
        "cash_return": cash_return,
        "cash_prior_close": cash_prior_close,
        "cash_after_accrual": cash_after_accrual,
        "split_factor": split_factor,
        "shares_prior_close": shares_prior_close,
        "shares_after_split": shares_after_split,
        "dividend_per_share": dividend_per_share,
        "dividend_cash_received": dividend_received,
        "wealth_pre_trade": wealth_pre,
        "exposure_pre_trade": exposure_pre,
        "trade_shares": trade_shares,
        "trade_notional": trade_notional,
        "transaction_cost": transaction_cost,
        "shares_close": shares_close,
        "cash_close": cash_close,
        "wealth_close": wealth_close,
        "exposure_close": exposure_close,
        "accounting_error": accounting_error,
        "event_type": "daily_close",
    }
    return DailyExecutionResult(state=new_state, ledger_row=ledger_row, fill_row=fill_row)


def _normalize_schedule(target_schedule: Mapping | pd.Series | None) -> dict[pd.Timestamp, float]:
    if target_schedule is None:
        return {}
    items = target_schedule.items()
    out: dict[pd.Timestamp, float] = {}
    for date, target in items:
        if target is None or (isinstance(target, float) and np.isnan(target)):
            continue
        ts = pd.Timestamp(date).normalize()
        val = float(target)
        if not 0.0 <= val <= 1.0:
            raise ValueError("All target exposures must lie in [0, 1]")
        out[ts] = val
    return out


def run_execution_path(
    *,
    asset_market: pd.DataFrame,
    target_schedule: Mapping | pd.Series | None,
    initial_wealth: float,
    initial_exposure: float,
    cost_bps_per_side: float,
) -> ExecutionPathResult:
    """Run an externally supplied close-t -> next-open target schedule."""
    market = prepare_execution_market(asset_market)
    schedule = _normalize_schedule(target_schedule)
    asset_id = str(market.loc[0, "asset_id"])
    first = market.iloc[0]
    state = make_initial_state(
        date=first["date"],
        asset_id=asset_id,
        close_price=float(first["close"]),
        initial_wealth=float(initial_wealth),
        initial_exposure=float(initial_exposure),
    )
    cost_model = ProportionalCostModel.from_bps(cost_bps_per_side)

    ledger_rows: list[dict] = [initial_ledger_row(state, first)]
    fill_rows: list[dict] = []

    for i in range(1, len(market)):
        prev = market.iloc[i - 1]
        row = market.iloc[i]
        decision_date = pd.Timestamp(prev["date"]).normalize()
        target = schedule.get(decision_date)
        step = execute_next_day(
            state=state,
            previous_row=prev,
            row=row,
            target_exposure=target,
            cost_model=cost_model,
        )
        state = step.state
        ledger_rows.append(step.ledger_row)
        if step.fill_row is not None:
            fill_rows.append(step.fill_row)

    ledger = pd.DataFrame(ledger_rows)
    fills = pd.DataFrame(fill_rows)
    if fills.empty:
        fills = pd.DataFrame(columns=[
            "asset_id", "decision_date", "execution_date", "execution_price",
            "target_exposure", "shares_before", "shares_after", "trade_shares",
            "trade_notional", "transaction_cost", "wealth_pre_trade",
            "wealth_post_cost",
        ])
    return ExecutionPathResult(ledger=ledger, fills=fills, final_state=state)
