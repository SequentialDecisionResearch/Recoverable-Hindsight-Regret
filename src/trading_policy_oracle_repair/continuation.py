from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import ProportionalCostModel
from .execution import execute_next_day, prepare_execution_market
from .policies.base import BasePolicy
from .state import make_initial_state


HOLD_REASONS = {"inside_channel_hold", "rsi_band_hold"}


@dataclass(frozen=True)
class PolicyActionTemplate:
    dates: np.ndarray
    hold_current: np.ndarray
    fixed_target: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.dates)
        if self.hold_current.shape != (n,) or self.fixed_target.shape != (n,):
            raise ValueError("PolicyActionTemplate arrays must have equal one-dimensional length")
        if (~np.isfinite(self.fixed_target)).any():
            raise ValueError("fixed_target contains non-finite values")
        if ((self.fixed_target < 0.0) | (self.fixed_target > 1.0)).any():
            raise ValueError("fixed_target must lie in [0, 1]")


def build_action_template(
    *,
    market_dates: pd.Series | pd.Index | np.ndarray,
    replay_actions: pd.DataFrame,
) -> PolicyActionTemplate:
    """Convert Phase-4 actions into an exposure-independent continuation template.

    The six canonical policies have only two action forms at a given date:
      * fixed target exposure, determined by market history; or
      * hold the current close exposure (warmup / hysteresis band).

    Phase 4 already evaluated the exact canonical policy definitions on every
    date.  We reuse their decision reasons instead of recomputing indicators.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(market_dates)).normalize()
    a = replay_actions.copy()
    required = {"date", "target_exposure", "reason"}
    missing = required - set(a.columns)
    if missing:
        raise ValueError(f"replay_actions missing columns: {sorted(missing)}")
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    if a["date"].duplicated().any():
        raise ValueError("replay_actions contains duplicate dates")
    a = a.set_index("date").reindex(dates)
    if a["reason"].isna().any():
        bad = dates[a["reason"].isna().to_numpy()][:3]
        raise ValueError(f"Missing Phase-4 action rows for dates: {list(bad)}")

    reason = a["reason"].astype(str)
    hold = reason.str.startswith("warmup:") | reason.isin(HOLD_REASONS)
    target = pd.to_numeric(a["target_exposure"], errors="raise").astype(float).to_numpy()
    return PolicyActionTemplate(
        dates=dates.to_numpy(dtype="datetime64[ns]"),
        hold_current=hold.to_numpy(dtype=bool),
        fixed_target=target,
    )


def _broadcast_float(*arrays):
    return [np.asarray(x, dtype=float) for x in np.broadcast_arrays(*arrays)]


def vectorized_account_step(
    *,
    wealth,
    exposure,
    previous_close,
    next_open,
    next_close,
    cash_daily_return,
    split_factor,
    dividend_cash,
    target_exposure,
    cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact homogeneous one-asset Phase-2 accounting step, vectorized.

    This is an algebraic fast path for offline label construction.  It uses the
    same target-weight/cost equation as Phase 2.  Regression tests compare it
    directly against execute_next_day().

    Returns (wealth_close, exposure_close, transaction_cost).
    """
    (
        wealth,
        exposure,
        previous_close,
        next_open,
        next_close,
        cash_daily_return,
        split_factor,
        dividend_cash,
        target_exposure,
    ) = _broadcast_float(
        wealth,
        exposure,
        previous_close,
        next_open,
        next_close,
        cash_daily_return,
        split_factor,
        dividend_cash,
        target_exposure,
    )
    if np.any(~np.isfinite(wealth)) or np.any(wealth <= 0.0):
        raise ValueError("wealth must be positive and finite")
    if np.any((exposure < -1e-12) | (exposure > 1.0 + 1e-12)):
        raise ValueError("exposure must lie in [0, 1]")
    if np.any((target_exposure < -1e-12) | (target_exposure > 1.0 + 1e-12)):
        raise ValueError("target_exposure must lie in [0, 1]")
    if np.any(previous_close <= 0.0) or np.any(next_open <= 0.0) or np.any(next_close <= 0.0):
        raise ValueError("prices must be positive")
    if np.any(split_factor <= 0.0):
        raise ValueError("split_factor must be positive")
    if np.any(cash_daily_return <= -1.0):
        raise ValueError("cash_daily_return must exceed -100%")

    exposure = np.clip(exposure, 0.0, 1.0)
    target_exposure = np.clip(target_exposure, 0.0, 1.0)
    shares_prior = exposure * wealth / previous_close
    cash_prior = (1.0 - exposure) * wealth
    cash_after_accrual = cash_prior * (1.0 + cash_daily_return)
    shares_after_split = shares_prior * split_factor
    cash_before_trade = cash_after_accrual + shares_after_split * dividend_cash
    risky_pre = shares_after_split * next_open
    wealth_pre = risky_pre + cash_before_trade

    c = float(cost_bps_per_side) / 10000.0
    if c < 0.0:
        raise ValueError("cost_bps_per_side must be nonnegative")
    if c == 0.0:
        wealth_net = wealth_pre
    else:
        buy = target_exposure * wealth_pre >= risky_pre
        denom_buy = 1.0 + c * target_exposure
        denom_sell = 1.0 - c * target_exposure
        if np.any(denom_sell <= 0.0):
            raise ValueError("transaction cost rate is incompatible with target exposure")
        wealth_buy = (wealth_pre + c * risky_pre) / denom_buy
        wealth_sell = (wealth_pre - c * risky_pre) / denom_sell
        wealth_net = np.where(buy, wealth_buy, wealth_sell)
    transaction_cost = wealth_pre - wealth_net

    risky_after = target_exposure * wealth_net
    shares_after = risky_after / next_open
    cash_after = (1.0 - target_exposure) * wealth_net
    wealth_close = shares_after * next_close + cash_after
    exposure_close = np.divide(
        shares_after * next_close,
        wealth_close,
        out=np.zeros_like(wealth_close),
        where=wealth_close > 0.0,
    )
    if np.any(~np.isfinite(wealth_close)) or np.any(wealth_close <= 0.0):
        raise ValueError("continuation step produced non-positive wealth")
    return wealth_close, np.clip(exposure_close, 0.0, 1.0), transaction_cost


def continuation_utility_grid(
    *,
    asset_market: pd.DataFrame,
    action_template: PolicyActionTemplate,
    start_exposures: list[float] | np.ndarray,
    horizon: int,
    cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all H-step policy continuation utilities in one vectorized pass.

    Output shapes are (n_start_dates, n_exposures) for utility/final wealth/final
    exposure.  `decision_dates` and `label_available_dates` each have length
    n_start_dates.
    """
    market = prepare_execution_market(asset_market)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(market) <= horizon:
        raise ValueError("market history must contain more than horizon rows")
    if len(action_template.dates) != len(market):
        raise ValueError("action template length does not match market")
    md = pd.DatetimeIndex(market["date"]).normalize().to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(md, action_template.dates):
        raise ValueError("action template dates do not match market dates")

    exposures0 = np.asarray(start_exposures, dtype=float)
    if exposures0.ndim != 1 or len(exposures0) == 0:
        raise ValueError("start_exposures must be a nonempty vector")
    if np.any((exposures0 < 0.0) | (exposures0 > 1.0)):
        raise ValueError("start_exposures must lie in [0, 1]")

    n = len(market) - horizon
    ecount = len(exposures0)
    wealth = np.ones((n, ecount), dtype=float)
    exposure = np.broadcast_to(exposures0[None, :], (n, ecount)).copy()
    starts = np.arange(n, dtype=int)

    close = market["close"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    cash_r = market["cash_daily_return"].to_numpy(dtype=float)
    split = market["split_factor"].to_numpy(dtype=float)
    div = market["dividend_cash"].to_numpy(dtype=float)

    for h in range(horizon):
        idx = starts + h
        nxt = idx + 1
        hold = action_template.hold_current[idx][:, None]
        fixed = action_template.fixed_target[idx][:, None]
        target = np.where(hold, exposure, fixed)
        wealth, exposure, _ = vectorized_account_step(
            wealth=wealth,
            exposure=exposure,
            previous_close=close[idx][:, None],
            next_open=open_[nxt][:, None],
            next_close=close[nxt][:, None],
            cash_daily_return=cash_r[nxt][:, None],
            split_factor=split[nxt][:, None],
            dividend_cash=div[nxt][:, None],
            target_exposure=target,
            cost_bps_per_side=cost_bps_per_side,
        )

    utility = np.log(wealth)
    decision_dates = md[:n]
    label_dates = md[horizon : horizon + n]
    return utility, wealth, exposure, np.stack([decision_dates, label_dates], axis=1)


def direct_policy_continuation(
    *,
    asset_market: pd.DataFrame,
    policy: BasePolicy,
    decision_index: int,
    horizon: int,
    initial_exposure: float,
    initial_wealth: float = 1.0,
    cost_bps_per_side: float = 5.0,
) -> tuple[float, float]:
    """Slow reference continuation using the actual policy + Phase-2 engine."""
    market = prepare_execution_market(asset_market)
    for col in ["high", "low", "signal_price"]:
        if col not in market.columns:
            raise ValueError(f"asset_market missing {col}")
    if decision_index < 0 or horizon <= 0 or decision_index + horizon >= len(market):
        raise ValueError("requested continuation extends beyond market")
    market.attrs["_policy_history_prevalidated"] = True
    row0 = market.iloc[decision_index]
    state = make_initial_state(
        date=row0["date"],
        asset_id=str(row0["asset_id"]),
        close_price=float(row0["close"]),
        initial_wealth=float(initial_wealth),
        initial_exposure=float(initial_exposure),
    )
    cost_model = ProportionalCostModel.from_bps(cost_bps_per_side)
    for i in range(decision_index, decision_index + horizon):
        history = market.iloc[: i + 1]
        history.attrs["_policy_history_prevalidated"] = True
        action = policy.act(history, state)
        step = execute_next_day(
            state=state,
            previous_row=market.iloc[i],
            row=market.iloc[i + 1],
            target_exposure=float(action.target_exposure),
            cost_model=cost_model,
        )
        state = step.state
    final_wealth = float(state.wealth(float(market.iloc[decision_index + horizon]["close"])))
    return float(np.log(final_wealth / initial_wealth)), final_wealth


def bounded_oracle_utility_grid(
    *,
    asset_market: pd.DataFrame,
    start_exposures: list[float] | np.ndarray,
    horizon: int,
    max_switches: int,
    cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized bounded-switch binary oracle for all starts/exposures.

    The first rebalance from a fractional starting exposure to {0,1} pays the
    exact Phase-2 transaction cost and counts as a switch unless the target is
    already exactly equal to the starting exposure.  This preserves Phase-5
    semantics for starting exposures 0 and 1 and extends them transparently to
    the full Phase-6 exposure grid.
    """
    market = prepare_execution_market(asset_market)
    if horizon <= 0 or len(market) <= horizon:
        raise ValueError("invalid horizon for oracle grid")
    if max_switches < 0:
        raise ValueError("max_switches must be nonnegative")
    e0 = np.asarray(start_exposures, dtype=float)
    if e0.ndim != 1 or len(e0) == 0 or np.any((e0 < 0.0) | (e0 > 1.0)):
        raise ValueError("start_exposures must be a nonempty vector in [0,1]")

    n = len(market) - horizon
    ecount = len(e0)
    kmax = min(int(max_switches), int(horizon))
    neg = -np.inf
    dp = np.full((n, ecount, kmax + 1, 2), neg, dtype=float)
    starts = np.arange(n, dtype=int)
    close = market["close"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    cash_r = market["cash_daily_return"].to_numpy(dtype=float)
    split = market["split_factor"].to_numpy(dtype=float)
    div = market["dividend_cash"].to_numpy(dtype=float)

    # First interval: previous exposure can be fractional.
    idx = starts
    nxt = idx + 1
    base_e = np.broadcast_to(e0[None, :], (n, ecount))
    for new_pos in (0, 1):
        target = np.full((n, ecount), float(new_pos))
        wealth1, _, _ = vectorized_account_step(
            wealth=np.ones((n, ecount)),
            exposure=base_e,
            previous_close=close[idx][:, None],
            next_open=open_[nxt][:, None],
            next_close=close[nxt][:, None],
            cash_daily_return=cash_r[nxt][:, None],
            split_factor=split[nxt][:, None],
            dividend_cash=div[nxt][:, None],
            target_exposure=target,
            cost_bps_per_side=cost_bps_per_side,
        )
        lg = np.log(wealth1)
        switches = (~np.isclose(base_e, float(new_pos), atol=1e-12, rtol=0.0)).astype(int)
        for k in range(kmax + 1):
            mask = switches == k
            if np.any(mask):
                dp[:, :, k, new_pos] = np.where(mask, lg, dp[:, :, k, new_pos])

    # Remaining intervals: previous and target positions are binary.
    for h in range(1, horizon):
        idx = starts + h
        nxt = idx + 1
        newdp = np.full_like(dp, neg)
        # Exact log gross for the four binary transitions; independent of e0.
        lg = np.empty((n, 2, 2), dtype=float)
        for prev_pos in (0, 1):
            for new_pos in (0, 1):
                w1, _, _ = vectorized_account_step(
                    wealth=np.ones(n),
                    exposure=np.full(n, float(prev_pos)),
                    previous_close=close[idx],
                    next_open=open_[nxt],
                    next_close=close[nxt],
                    cash_daily_return=cash_r[nxt],
                    split_factor=split[nxt],
                    dividend_cash=div[nxt],
                    target_exposure=np.full(n, float(new_pos)),
                    cost_bps_per_side=cost_bps_per_side,
                )
                lg[:, prev_pos, new_pos] = np.log(w1)

        for new_pos in (0, 1):
            stay = dp[:, :, :, new_pos] + lg[:, new_pos, new_pos][:, None, None]
            newdp[:, :, :, new_pos] = stay
            if kmax >= 1:
                other = 1 - new_pos
                switched = dp[:, :, :-1, other] + lg[:, other, new_pos][:, None, None]
                newdp[:, :, 1:, new_pos] = np.maximum(newdp[:, :, 1:, new_pos], switched)
        dp = newdp

    best = np.max(dp, axis=(2, 3))
    if np.any(~np.isfinite(best)):
        raise ValueError("No feasible bounded-oracle path for some start state")
    return best, np.exp(best)
