import pandas as pd
from pandas.testing import assert_series_equal

from trading_policy_oracle_repair.execution import run_execution_path
from trading_policy_oracle_repair.policies import BuyHoldPolicy
from trading_policy_oracle_repair.replay import replay_policy


def test_buyhold_replay_matches_phase2_schedule_engine():
    dates = pd.bdate_range("2026-01-02", periods=8)
    close = [100, 101, 103, 102, 104, 105, 107, 108]
    market = pd.DataFrame({
        "date": dates,
        "asset_id": ["AAA"] * 8,
        "open": close,
        "high": [x + 1 for x in close],
        "low": [x - 1 for x in close],
        "close": close,
        "signal_price": close,
        "dividend_cash": [0.0] * 8,
        "split_factor": [1.0] * 8,
        "cash_daily_return": [0.0] * 8,
    })
    replay = replay_policy(
        asset_market=market,
        policy=BuyHoldPolicy(),
        initial_wealth=1000,
        initial_exposure=0,
        cost_bps_per_side=5,
    )
    schedule = {d: 1.0 for d in dates[:-1]}
    phase2 = run_execution_path(
        asset_market=market,
        target_schedule=schedule,
        initial_wealth=1000,
        initial_exposure=0,
        cost_bps_per_side=5,
    )
    assert_series_equal(
        replay.ledger["wealth_close"], phase2.ledger["wealth_close"],
        check_names=False, rtol=1e-12, atol=1e-12,
    )
