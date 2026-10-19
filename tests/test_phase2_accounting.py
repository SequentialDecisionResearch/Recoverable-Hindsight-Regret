import pandas as pd
import pytest

from trading_policy_oracle_repair.execution import run_execution_path


def test_constant_price_with_cost_cannot_create_money():
    market = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        "asset_id": ["AAA"] * 3,
        "open": [100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 100.0],
        "dividend_cash": [0.0, 0.0, 0.0],
        "split_factor": [1.0, 1.0, 1.0],
        "cash_daily_return": [0.0, 0.0, 0.0],
    })
    result = run_execution_path(
        asset_market=market,
        target_schedule={pd.Timestamp("2026-01-02"): 1.0, pd.Timestamp("2026-01-05"): 1.0},
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=10.0,
    )
    assert result.ledger.iloc[-1]["wealth_close"] < 1000.0
    assert result.ledger.iloc[-1]["wealth_close"] == pytest.approx(result.ledger.iloc[1]["wealth_close"])
    assert result.ledger["accounting_error"].abs().max() < 1e-12
    assert result.ledger["transaction_cost"].sum() > 0.0


def test_cash_accrual_is_applied_to_carried_cash_before_open():
    market = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "asset_id": ["AAA", "AAA"],
        "open": [100.0, 100.0],
        "close": [100.0, 100.0],
        "dividend_cash": [0.0, 0.0],
        "split_factor": [1.0, 1.0],
        "cash_daily_return": [0.0, 0.01],
    })
    result = run_execution_path(
        asset_market=market,
        target_schedule=None,
        initial_wealth=1000.0,
        initial_exposure=0.0,
        cost_bps_per_side=0.0,
    )
    assert result.ledger.iloc[-1]["cash_after_accrual"] == pytest.approx(1010.0)
    assert result.ledger.iloc[-1]["wealth_close"] == pytest.approx(1010.0)
