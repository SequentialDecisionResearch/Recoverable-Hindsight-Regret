import pandas as pd

from trading_policy_oracle_repair.policies import MomentumPolicy
from trading_policy_oracle_repair.policy_runner import evaluate_policy_at


def test_future_rows_cannot_change_action_at_fixed_decision_date():
    dates = pd.bdate_range("2026-01-02", periods=8)
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    base = pd.DataFrame({
        "date": dates,
        "asset_id": ["T"] * 8,
        "open": prices,
        "high": [x + 1 for x in prices],
        "low": [x - 1 for x in prices],
        "close": prices,
        "signal_price": prices,
    })
    decision_date = dates[4]
    altered = base.copy()
    altered.loc[altered["date"] > decision_date, ["open", "high", "low", "close", "signal_price"]] = 1_000_000.0
    policy = MomentumPolicy(3)
    a = evaluate_policy_at(policy=policy, full_market=base, decision_date=decision_date, current_exposure=0.2)
    b = evaluate_policy_at(policy=policy, full_market=altered, decision_date=decision_date, current_exposure=0.2)
    assert a.target_exposure == b.target_exposure
    assert dict(a.diagnostics) == dict(b.diagnostics)
