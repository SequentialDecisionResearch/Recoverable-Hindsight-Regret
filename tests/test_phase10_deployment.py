import numpy as np
import pandas as pd

from trading_policy_oracle_repair.deployment import FrozenFoldRuntime, run_deployment_path
from trading_policy_oracle_repair.features import BASE_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from trading_policy_oracle_repair.policies import BuyHoldPolicy, MomentumPolicy
from trading_policy_oracle_repair.preprocessing import FrozenPreprocessor


class Const:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v)


def setup():
    dates = pd.bdate_range("2026-01-02", periods=12)
    p = np.linspace(100, 122, len(dates))
    market = pd.DataFrame({"date":dates,"asset_id":"AAA","open":p,"high":p*1.001,"low":p*.999,"close":p,"signal_price":p,"dividend_cash":0.0,"split_factor":1.0,"cash_daily_return":0.0})
    f = pd.DataFrame({"date":dates,"asset_id":"AAA","feature_source_date":dates})
    for c in BASE_FEATURE_COLUMNS: f[c] = 0.0
    pre = FrozenPreprocessor(tuple(MODEL_FEATURE_COLUMNS), {c:0 for c in MODEL_FEATURE_COLUMNS}, {c:0 for c in MODEL_FEATURE_COLUMNS}, {c:1 for c in MODEL_FEATURE_COLUMNS}, 100)
    models = {"buy_hold":Const(.05),"mom_126":Const(0.0)}
    q = {("mom_126","buy_hold","empirical"):0.0,("buy_hold","mom_126","empirical"):-.1}
    rt = FrozenFoldRuntime("toy","ridge",pre,models,q)
    policies = {"buy_hold":BuyHoldPolicy(),"mom_126":MomentumPolicy(lookback=126)}
    ex = list(pd.DatetimeIndex(dates[1:]).normalize())
    return market,f,rt,policies,ex


def test_deployment_api_requires_no_oracle_or_label_artifact():
    market,f,rt,policies,ex = setup()
    # There is intentionally no label/oracle input to this call.
    out = run_deployment_path(asset_market=market, feature_table_asset=f, execution_dates=ex,
        runtime_by_execution_date={d:rt for d in ex}, policies=policies, strategy_type="selective_repair",
        model_name="ridge", lcb_method="empirical", baseline_policy_id="mom_126", initial_policy_id="mom_126",
        initial_wealth=1.0, initial_exposure=0.0, cost_bps_per_side=5.0, deploy_margin=0.0,
        meta_switch_penalty=.0005)
    assert len(out.actions) == len(ex)
    assert out.actions["deploy_flag"].all()
    assert out.performance["final_wealth"] > 1.0


def test_same_input_is_deterministic():
    market,f,rt,policies,ex = setup()
    kwargs=dict(asset_market=market, feature_table_asset=f, execution_dates=ex,
        runtime_by_execution_date={d:rt for d in ex}, policies=policies, strategy_type="meta_selector",
        model_name="ridge", lcb_method="empirical", baseline_policy_id=None, initial_policy_id="buy_hold",
        initial_wealth=1.0, initial_exposure=0.0, cost_bps_per_side=5.0, deploy_margin=0.0,
        meta_switch_penalty=.0005)
    a=run_deployment_path(**kwargs).actions
    b=run_deployment_path(**kwargs).actions
    pd.testing.assert_frame_equal(a,b)
