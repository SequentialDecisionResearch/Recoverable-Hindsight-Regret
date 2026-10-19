import numpy as np

from trading_policy_oracle_repair.deployment import (
    FrozenFoldRuntime,
    meta_selector_choice,
    selective_repair_choice,
)
from trading_policy_oracle_repair.features import MODEL_FEATURE_COLUMNS
from trading_policy_oracle_repair.preprocessing import FrozenPreprocessor


class Const:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v)


def runtime(q):
    pre = FrozenPreprocessor(tuple(MODEL_FEATURE_COLUMNS), {c:0 for c in MODEL_FEATURE_COLUMNS}, {c:0 for c in MODEL_FEATURE_COLUMNS}, {c:1 for c in MODEL_FEATURE_COLUMNS}, 10)
    return FrozenFoldRuntime("f", "ridge", pre, {"a":Const(0), "b":Const(0)}, q)


def test_selective_repair_uses_lcb_not_raw_argmax():
    rt = runtime({("a","b","empirical"):-0.03, ("b","a","empirical"):-0.03})
    c = selective_repair_choice({"a":0.0,"b":0.02}, baseline_policy_id="a", runtime=rt, lcb_method="empirical", deploy_margin=0.0)
    assert c.selected_policy_id == "a"
    assert not c.deploy_flag
    assert np.isclose(c.lower_bound, -0.01)


def test_meta_selector_penalty_and_lcb_hysteresis():
    rt = runtime({("a","b","empirical"):-0.005, ("b","a","empirical"):-0.005})
    c = meta_selector_choice({"a":0.0,"b":0.03}, current_policy_id="a", runtime=rt, lcb_method="empirical", deploy_margin=0.0, switch_penalty=0.01)
    assert c.selected_policy_id == "b"
    assert c.deploy_flag
    assert np.isclose(c.lower_bound, 0.015)
