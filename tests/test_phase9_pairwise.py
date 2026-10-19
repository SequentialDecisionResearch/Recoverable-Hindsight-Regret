import numpy as np
import pandas as pd

from trading_policy_oracle_repair.phase9_builder import _pair_frame, _choose_best_candidate
from trading_policy_oracle_repair.uncertainty import PairwiseCalibrator


def _wide():
    idx = pd.MultiIndex.from_arrays([
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
        pd.to_datetime(["2024-03-27", "2024-03-28"]),
        ["AAA", "AAA"], [0.5, 0.5], [60, 60],
    ], names=["date", "label_available_date", "asset_id", "start_exposure", "horizon"])
    actual = pd.DataFrame({"p1": [0.01, .02], "p2": [.03, .01], "p3": [.00, .04]}, index=idx)
    predicted = pd.DataFrame({"p1": [.012, .018], "p2": [.025, .012], "p3": [.005, .035]}, index=idx)
    return actual, predicted


def test_pair_frame_advantage_is_candidate_minus_baseline():
    a, p = _wide()
    f = _pair_frame(a, p, "p1", "p2")
    assert np.allclose(f.actual_advantage, [.02, -.01])
    assert np.allclose(f.predicted_advantage, [.013, -.006])


def test_best_candidate_uses_lcb_not_actual_advantage():
    a, p = _wide()
    c = PairwiseCalibrator(.1, -.002, -.003, 100, .9, .91)
    calibrators = {("p1", "p2"): c, ("p1", "p3"): c}
    out = _choose_best_candidate(a, p, baseline="p1", calibrators=calibrators, method="empirical")
    assert list(out.selected_candidate_policy_id) == ["p2", "p3"]
